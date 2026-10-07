"""Email (IMAP/SMTP) channel adapter: polls IMAP for new
messages, normalizes them, and routes them through the shared dispatch
pipeline (app/channels/dispatch.py) — same shape as the Telegram
adapter (app/channels/telegram.py), IMAP/SMTP instead of the Bot API.

The mailbox is shared with ordinary mail (website contact form,
customer questions), so the adapter only touches messages whose subject
contains EMAIL_TRIGGER_TAG. Every other message is never fetched,
flagged or answered: it stays unread for a human. Messages are read with
BODY.PEEK, which sets no flag. Once the turn has succeeded (or the sender
is unknown, which is not retried) the message is marked Seen and filed into
EMAIL_AGENT_FOLDER, so it leaves the INBOX humans read. If the turn fails
it stays unread and is tried again on the next poll, up to MAX_ATTEMPTS,
then it is filed into "<folder>.Failed". Nothing is ever deleted or
purged automatically.

IMAP/SMTP calls are synchronous (stdlib imaplib/smtplib) — wrapped in
asyncio.to_thread so a slow mail server doesn't block the event loop
the Telegram adapter and Admin API also run in.
"""

import asyncio
import email
import email.message
import email.utils
import imaplib
import logging
import re
import secrets
import smtplib
from dataclasses import dataclass
from email.header import decode_header
from email.mime.text import MIMEText

from app import health
from app.channels import confirmations
from app.channels.dispatch import DispatchOutcome, dispatch_event
from app.channels.schema import NormalizedEvent
from app.config import get_settings
from app.db.models import Channel
from app.db.session import session_scope

logger = logging.getLogger("channelagent")

POLL_INTERVAL_SECONDS = 15
# The container is reported unhealthy when no poll completed for this long.
LIVENESS_MAX_AGE_SECONDS = 300


def _decode(value: str) -> str:
    parts = decode_header(value)
    return "".join(
        part.decode(enc or "utf-8", errors="replace") if isinstance(part, bytes) else str(part)
        for part, enc in parts
    )


def _extract_body(msg: "email.message.Message") -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                return payload.decode(charset, errors="replace") if payload else ""
        return ""
    charset = msg.get_content_charset() or "utf-8"
    payload = msg.get_payload(decode=True)
    return payload.decode(charset, errors="replace") if payload else ""


def _tag_is_usable(tag: str) -> bool:
    """The tag goes into an IMAP SEARCH string, so it must be plain
    ASCII with nothing that needs escaping. An empty tag is unusable on
    purpose: it would match every message in the shared mailbox.
    """
    return bool(tag.strip()) and tag.isascii() and '"' not in tag and "\\" not in tag


def _subject_has_tag(subject: str, tag: str) -> bool:
    return _tag_is_usable(tag) and tag.lower() in subject.lower()


def _folder_is_usable(folder: str) -> bool:
    """The folder name goes into IMAP commands unescaped, so it must be
    plain ASCII without quotes, backslashes or wildcards. Empty is valid
    and means "do not move handled messages".
    """
    return not folder.strip() or (
        folder.isascii() and not any(c in folder for c in '"\\*%')
    )


def _ensure_folder(imap: imaplib.IMAP4, folder: str) -> bool:
    """Create the agent folder if it does not exist yet, and subscribe to
    it so webmail clients list it. Returns whether it exists afterwards.
    """
    typ, data = imap.list(pattern=f'"{folder}"')
    exists = typ == "OK" and any(item for item in data)
    if not exists:
        typ, _ = imap.create(folder)
        if typ != "OK":
            return False
    imap.subscribe(folder)
    return True


def _capabilities(imap: imaplib.IMAP4) -> tuple[str, ...]:
    """What the server supports after login. Under Python 3.12 (the image),
    `imap.capabilities` keeps the list read before authentication, which has neither MOVE
    nor UIDPLUS, so no message was ever filed (found on the live mailbox, 2026-09-21).
    Python 3.14 refreshes it. When the client's list lacks both, ask the server.
    """
    caps = tuple(getattr(imap, "capabilities", ()) or ())
    if not {"MOVE", "UIDPLUS"} & set(caps):
        try:
            typ, data = imap.capability()
            if typ == "OK" and data and data[0]:
                caps = tuple(data[0].decode().upper().split())
        except (imaplib.IMAP4.error, OSError):
            logger.debug("CAPABILITY failed, using the client's list", exc_info=True)
    return caps


def _move_message(imap: imaplib.IMAP4, uid: bytes, folder: str) -> bool:
    """File one message, addressed by UID, into `folder`.

    Prefers MOVE. Without it, COPY then flag Deleted then UID EXPUNGE,
    which only removes this UID. A plain EXPUNGE is never used: it would
    also remove any message a human client flagged Deleted but has not
    expunged yet. Without UIDPLUS (needed for UID EXPUNGE) nothing is
    moved at all.
    """
    caps = _capabilities(imap)
    if "MOVE" in caps and "MOVE" in imaplib.Commands:
        typ, _ = imap.uid("MOVE", uid, folder)
        return typ == "OK"
    if "UIDPLUS" in caps:
        typ, _ = imap.uid("COPY", uid, folder)
        if typ != "OK":
            return False
        imap.uid("STORE", uid, "+FLAGS", "\\Deleted")
        imap.uid("EXPUNGE", uid)
        return True
    return False


@dataclass(frozen=True)
class TaggedMessage:
    uid: bytes
    from_addr: str
    subject: str
    body: str


# A message whose turn fails is left unread in the INBOX and tried again on
# the next poll, at most this many times, then filed into the "failed"
# folder so a poison message cannot loop for ever. Counted in memory:
# a restart gives every message a fresh set of attempts.
MAX_ATTEMPTS = 3
_attempts: dict[bytes, int] = {}


def failed_folder(folder: str) -> str:
    """Where a message goes after MAX_ATTEMPTS failures. Empty when messages
    are not being filed at all: it is then only marked Seen.
    """
    return f"{folder.strip()}.Failed" if folder.strip() else ""


def _open_inbox() -> imaplib.IMAP4:
    settings = get_settings()
    imap = imaplib.IMAP4_SSL(settings.email_imap_host, settings.email_imap_port, timeout=10)
    try:
        imap.login(settings.email_username, settings.email_password)
        imap.select("INBOX")
    except Exception:
        _logout(imap)
        raise
    return imap


def _logout(imap: imaplib.IMAP4) -> None:
    try:
        imap.logout()
    except Exception:
        logger.debug("IMAP logout raised, ignoring", exc_info=True)


def _mark_seen_and_file(imap: imaplib.IMAP4, uid: bytes, folder: str) -> None:
    """Mark one message (by UID) Seen and, when `folder` is set, move it
    there. A failed move is logged and leaves the message in the INBOX,
    already flagged Seen.
    """
    imap.uid("STORE", uid, "+FLAGS", "\\Seen")
    if not folder:
        return
    try:
        if not _ensure_folder(imap, folder) or not _move_message(imap, uid, folder):
            logger.warning(
                "Could not move message %s to %r, it stays in INBOX", uid.decode(), folder
            )
    except imaplib.IMAP4.error:
        logger.warning(
            "IMAP error moving message %s to %r, it stays in INBOX",
            uid.decode(), folder, exc_info=True,
        )


def _is_automated(msg: email.message.Message) -> bool:
    """Machine-generated mail (RFC 3834 and the common bulk markers): an
    `Auto-Submitted` value other than `no`, or `Precedence: bulk|list|junk`.
    Such a message is never a person asking for access, and answering it can
    start a reply loop with another robot.
    """
    auto = str(msg.get("Auto-Submitted", "")).split(";")[0].strip().lower()
    if auto and auto != "no":
        return True
    return str(msg.get("Precedence", "")).strip().lower() in ("bulk", "list", "junk")


def _fetch_tagged_unseen(tag: str, folder: str = "") -> list[TaggedMessage]:
    """Sync IMAP work: connect, find UNSEEN messages whose subject carries
    the trigger tag, read only those (BODY.PEEK, so reading sets no flag)
    and disconnect. Returns the messages that have something to answer.

    Nothing is flagged or moved here for those: that happens in
    `_finalize_message`, once the turn has succeeded, so a failed turn
    leaves the message unread for another attempt. A tagged message with no
    sender or an empty body has nothing to answer and would be fetched again
    on every poll, so it is filed straight away.

    Everything is addressed by UID, not by message number: a move renumbers
    the messages that follow it in the same session. The server-side SEARCH
    narrows the set, and the subject is checked again here because servers
    differ in how they match SUBJECT: a message that fails that check is
    never touched.
    """
    if not _tag_is_usable(tag):
        return []
    if not _folder_is_usable(folder):
        logger.error("EMAIL_AGENT_FOLDER %r is not usable, handled mail stays in INBOX", folder)
        folder = ""
    folder = folder.strip()
    imap = _open_inbox()
    results: list[TaggedMessage] = []
    try:
        typ, data = imap.uid("SEARCH", "UNSEEN", "SUBJECT", f'"{tag}"')
        if typ == "OK":
            for uid in data[0].split():
                typ, msg_data = imap.uid("FETCH", uid, "(BODY.PEEK[])")
                if typ != "OK" or not msg_data or not msg_data[0]:
                    continue
                msg = email.message_from_bytes(msg_data[0][1])
                subject = _decode(msg.get("Subject", ""))
                if not _subject_has_tag(subject, tag):
                    continue
                if _is_automated(msg):
                    logger.info(
                        "Ignoring automated mail (uid %s): no request, no reply", uid.decode()
                    )
                    _mark_seen_and_file(imap, uid, folder)
                    continue
                from_addr = email.utils.parseaddr(msg.get("From", ""))[1]
                body = _extract_body(msg).strip()
                if from_addr and body:
                    results.append(TaggedMessage(uid, from_addr, subject, body))
                else:
                    _mark_seen_and_file(imap, uid, folder)
    finally:
        _logout(imap)
    return results


def _finalize_message(uid: bytes, folder: str) -> None:
    """Sync IMAP work: mark one handled message Seen and file it."""
    if not _folder_is_usable(folder):
        folder = ""
    imap = _open_inbox()
    try:
        _mark_seen_and_file(imap, uid, folder.strip())
    finally:
        _logout(imap)


def _send_reply_sync(to_address: str, subject: str, body: str) -> None:
    settings = get_settings()
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = settings.email_username
    msg["To"] = to_address
    # RFC 3834: tells other systems not to answer this automatic reply.
    msg["Auto-Submitted"] = "auto-replied"
    with smtplib.SMTP_SSL(settings.email_smtp_host, settings.email_smtp_port, timeout=10) as smtp:
        smtp.login(settings.email_username, settings.email_password)
        smtp.send_message(msg)


# --- tool confirmations ---
# The question is sent with a random code in its subject. A From address can be forged,
# the code cannot be guessed: only a reply that carries the code and comes from the
# address the question was sent to answers it. The mailbox is read for that code while
# the turn waits (the regular poll runs one turn at a time and would not see it).
CONFIRM_POLL_SECONDS = 5
_CONFIRM_SUBJECT = re.compile(r"Confirm ([0-9a-f]{8})\b")


def _find_confirmation_sync(code: str, from_addr: str, folder: str) -> bool | None:
    """Sync IMAP work: the answer in an unread reply carrying `code` from `from_addr`
    (True yes, False no), filed like a handled message; None when there is none yet."""
    if not _folder_is_usable(folder):
        folder = ""
    imap = _open_inbox()
    try:
        typ, data = imap.uid("SEARCH", "UNSEEN", "SUBJECT", f'"Confirm {code}"')
        if typ != "OK":
            return None
        for uid in data[0].split():
            typ, msg_data = imap.uid("FETCH", uid, "(BODY.PEEK[])")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            msg = email.message_from_bytes(msg_data[0][1])
            sender = email.utils.parseaddr(msg.get("From", ""))[1]
            if f"Confirm {code}" not in _decode(msg.get("Subject", "")):
                continue
            if sender.lower() != from_addr.lower():
                logger.warning("A confirmation reply came from another address: ignored")
                continue
            lines = [line for line in _extract_body(msg).splitlines() if line.strip()]
            answer = confirmations.parse_answer(lines[0]) if lines else None
            if answer is None:
                continue
            _mark_seen_and_file(imap, uid, folder.strip())
            return answer
    finally:
        _logout(imap)
    return None


async def confirm_by_email(
    from_addr: str, subject: str, question: str, timeout: float
) -> bool | None:
    settings = get_settings()
    code = secrets.token_hex(4)
    body = f"{question}\n\nReply to this message with yes or no on the first line."
    await asyncio.to_thread(
        _send_reply_sync,
        from_addr,
        f"{settings.email_trigger_tag} Confirm {code}: {subject}",
        body,
    )
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            answer = await asyncio.to_thread(
                _find_confirmation_sync, code, from_addr, settings.email_agent_folder
            )
        except Exception:
            logger.warning("Reading the mailbox for a confirmation failed", exc_info=True)
            answer = None
        if answer is not None:
            return answer
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(CONFIRM_POLL_SECONDS, remaining))


async def _poll_once() -> None:
    settings = get_settings()
    folder = settings.email_agent_folder
    messages = await asyncio.to_thread(_fetch_tagged_unseen, settings.email_trigger_tag, folder)
    health.mark("email")  # the mailbox answered: the poll is not stuck on the network
    for message in messages:
        logger.info("Email received from %s: %s", message.from_addr, message.subject)

        async def reply(text: str, _msg: TaggedMessage = message) -> None:
            await asyncio.to_thread(
                _send_reply_sync, _msg.from_addr, f"Re: {_msg.subject}", text
            )

        if _CONFIRM_SUBJECT.search(message.subject):
            # A reply to a confirmation question that came too late, or twice: the
            # question is gone, and the answer is not a message for the agent.
            logger.info("Late reply to a tool confirmation from %s: filed", message.from_addr)
            try:
                await asyncio.to_thread(_finalize_message, message.uid, folder)
            except Exception:
                logger.exception("Could not file the late confirmation reply")
            continue

        async def confirm(
            question: str, timeout: float, _msg: TaggedMessage = message
        ) -> bool | None:
            return await confirm_by_email(_msg.from_addr, _msg.subject, question, timeout)

        event = NormalizedEvent(
            user_id=message.from_addr,
            channel=Channel.EMAIL,
            text=message.body,
            reply=reply,
            confirm=confirm,
        )
        attempts = _attempts.get(message.uid, 0)
        try:
            # The turn may wait for the engine up to the task limit per request, and a
            # turn can make several: a reply asked again without thinking, tool rounds (live,
            # 2026-10-04: 1025 s for two requests, the heartbeat stopped at 900). Three times the
            # limit, then the adapter counts as stuck.
            with health.working("email", 3 * settings.llm_task_timeout_seconds):
                async with session_scope() as session:
                    outcome = await dispatch_event(
                        session, event, apologize=False, retry=attempts > 0
                    )
        except Exception:
            logger.exception("Handling the email from %s failed", message.from_addr)
            outcome = DispatchOutcome.FAILED

        if outcome == DispatchOutcome.LIMITED:
            # Over the sender's limit: left unread, taken again at a later poll; no
            # reply by email, no attempt counted.
            continue
        if outcome in (DispatchOutcome.FAILED, DispatchOutcome.UNDELIVERED):
            attempts += 1
            _attempts[message.uid] = attempts
            if attempts < MAX_ATTEMPTS:
                logger.warning(
                    "Email from %s: attempt %s of %s failed, left unread in INBOX for a retry",
                    message.from_addr, attempts, MAX_ATTEMPTS,
                )
                continue
            logger.error(
                "Email from %s: giving up after %s attempts, filing it as failed",
                message.from_addr, MAX_ATTEMPTS,
            )
            target = failed_folder(folder)
        else:
            target = folder
        _attempts.pop(message.uid, None)
        try:
            await asyncio.to_thread(_finalize_message, message.uid, target)
        except Exception:
            logger.exception("Could not mark the email from %s as handled", message.from_addr)


async def run_email_adapter() -> None:
    """Runs until cancelled — polls IMAP every POLL_INTERVAL_SECONDS."""
    tag = get_settings().email_trigger_tag
    if not _tag_is_usable(tag):
        logger.error(
            "Email adapter NOT started: EMAIL_TRIGGER_TAG %r is empty or not plain ASCII "
            "without quotes/backslashes. The mailbox is shared with ordinary mail, so the "
            "adapter refuses to run without a usable tag.",
            tag,
        )
        return
    folder = get_settings().email_agent_folder
    logger.info(
        "Email adapter started (polling every %ss, only subjects containing %r, "
        "handled mail filed into %s).",
        POLL_INTERVAL_SECONDS,
        tag,
        repr(folder) if folder.strip() else "nowhere (stays in INBOX)",
    )
    health.register("email", LIVENESS_MAX_AGE_SECONDS)
    try:
        while True:
            try:
                await _poll_once()
                health.mark("email")
            except Exception:
                logger.exception("Email poll failed")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
    finally:
        health.unregister("email")
