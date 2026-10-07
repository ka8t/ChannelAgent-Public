"""Tests: email parsing helpers (offline, real email.message
objects, no network) and dispatch wiring via Channel.EMAIL (mock LLM,
reusing the pattern already proven for Telegram).
"""

import email
import json
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.channels import email as email_adapter
from app.channels.email import (
    TaggedMessage,
    _decode,
    _extract_body,
    _fetch_tagged_unseen,
    _finalize_message,
)


def test_decode_plain_ascii_subject():
    assert _decode("Hello world") == "Hello world"


def test_decode_rfc2047_encoded_subject():
    # A real MIME-encoded header, as a non-ASCII subject actually arrives.
    encoded = "=?utf-8?b?QsOgIGVzc2Fp?="  # "Bà essai"
    assert _decode(encoded) == "Bà essai"


def test_extract_body_plain_text_message():
    msg = MIMEText("hello from a plain message", "plain", "utf-8")
    parsed = email.message_from_bytes(msg.as_bytes())
    assert _extract_body(parsed).strip() == "hello from a plain message"


def test_extract_body_multipart_prefers_plain_text_part():
    msg = MIMEMultipart("alternative")
    msg.attach(MIMEText("plain version", "plain", "utf-8"))
    msg.attach(MIMEText("<p>html version</p>", "html", "utf-8"))
    parsed = email.message_from_bytes(msg.as_bytes())
    assert _extract_body(parsed).strip() == "plain version"


class _MockLlama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":  # no tokenizer on this mock
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        resp = {"choices": [{"message": {"role": "assistant", "content": "email reply"}}]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.mark.asyncio
async def test_dispatch_event_over_email_channel(fresh_db, monkeypatch):
    server = HTTPServer(("127.0.0.1", 8095), _MockLlama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:8095")
    from app import config

    config.get_settings.cache_clear()

    from app.channels.dispatch import dispatch_event
    from app.channels.schema import NormalizedEvent
    from app.db.models import Channel, ChannelIdentity, PermissionKind, User
    from app.db.session import init_db, session_scope
    from app.security.auth import grant_permission
    from app.security.hashing import hash_email

    await init_db()
    async with session_scope() as session:
        user = User(display_name="Email test")
        session.add(user)
        await session.flush()
        identity = ChannelIdentity(
            user_id=user.id,
            channel=Channel.EMAIL,
            external_id=hash_email("sender@example.com"),
            raw_address="sender@example.com",
        )
        session.add(identity)
        await session.flush()
        await grant_permission(session, identity, PermissionKind.CHAT)
        await session.commit()

    sent = []

    async def reply(text: str) -> None:
        sent.append(text)

    async with session_scope() as session:
        event = NormalizedEvent(
            user_id="sender@example.com", channel=Channel.EMAIL, text="hi", reply=reply
        )
        await dispatch_event(session, event)

    assert sent == ["email reply"]
    server.shutdown()
    config.get_settings.cache_clear()


# --- Shared mailbox: only messages carrying the trigger tag are touched ---


def _raw(sender: str, subject: str, body: str) -> bytes:
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = sender
    msg["Subject"] = subject
    return msg.as_bytes()


class _FakeIMAP:
    """Records every command so a test can count what was fetched,
    flagged, created and moved. Messages are addressed by UID, like the
    adapter does.

    search_ignores_subject simulates a server that does not honour
    SEARCH SUBJECT, so the client-side check is what protects.
    capabilities controls whether MOVE / UIDPLUS are advertised.
    """

    def __init__(
        self,
        messages: dict[bytes, bytes],
        search_ignores_subject: bool = False,
        capabilities: tuple[str, ...] = ("IMAP4REV1", "MOVE", "UIDPLUS"),
        folders: set[str] | None = None,
        create_fails: bool = False,
    ):
        self.messages = dict(messages)
        self.search_ignores_subject = search_ignores_subject
        self.capabilities = capabilities
        self.folders = set(folders or {"INBOX"})
        self.create_fails = create_fails
        self.fetches: list[tuple[bytes, str]] = []
        self.stores: list[tuple[bytes, str, str]] = []
        self.moves: list[tuple[bytes, str]] = []
        self.copies: list[tuple[bytes, str]] = []
        self.uid_expunges: list[bytes] = []
        self.plain_expunges = 0
        self.creates: list[str] = []
        self.subscribes: list[str] = []
        self.lists = 0
        self.filed: dict[str, list[bytes]] = {}

    def login(self, *a):
        return "OK", []

    def capability(self):
        return "OK", [" ".join(self.capabilities).encode()]

    def select(self, *a):
        return "OK", [b"1"]

    def list(self, directory='""', pattern="*"):
        self.lists += 1
        name = pattern.strip('"')
        if name in self.folders:
            return "OK", [f'(\\HasNoChildren) "." "{name}"'.encode()]
        return "OK", [None]

    def create(self, folder):
        self.creates.append(folder)
        if self.create_fails:
            return "NO", [b"permission denied"]
        self.folders.add(folder)
        return "OK", [b"created"]

    def subscribe(self, folder):
        self.subscribes.append(folder)
        return "OK", []

    def expunge(self):
        self.plain_expunges += 1
        return "OK", []

    def uid(self, command, *args):
        command = command.upper()
        if command == "SEARCH":
            uids = list(self.messages)
            if "SUBJECT" in args and not self.search_ignores_subject:
                wanted = args[args.index("SUBJECT") + 1].strip('"').lower()
                uids = [
                    u for u in uids
                    if wanted in email.message_from_bytes(self.messages[u])["Subject"].lower()
                ]
            return "OK", [b" ".join(uids)]
        if command == "FETCH":
            uid, spec = args
            self.fetches.append((uid, spec))
            return "OK", [(b"1 (UID 1 BODY[] {0}", self.messages[uid]), b")"]
        if command == "STORE":
            uid, cmd, flags = args
            self.stores.append((uid, cmd, flags))
            return "OK", []
        if command == "MOVE":
            uid, folder = args
            self.moves.append((uid, folder))
            self.messages.pop(uid, None)
            self.filed.setdefault(folder, []).append(uid)
            return "OK", []
        if command == "COPY":
            uid, folder = args
            self.copies.append((uid, folder))
            self.filed.setdefault(folder, []).append(uid)
            return "OK", []
        if command == "EXPUNGE":
            (uid,) = args
            self.uid_expunges.append(uid)
            self.messages.pop(uid, None)
            return "OK", []
        raise AssertionError(f"unexpected UID command {command}")

    def logout(self):
        return "BYE", []


@pytest.fixture
def fake_imap(monkeypatch):
    holder = {"connections": 0}

    def install(messages, **kwargs):
        fake = _FakeIMAP(messages, **kwargs)

        def factory(*a, **k):
            holder["connections"] += 1
            return fake

        monkeypatch.setattr(email_adapter.imaplib, "IMAP4_SSL", factory)
        monkeypatch.setattr(
            email_adapter,
            "get_settings",
            lambda: type("S", (), {
                "email_imap_host": "h", "email_imap_port": 993,
                "email_username": "u", "email_password": "p",
            })(),
        )
        return fake

    install.holder = holder
    return install


CUSTOMER = ("customer@example.com", "Question about your offer", "not for the bot")
TAGGED = ("sender@example.com", "Re: [Agent] hello", "for the bot")


def test_untagged_messages_are_never_fetched_flagged_or_moved(fake_imap):
    fake = fake_imap({
        b"11": _raw(*CUSTOMER),
        b"12": _raw("other@example.com", "Contact form", "hi"),
    })
    assert _fetch_tagged_unseen("[agent]", "INBOX.Agent") == []
    assert fake.fetches == [] and fake.stores == []
    assert fake.moves == [] and fake.creates == [] and fake.lists == 0


def test_reading_a_tagged_message_flags_and_moves_nothing(fake_imap):
    """The message is only read here. It is flagged and filed by
    _finalize_message once the turn has succeeded, so a failed turn leaves
    it unread for another attempt.
    """
    fake = fake_imap({b"11": _raw(*CUSTOMER), b"12": _raw(*TAGGED)})
    result = _fetch_tagged_unseen("[agent]", "INBOX.Agent")
    assert result == [
        TaggedMessage(b"12", "sender@example.com", "Re: [Agent] hello", "for the bot")
    ]
    assert [u for u, _ in fake.fetches] == [b"12"]
    assert all("PEEK" in spec for _, spec in fake.fetches)
    assert fake.stores == [] and fake.moves == [] and fake.creates == []
    assert set(fake.messages) == {b"11", b"12"}, "both messages are still in the INBOX"


def test_client_side_check_protects_when_server_ignores_subject_search(fake_imap):
    fake = fake_imap(
        {b"11": _raw(*CUSTOMER), b"12": _raw("s@example.com", "[agent] hello", "for the bot")},
        search_ignores_subject=True,
    )
    result = _fetch_tagged_unseen("[agent]", "INBOX.Agent")
    assert [m.from_addr for m in result] == ["s@example.com"]
    assert all("PEEK" in spec for _, spec in fake.fetches)
    assert fake.stores == [] and fake.moves == []


@pytest.mark.parametrize(
    ("sender", "body"),
    [("", "no sender"), ("sender@example.com", ""), ("sender@example.com", "   \n ")],
)
def test_tagged_message_with_nothing_to_answer_is_filed_straight_away(fake_imap, sender, body):
    raw = _raw(sender, "[agent] hello", body) if sender else _raw("", "[agent] hello", body)
    fake = fake_imap({b"12": raw})
    assert _fetch_tagged_unseen("[agent]", "INBOX.Agent") == []
    assert fake.stores == [(b"12", "+FLAGS", "\\Seen")]
    assert fake.moves == [(b"12", "INBOX.Agent")]


@pytest.mark.parametrize("tag", ["", "   ", 'a"b', "a\\b", "é-tag"])
def test_unusable_tag_processes_nothing_and_never_connects(fake_imap, tag):
    fake = fake_imap({b"12": _raw("sender@example.com", "anything", "body")})
    assert _fetch_tagged_unseen(tag, "INBOX.Agent") == []
    assert fake_imap.holder["connections"] == 0
    assert fake.fetches == [] and fake.stores == [] and fake.moves == []


# --- _finalize_message: mark Seen and file, after a successful turn ---


def test_finalize_marks_seen_and_moves_only_that_uid(fake_imap):
    fake = fake_imap({b"11": _raw(*CUSTOMER), b"12": _raw(*TAGGED)})
    _finalize_message(b"12", "INBOX.Agent")
    assert fake.stores == [(b"12", "+FLAGS", "\\Seen")]
    assert fake.moves == [(b"12", "INBOX.Agent")]
    assert fake.creates == ["INBOX.Agent"] and fake.subscribes == ["INBOX.Agent"]
    assert list(fake.messages) == [b"11"], "the customer mail must still be in the INBOX"
    assert fake.plain_expunges == 0


def test_finalize_does_not_recreate_an_existing_folder(fake_imap):
    fake = fake_imap({b"12": _raw(*TAGGED)}, folders={"INBOX", "INBOX.Agent"})
    _finalize_message(b"12", "INBOX.Agent")
    assert fake.creates == []
    assert fake.moves == [(b"12", "INBOX.Agent")]


def test_finalize_without_move_falls_back_to_copy_flag_and_uid_expunge(fake_imap):
    fake = fake_imap(
        {b"11": _raw(*CUSTOMER), b"12": _raw(*TAGGED)},
        capabilities=("IMAP4REV1", "UIDPLUS"),
    )
    _finalize_message(b"12", "INBOX.Agent")
    assert fake.moves == []
    assert fake.copies == [(b"12", "INBOX.Agent")]
    assert (b"12", "+FLAGS", "\\Deleted") in fake.stores
    assert fake.uid_expunges == [b"12"]
    assert fake.plain_expunges == 0
    assert b"11" in fake.messages


def test_finalize_without_move_or_uidplus_only_marks_seen(fake_imap):
    fake = fake_imap({b"12": _raw(*TAGGED)}, capabilities=("IMAP4REV1",))
    _finalize_message(b"12", "INBOX.Agent")
    assert fake.stores == [(b"12", "+FLAGS", "\\Seen")]
    assert fake.moves == [] and fake.copies == []
    assert fake.uid_expunges == [] and fake.plain_expunges == 0


def test_finalize_folder_creation_failure_leaves_it_seen_in_the_inbox(fake_imap):
    fake = fake_imap({b"12": _raw(*TAGGED)}, create_fails=True)
    _finalize_message(b"12", "INBOX.Agent")
    assert fake.stores == [(b"12", "+FLAGS", "\\Seen")]
    assert fake.moves == [] and fake.copies == []
    assert b"12" in fake.messages


@pytest.mark.parametrize("folder", ["", "   ", 'a"b', "a\\b", "In*box", "Bo%x", "dossié"])
def test_finalize_with_no_or_unusable_folder_only_marks_seen(fake_imap, folder):
    fake = fake_imap({b"12": _raw(*TAGGED)})
    _finalize_message(b"12", folder)
    assert fake.stores == [(b"12", "+FLAGS", "\\Seen")]
    assert fake.lists == 0 and fake.creates == [] and fake.moves == [] and fake.copies == []


def test_finalize_imap_error_while_moving_does_not_raise(fake_imap):
    fake = fake_imap({b"12": _raw(*TAGGED)})
    original = fake.uid

    def uid(command, *args):
        if command.upper() == "MOVE":
            raise email_adapter.imaplib.IMAP4.error("boom")
        return original(command, *args)

    fake.uid = uid
    _finalize_message(b"12", "INBOX.Agent")
    assert fake.stores == [(b"12", "+FLAGS", "\\Seen")]
    assert b"12" in fake.messages


def test_failed_folder_name():
    assert email_adapter.failed_folder("INBOX.Agent") == "INBOX.Agent.Failed"
    assert email_adapter.failed_folder("  INBOX.Agent ") == "INBOX.Agent.Failed"
    assert email_adapter.failed_folder("") == ""
    assert email_adapter.failed_folder("   ") == ""


# --- _poll_once: retries after a failed turn ---


class _PollHarness:
    """Drives _poll_once with scripted dispatch outcomes and records what it
    calls, without a mailbox or a database.
    """

    def __init__(self, monkeypatch, folder="INBOX.Agent"):
        from contextlib import asynccontextmanager

        self.outcomes: list = []
        self.dispatch_calls: list[dict] = []
        self.finalized: list[tuple[bytes, str]] = []
        self.messages: list[TaggedMessage] = []
        email_adapter._attempts.clear()

        @asynccontextmanager
        async def fake_session_scope():
            yield object()

        async def fake_dispatch(session, event, *, apologize=True, retry=False):
            self.dispatch_calls.append(
                {"user": event.user_id, "apologize": apologize, "retry": retry}
            )
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(email_adapter, "session_scope", fake_session_scope)
        monkeypatch.setattr(email_adapter, "dispatch_event", fake_dispatch)
        monkeypatch.setattr(
            email_adapter, "_fetch_tagged_unseen", lambda tag, folder="": list(self.messages)
        )
        monkeypatch.setattr(
            email_adapter, "_finalize_message", lambda uid, f: self.finalized.append((uid, f))
        )
        monkeypatch.setattr(
            email_adapter,
            "get_settings",
            lambda: type("S", (), {
                "email_trigger_tag": "[agent]", "email_agent_folder": folder,
                "llm_task_timeout_seconds": 600,
            })(),
        )


def _msg(uid=b"7", sender="sender@example.com"):
    return TaggedMessage(uid, sender, "[agent] hi", "question")


@pytest.mark.parametrize("outcome", ["ok", "denied"])
async def test_poll_files_the_message_after_a_handled_turn(monkeypatch, outcome):
    from app.channels.dispatch import DispatchOutcome

    h = _PollHarness(monkeypatch)
    h.messages = [_msg()]
    h.outcomes = [DispatchOutcome(outcome)]
    await email_adapter._poll_once()
    assert h.finalized == [(b"7", "INBOX.Agent")]
    assert h.dispatch_calls == [{"user": "sender@example.com", "apologize": False, "retry": False}]
    assert email_adapter._attempts == {}


async def test_poll_leaves_a_failed_message_unread_for_a_retry(monkeypatch):
    from app.channels.dispatch import DispatchOutcome

    h = _PollHarness(monkeypatch)
    h.messages = [_msg()]
    h.outcomes = [DispatchOutcome.FAILED]
    await email_adapter._poll_once()
    assert h.finalized == [], "a failed turn must not mark or move the message"
    assert email_adapter._attempts == {b"7": 1}


async def test_poll_treats_an_undelivered_answer_like_a_failure_to_retry(monkeypatch):
    """The answer is kept and only the delivery is retried, so the message
    stays unread and counts an attempt, then is filed after the third one.
    """
    from app.channels.dispatch import DispatchOutcome

    h = _PollHarness(monkeypatch)
    h.messages = [_msg()]
    h.outcomes = [DispatchOutcome.UNDELIVERED]
    await email_adapter._poll_once()
    assert h.finalized == []
    assert email_adapter._attempts == {b"7": 1}

    h.outcomes = [DispatchOutcome.UNDELIVERED, DispatchOutcome.UNDELIVERED]
    await email_adapter._poll_once()
    await email_adapter._poll_once()
    assert h.finalized == [(b"7", email_adapter.failed_folder("INBOX.Agent"))]
    assert [c["retry"] for c in h.dispatch_calls] == [False, True, True]


async def test_poll_retries_then_files_the_message_after_a_success(monkeypatch):
    from app.channels.dispatch import DispatchOutcome

    h = _PollHarness(monkeypatch)
    h.messages = [_msg()]
    h.outcomes = [DispatchOutcome.FAILED, DispatchOutcome.OK]
    await email_adapter._poll_once()
    await email_adapter._poll_once()
    assert h.finalized == [(b"7", "INBOX.Agent")]
    assert [c["retry"] for c in h.dispatch_calls] == [False, True], "the inbound is logged once"
    assert email_adapter._attempts == {}


async def test_poll_gives_up_after_three_failures_and_files_it_as_failed(monkeypatch):
    from app.channels.dispatch import DispatchOutcome

    h = _PollHarness(monkeypatch)
    h.messages = [_msg()]
    h.outcomes = [DispatchOutcome.FAILED] * 3
    for _ in range(3):
        await email_adapter._poll_once()
    assert email_adapter.MAX_ATTEMPTS == 3
    assert h.finalized == [(b"7", "INBOX.Agent.Failed")]
    assert [c["retry"] for c in h.dispatch_calls] == [False, True, True]
    assert all(c["apologize"] is False for c in h.dispatch_calls), "no apology email"
    assert email_adapter._attempts == {}


async def test_poll_after_three_failures_without_a_folder_only_marks_seen(monkeypatch):
    from app.channels.dispatch import DispatchOutcome

    h = _PollHarness(monkeypatch, folder="")
    h.messages = [_msg()]
    h.outcomes = [DispatchOutcome.FAILED] * 3
    for _ in range(3):
        await email_adapter._poll_once()
    assert h.finalized == [(b"7", "")]


async def test_poll_an_exception_in_one_message_does_not_lose_the_others(monkeypatch):
    from app.channels.dispatch import DispatchOutcome

    h = _PollHarness(monkeypatch)
    h.messages = [_msg(b"7", "a@example.com"), _msg(b"8", "b@example.com")]
    h.outcomes = [RuntimeError("database is locked"), DispatchOutcome.OK]
    await email_adapter._poll_once()
    assert h.finalized == [(b"8", "INBOX.Agent")]
    assert email_adapter._attempts == {b"7": 1}, "the first one will be retried"


async def test_poll_keeps_going_when_finalizing_fails(monkeypatch):
    from app.channels.dispatch import DispatchOutcome

    h = _PollHarness(monkeypatch)
    h.messages = [_msg(b"7", "a@example.com"), _msg(b"8", "b@example.com")]
    h.outcomes = [DispatchOutcome.OK, DispatchOutcome.OK]

    def flaky(uid, folder):
        if uid == b"7":
            raise OSError("connection reset")
        h.finalized.append((uid, folder))

    monkeypatch.setattr(email_adapter, "_finalize_message", flaky)
    await email_adapter._poll_once()
    assert h.finalized == [(b"8", "INBOX.Agent")]


def test_default_trigger_tag_and_folder(monkeypatch):
    monkeypatch.delenv("EMAIL_TRIGGER_TAG", raising=False)
    monkeypatch.delenv("EMAIL_AGENT_FOLDER", raising=False)
    from app.config import Settings

    settings = Settings(_env_file=None)
    assert settings.email_trigger_tag == "[agent]"
    assert settings.email_agent_folder == "INBOX.Agent"


@pytest.mark.asyncio
async def test_unknown_email_sender_gets_a_request_but_no_reply(fresh_db):
    from sqlalchemy import func, select

    from app.channels.dispatch import dispatch_event
    from app.channels.schema import NormalizedEvent
    from app.db.models import AccessRequest, Channel
    from app.db.session import init_db, session_scope

    await init_db()
    sent: list[str] = []

    async def reply(text: str) -> None:
        sent.append(text)

    async with session_scope() as session:
        await dispatch_event(
            session, NormalizedEvent("stranger@example.com", Channel.EMAIL, "let me in", reply)
        )
    async with session_scope() as session:
        n = (await session.execute(select(func.count()).select_from(AccessRequest))).scalar_one()
    assert sent == []
    assert n == 1


@pytest.mark.asyncio
async def test_unknown_telegram_sender_still_gets_the_denial_reply(fresh_db):
    from app.channels.dispatch import DENIED_MESSAGE, dispatch_event
    from app.channels.schema import NormalizedEvent
    from app.db.models import Channel
    from app.db.session import init_db, session_scope

    await init_db()
    sent: list[str] = []

    async def reply(text: str) -> None:
        sent.append(text)

    async with session_scope() as session:
        await dispatch_event(session, NormalizedEvent("424242", Channel.TELEGRAM, "hi", reply))
    assert sent == [DENIED_MESSAGE]


def test_capabilities_are_asked_of_the_server_when_the_client_list_predates_login(fake_imap):
    """Python 3.12 keeps the capabilities read before authentication (no MOVE, no UIDPLUS):
    the message was never filed (live mailbox, 2026-09-21).
    """
    fake = fake_imap({b"12": _raw(*TAGGED)})
    fake.capabilities = ("IMAP4REV1", "AUTH=PLAIN")
    fake.capability = lambda: ("OK", [b"IMAP4rev1 MOVE UIDPLUS NAMESPACE"])
    _finalize_message(b"12", "INBOX.Agent")
    assert fake.moves == [(b"12", "INBOX.Agent")], "moved although the client list had no MOVE"


def test_a_server_that_really_has_no_move_or_uidplus_still_moves_nothing(fake_imap):
    fake = fake_imap({b"12": _raw(*TAGGED)})
    fake.capabilities = ("IMAP4REV1",)
    fake.capability = lambda: ("OK", [b"IMAP4rev1"])
    _finalize_message(b"12", "INBOX.Agent")
    assert fake.moves == [] and fake.copies == [] and set(fake.messages) == {b"12"}


def test_capability_names_are_matched_whatever_their_case(fake_imap):
    fake = fake_imap({b"12": _raw(*TAGGED)})
    fake.capabilities = ("IMAP4REV1",)
    fake.capability = lambda: ("OK", [b"imap4rev1 Move uidplus"])
    _finalize_message(b"12", "INBOX.Agent")
    assert fake.moves == [(b"12", "INBOX.Agent")]


async def test_poll_leaves_a_limited_message_unread_without_counting_an_attempt(monkeypatch):
    """Over the sender's limit, the email is taken again at a later poll: not filed,
    no attempt counted, then handled normally."""
    from app.channels.dispatch import DispatchOutcome

    h = _PollHarness(monkeypatch)
    h.messages = [_msg()]
    h.outcomes = [DispatchOutcome.LIMITED]
    await email_adapter._poll_once()
    assert h.finalized == [] and email_adapter._attempts == {}
    h.outcomes = [DispatchOutcome.OK]
    await email_adapter._poll_once()
    assert h.finalized == [(b"7", "INBOX.Agent")]
    assert h.dispatch_calls[1]["retry"] is False
