"""Tests: the policy functions (M4 defaults, definition hash, argument
redaction), the registry of questions waiting for a yes or no, and how Telegram,
email and the dispatch pipeline carry a confirmation.
"""

import asyncio
import email.message
from types import SimpleNamespace

import pytest

from app.channels import confirmations
from app.db.models import Channel
from app.mcp import policy

# --- policy ---


@pytest.mark.parametrize(
    ("annotations", "expected"),
    [
        (None, "confirm"),  # unannotated: treated as destructive
        ({}, "confirm"),
        ({"readOnlyHint": True, "openWorldHint": False}, "allow"),
        ({"readOnlyHint": True}, "confirm"),  # open-world unless it says otherwise
        ({"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}, "confirm"),
        ({"readOnlyHint": True, "openWorldHint": True}, "confirm"),
    ],
)
def test_the_default_policy_follows_m4(annotations, expected):
    definition = {"name": "t"} if annotations is None else {"name": "t", "annotations": annotations}
    assert policy.default_policy(definition) == expected


def test_an_explicit_policy_wins_and_a_bad_one_falls_back_to_the_default():
    definition = {"name": "t"}
    assert policy.effective_policy("t", definition, {"t": "allow"}) == "allow"
    assert policy.effective_policy("t", definition, {"t": "whatever"}) == "confirm"
    assert policy.effective_policy("t", definition, {"other": "deny"}) == "confirm"


def test_the_definition_hash_ignores_key_order_and_sees_every_change():
    a = {"name": "t", "description": "d", "inputSchema": {"type": "object", "a": 1}}
    b = {"inputSchema": {"a": 1, "type": "object"}, "description": "d", "name": "t"}
    assert policy.definition_hash(a) == policy.definition_hash(b)
    assert policy.definition_hash(a) != policy.definition_hash({**a, "description": "d "})
    assert policy.definition_hash(a) != policy.definition_hash({**a, "annotations": {}})


def test_a_grant_covers_one_tool_or_the_whole_server():
    assert policy.is_granted(frozenset({("s", None)}), "s", "t")
    assert policy.is_granted(frozenset({("s", "t")}), "s", "t")
    assert not policy.is_granted(frozenset({("s", "u")}), "s", "t")
    assert not policy.is_granted(frozenset({("other", None)}), "s", "t")


def test_redaction_covers_secret_names_nested_values_and_configured_secrets():
    from app.logging_setup import install_redaction

    secret = "test-secret-" + "z" * 20
    install_redaction([secret])
    try:
        stored = policy.redacted_arguments(
            {
                "query": f"look up {secret}",
                "Authorization": "Bearer abc",
                "items": [{"client_secret": "x"}, "plain"],
                "count": 3,
            }
        )
    finally:
        install_redaction([])
    assert secret not in stored and "Bearer abc" not in stored and '"x"' not in stored
    assert '"plain"' in stored and '"count": 3' in stored


def test_stored_arguments_are_capped():
    assert len(policy.redacted_arguments({"a": "b" * 10_000})) < 4100


# --- the waiting registry ---


async def _ask(user="u1", timeout=2.0):
    sent = []

    async def send(nonce):
        sent.append(nonce)

    task = asyncio.create_task(confirmations.ask(Channel.TELEGRAM, user, send, timeout))
    for _ in range(50):
        if sent:
            break
        await asyncio.sleep(0.01)
    return task, sent[0]


async def test_a_typed_yes_or_no_answers_the_waiting_question():
    task, _nonce = await _ask()
    assert not confirmations.answer_text(Channel.TELEGRAM, "u1", "tell me more")
    assert confirmations.answer_text(Channel.TELEGRAM, "u1", "Oui!")
    assert await task is True
    task, _nonce = await _ask()
    assert confirmations.answer_text(Channel.TELEGRAM, "u1", "no")
    assert await task is False
    assert not confirmations.is_waiting(Channel.TELEGRAM, "u1")


async def test_an_answer_from_another_user_or_channel_does_not_count():
    task, nonce = await _ask()
    assert not confirmations.answer_text(Channel.TELEGRAM, "u2", "yes")
    assert not confirmations.answer_text(Channel.EMAIL, "u1", "yes")
    assert not confirmations.answer_nonce(Channel.TELEGRAM, "u2", nonce, True)
    assert confirmations.answer_nonce(Channel.TELEGRAM, "u1", nonce, True)
    assert await task is True


async def test_a_button_of_an_older_question_does_not_answer_the_new_one():
    task, nonce = await _ask()
    assert not confirmations.answer_nonce(Channel.TELEGRAM, "u1", "0" * 8, True)
    assert confirmations.is_waiting(Channel.TELEGRAM, "u1")
    confirmations.answer_nonce(Channel.TELEGRAM, "u1", nonce, False)
    assert await task is False


async def test_no_answer_in_time_is_none_and_clears_the_question():
    task, _nonce = await _ask(timeout=0.2)
    # Bounded here, so a wait that never ends fails the test instead of hanging it.
    assert await asyncio.wait_for(task, 5) is None
    assert not confirmations.is_waiting(Channel.TELEGRAM, "u1")
    assert not confirmations.answer_text(Channel.TELEGRAM, "u1", "yes")


async def test_a_second_question_while_one_waits_is_refused():
    task, nonce = await _ask()

    async def send(_nonce):
        raise AssertionError("never sent")

    assert await confirmations.ask(Channel.TELEGRAM, "u1", send, 1.0) is False
    confirmations.answer_nonce(Channel.TELEGRAM, "u1", nonce, True)
    assert await task is True


# --- Telegram ---


class _Bot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, reply_markup=None):
        self.sent.append((chat_id, text, reply_markup))


def _update(text="hello", user_id=55, chat_id=999):
    return SimpleNamespace(
        message=SimpleNamespace(text=text),
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
    )


async def test_telegram_asks_with_yes_and_no_buttons_bound_to_the_question():
    from app.channels import telegram

    bot = _Bot()
    event = telegram._event(_update(), SimpleNamespace(bot=bot), "hello")
    task = asyncio.create_task(event.confirm("Run it?", 2.0))
    for _ in range(50):
        if bot.sent:
            break
        await asyncio.sleep(0.01)
    (chat_id, text, markup) = bot.sent[0]
    assert (chat_id, text) == (999, "Run it?")
    data = [b.callback_data for row in markup.inline_keyboard for b in row]
    nonce = data[0].split(":")[1]
    assert data == [f"mcpc:{nonce}:y", f"mcpc:{nonce}:n"]

    answers = []

    async def answer(text):
        answers.append(text)

    async def edit(text):
        answers.append(text)

    def press(user_id, choice):
        return SimpleNamespace(
            callback_query=SimpleNamespace(
                from_user=SimpleNamespace(id=user_id),
                data=f"mcpc:{nonce}:{choice}",
                message=SimpleNamespace(text="Run it?"),
                answer=answer,
                edit_message_text=edit,
            )
        )

    await telegram._on_button(press(77, "y"), None)  # someone else's button press
    assert not task.done() and answers[0] == "Expired"
    await telegram._on_button(press(55, "y"), None)
    assert await task is True
    assert "Allowed" in answers and "Run it?\n\n(allowed)" in answers


async def test_telegram_takes_a_typed_answer_without_starting_a_turn(monkeypatch):
    from app.channels import telegram

    turns = []

    async def fake_dispatch(session, event, **kwargs):
        turns.append(event.text)

    monkeypatch.setattr(telegram, "dispatch_event", fake_dispatch)
    task, _nonce = await _ask(user="55")
    await telegram._on_message(_update("yes"), SimpleNamespace(bot=_Bot()))
    assert await task is True
    assert turns == []


async def test_telegram_handlers_do_not_block_the_update_loop(monkeypatch):
    """A turn waiting for a confirmation must let the answer through (block=False)."""
    from app.channels import telegram
    from app.config import get_settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:" + "a" * 35)
    get_settings.cache_clear()
    try:
        application = telegram.build_application()
    finally:
        get_settings.cache_clear()
    handlers = [h for group in application.handlers.values() for h in group]
    # text, /agent, /model, /prompt, /task, /new, /export, buttons,
    # /newagent, /cancel and the builder's answer buttons, /editagent, /myagents
    assert len(handlers) == 13
    assert all(h.block is False for h in handlers)


async def test_telegram_turns_still_run_one_at_a_time(monkeypatch):
    from contextlib import asynccontextmanager

    from app.channels import telegram

    running, peak = [0], [0]

    async def fake_dispatch(session, event, **kwargs):
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        await asyncio.sleep(0.05)
        running[0] -= 1

    @asynccontextmanager
    async def fake_scope():
        yield object()

    monkeypatch.setattr(telegram, "dispatch_event", fake_dispatch)
    monkeypatch.setattr(telegram, "session_scope", fake_scope)
    await asyncio.gather(
        *(
            telegram._on_message(_update(f"m{i}", user_id=i), SimpleNamespace(bot=_Bot()))
            for i in range(4)
        )
    )
    assert peak[0] == 1


# --- dispatch carries the event's confirm to the turn ---


async def test_dispatch_hands_the_event_confirm_to_the_turn_and_resets_it(fresh_db, monkeypatch):
    from app.admin.service import (
        add_channel_identity,
        create_user,
        grant_identity_permission,
    )
    from app.channels import dispatch
    from app.channels.schema import NormalizedEvent
    from app.db.models import PermissionKind
    from app.db.session import init_db, session_scope
    from app.mcp.confirm import current_confirmer

    await init_db()
    seen = []

    async def fake_turn(*args, **kwargs):
        seen.append(current_confirmer.get())
        return "ok"

    monkeypatch.setattr(dispatch, "run_turn", fake_turn)

    async def reply(text):
        pass

    async def confirm(question, timeout):
        return True

    async with session_scope() as session:
        user = await create_user(session, "Sam")
        identity = await add_channel_identity(session, user.id, Channel.TELEGRAM, "31")
        await grant_identity_permission(session, user.id, identity.id, PermissionKind.CHAT)
        await session.commit()
        event = NormalizedEvent("31", Channel.TELEGRAM, "hi", reply, confirm)
        outcome = await dispatch.dispatch_event(session, event)
    assert outcome == "ok"
    assert seen == [confirm]
    assert current_confirmer.get() is None


# --- email ---


def _mail(subject, sender, body):
    msg = email.message.EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, sender, "bot@example.org"
    msg.set_content(body)
    return msg.as_bytes()


@pytest.fixture
def mailbox(monkeypatch):
    from app.channels import email as email_adapter
    from tests.test_email_adapter import _FakeIMAP

    def install(messages):
        fake = _FakeIMAP(messages)
        monkeypatch.setattr(email_adapter.imaplib, "IMAP4_SSL", lambda *a, **k: fake)
        monkeypatch.setattr(
            email_adapter,
            "get_settings",
            lambda: SimpleNamespace(
                email_imap_host="h",
                email_imap_port=993,
                email_username="u",
                email_password="p",
                email_trigger_tag="[agent]",
                email_agent_folder="",
            ),
        )
        return fake

    return install


def test_email_takes_the_coded_reply_from_the_asked_address_only(mailbox):
    from app.channels import email as email_adapter

    fake = mailbox(
        {
            b"1": _mail("Re: [agent] Confirm abcd1234: x", "Mallory <m@evil.example>", "yes"),
            b"2": _mail("Re: [agent] Confirm abcd1234: x", "Sam <sam@example.org>", "maybe"),
        }
    )
    find = email_adapter._find_confirmation_sync
    assert find("abcd1234", "sam@example.org", "") is None  # forged sender, then no answer
    assert fake.stores == []
    fake.messages[b"3"] = _mail("Re: [agent] Confirm abcd1234: x", "sam@example.org", "Non\n")
    assert find("abcd1234", "SAM@example.org", "") is False
    assert [s[0] for s in fake.stores] == [b"3"]
    assert find("ffff0000", "sam@example.org", "") is None  # another question's code


async def test_email_confirmation_times_out_when_nobody_answers(mailbox, monkeypatch):
    from app.channels import email as email_adapter

    mailbox({})
    sent = []
    monkeypatch.setattr(email_adapter, "_send_reply_sync", lambda *a: sent.append(a))
    monkeypatch.setattr(email_adapter, "CONFIRM_POLL_SECONDS", 0.05)
    answer = await email_adapter.confirm_by_email("sam@example.org", "Hello", "Run it?", 0.2)
    assert answer is None
    (to, subject, body) = sent[0]
    assert to == "sam@example.org"
    assert subject.startswith("[agent] Confirm ") and subject.endswith(": Hello")
    assert "Run it?" in body


async def test_a_late_email_answer_is_filed_and_never_becomes_a_turn(mailbox, monkeypatch):
    from app.channels import email as email_adapter

    fake = mailbox({b"9": _mail("Re: [agent] Confirm abcd1234: Hello", "sam@example.org", "yes")})
    turns = []

    async def fake_dispatch(session, event, **kwargs):
        turns.append(event.text)

    monkeypatch.setattr(email_adapter, "dispatch_event", fake_dispatch)
    await email_adapter._poll_once()
    assert turns == []
    assert [s[0] for s in fake.stores] == [b"9"]
