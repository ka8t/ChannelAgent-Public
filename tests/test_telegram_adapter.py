"""Tests: the Telegram adapter's own behavior, without the network:
what it ignores, how it normalizes, how it replies, how it polls.
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.channels import telegram
from app.db.models import Channel


def _update(text="hello", user_id=55, chat_id=999, message=True, user=True):
    return SimpleNamespace(
        message=SimpleNamespace(text=text) if message else None,
        effective_user=SimpleNamespace(id=user_id) if user else None,
        effective_chat=SimpleNamespace(id=chat_id),
    )


class _Bot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


@pytest.fixture
def spy(monkeypatch):
    events = []

    async def fake_dispatch(session, event, **kwargs):
        events.append(event)

    monkeypatch.setattr(telegram, "dispatch_event", fake_dispatch)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_scope():
        yield object()

    monkeypatch.setattr(telegram, "session_scope", fake_scope)
    return events


@pytest.mark.parametrize(
    "update",
    [
        _update(message=False),
        _update(text=None),
        _update(user=False),
    ],
    ids=["no message", "no text (photo, sticker...)", "no user"],
)
async def test_updates_without_text_or_user_are_ignored(spy, update):
    await telegram._on_message(update, SimpleNamespace(bot=_Bot()))
    assert spy == []


async def test_a_text_message_becomes_a_normalized_event(spy):
    await telegram._on_message(_update("hi there", user_id=55), SimpleNamespace(bot=_Bot()))
    assert len(spy) == 1
    event = spy[0]
    assert (event.user_id, event.channel, event.text) == ("55", Channel.TELEGRAM, "hi there")


async def test_the_user_id_is_a_string_so_it_matches_the_stored_identity(spy):
    await telegram._on_message(_update(user_id=100200300), SimpleNamespace(bot=_Bot()))
    assert spy[0].user_id == "100200300" and isinstance(spy[0].user_id, str)


async def test_the_reply_callback_sends_to_the_chat_the_message_came_from(spy):
    bot = _Bot()
    await telegram._on_message(_update(chat_id=4242), SimpleNamespace(bot=bot))
    await spy[0].reply("the answer")
    assert bot.sent == [(4242, "the answer")]


def test_building_the_application_requires_a_token(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        telegram.build_application()


class _StubBot:
    """CommandHandler reads the bot's username to recognize /command@bot."""

    username = "testbot"


def _real_update(text: str, entities=None):
    from telegram import Update

    return Update.de_json(
        {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1_700_000_000,
                "chat": {"id": 1, "type": "private"},
                "from": {"id": 55, "is_bot": False, "first_name": "A"},
                "text": text,
                **({"entities": entities} if entities else {}),
            },
        },
        _StubBot(),
    )


def test_the_handlers_take_plain_text_and_the_agent_command_only(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABCDEF-test-token")
    get_settings.cache_clear()
    application = telegram.build_application()
    handlers = [h for group in application.handlers.values() for h in group]
    # Plain text, /agent, /model, /prompt, /task, /new and /export,
    # the Yes/No buttons of tool confirmations, and /newagent, /cancel and the
    # answer buttons of the agent builder, /editagent and /myagents.
    assert len(handlers) == 13
    by_callback = {h.callback: h for h in handlers}
    assert telegram._on_button in by_callback and telegram._on_choice in by_callback
    text_handler, agent_handler = by_callback[telegram._on_message], by_callback[telegram._on_agent]
    model_handler = by_callback[telegram._on_model]
    prompt_handler = by_callback[telegram._on_prompt]

    def command(name, arg=""):
        text = f"/{name} {arg}".strip()
        return _real_update(text, [{"type": "bot_command", "offset": 0, "length": len(name) + 1}])

    assert text_handler.check_update(_real_update("hello")) is not None
    assert not text_handler.check_update(command("agent", "work")), "commands are not plain text"
    assert agent_handler.check_update(command("agent", "work")) is not None
    assert agent_handler.check_update(command("agent")) is not None
    assert not agent_handler.check_update(_real_update("hello"))
    assert not agent_handler.check_update(command("start")), "other commands stay ignored"
    assert model_handler.check_update(command("model", "a.gguf hi")) is not None
    assert not model_handler.check_update(command("agent", "work"))
    assert prompt_handler.check_update(command("prompt", "time-in city=Paris")) is not None
    assert not prompt_handler.check_update(command("model", "a.gguf hi"))
    assert not text_handler.check_update(command("model", "a.gguf hi"))
    assert not text_handler.check_update(command("start"))


class _FakeApplication:
    def __init__(self):
        self.calls = []
        self.updater = SimpleNamespace(start_polling=self._start_polling, stop=self._updater_stop)

        async def get_me():
            return None

        async def set_my_commands(commands, scope=None):
            self.calls.append(("set_my_commands", [c.command for c in commands], scope.type))

        async def delete_my_commands(scope=None):
            self.calls.append(("delete_my_commands", scope.type))

        self.bot = SimpleNamespace(
            get_me=get_me, set_my_commands=set_my_commands, delete_my_commands=delete_my_commands
        )

    async def __aenter__(self):
        self.calls.append("enter")
        return self

    async def __aexit__(self, *exc):
        self.calls.append("exit")

    async def start(self):
        self.calls.append("start")

    async def stop(self):
        self.calls.append("stop")

    async def _start_polling(self, **kwargs):
        self.calls.append(("start_polling", kwargs))

    async def _updater_stop(self):
        self.calls.append("updater_stop")


async def test_the_adapter_polls_dropping_pending_updates_and_stops_cleanly(monkeypatch):
    fake = _FakeApplication()
    monkeypatch.setattr(telegram, "build_application", lambda: fake)
    task = asyncio.create_task(telegram.run_telegram_adapter())
    for _ in range(50):
        await asyncio.sleep(0.01)
        if ("start_polling", {"drop_pending_updates": True}) in fake.calls:
            break
    assert ("start_polling", {"drop_pending_updates": True}) in fake.calls
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.calls[-3:] == ["updater_stop", "stop", "exit"], "shut down in order on cancel"


async def test_the_running_adapter_registers_with_the_healthcheck_and_proves_getme(monkeypatch):
    """While running, the adapter is expected to succeed regularly and calls getMe;
    once stopped it no longer holds the healthcheck.
    """
    from app import health

    fake = _FakeApplication()
    asked = []

    async def get_me():
        asked.append(1)

    fake.bot = SimpleNamespace(get_me=get_me)
    monkeypatch.setattr(telegram, "build_application", lambda: fake)
    task = asyncio.create_task(telegram.run_telegram_adapter())
    for _ in range(100):
        await asyncio.sleep(0.01)
        if asked:
            break
    assert asked == [1] and "telegram" in health._components
    assert health.stale_components() == []
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "telegram" not in health._components


async def test_model_command_keeps_the_message_as_typed(spy, monkeypatch):
    """`/model <name> <message>`: the name is the first word after the command, the
    message is the rest of the text as typed, line breaks kept."""
    calls = []

    async def fake_model_command(session, event, name, message):
        calls.append((event.text, name, message))

    monkeypatch.setattr(telegram, "handle_model_command", fake_model_command)
    bot = SimpleNamespace(bot=_Bot(), args=["a.gguf", "line"])
    await telegram._on_model(_update("/model a.gguf line one\n  line two"), bot)
    await telegram._on_model(_update("/model"), bot)
    assert calls == [
        ("/model a.gguf", "a.gguf", "line one\n  line two"),
        ("/model", "", ""),
    ]


async def test_prompt_command_splits_the_name_from_its_arguments(spy, monkeypatch):
    """`/prompt <name> key=value ...`: the name is the first word, the arguments the rest."""
    calls = []

    async def fake_prompt_command(session, event, name, arguments):
        calls.append((event.text, name, arguments))

    monkeypatch.setattr(telegram, "handle_prompt_command", fake_prompt_command)
    bot = SimpleNamespace(bot=_Bot(), args=[])
    await telegram._on_prompt(_update("/prompt time-in city='New York'"), bot)
    await telegram._on_prompt(_update("/prompt"), bot)
    assert calls == [("/prompt time-in", "time-in", "city='New York'"), ("/prompt", "", "")]


async def _run_until_polling(fake, monkeypatch):
    monkeypatch.setattr(telegram, "build_application", lambda: fake)
    task = asyncio.create_task(telegram.run_telegram_adapter())
    for _ in range(100):
        await asyncio.sleep(0.01)
        if any(isinstance(c, tuple) and c[0] == "start_polling" for c in fake.calls):
            break
    return task


async def test_the_menu_is_replaced_by_the_handled_commands_before_polling(monkeypatch):
    """The bot kept a previous deployment's menu of 60 commands (/agents, /help ...) that
    nothing answered (2026-09-28): the adapter publishes its own at every start."""
    fake = _FakeApplication()
    task = await _run_until_polling(fake, monkeypatch)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    menu = [c for c in fake.calls if isinstance(c, tuple) and c[0] == "set_my_commands"]
    handled = [
        "agent", "model", "prompt", "task", "new", "export", "newagent", "editagent", "myagents",
        "cancel",
    ]  # fmt: skip
    assert menu == [("set_my_commands", handled, "default")]
    cleared = {c[1] for c in fake.calls if isinstance(c, tuple) and c[0] == "delete_my_commands"}
    assert cleared == {"all_private_chats", "all_group_chats", "all_chat_administrators"}
    names = [c if isinstance(c, str) else c[0] for c in fake.calls]
    assert names.index("set_my_commands") < names.index("start_polling")


def test_the_menu_lists_exactly_the_commands_that_have_a_handler(monkeypatch):
    from telegram.ext import CommandHandler

    from app.config import get_settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABCDEF-test-token")
    get_settings.cache_clear()
    application = telegram.build_application()
    handled = {
        name
        for group in application.handlers.values()
        for h in group
        if isinstance(h, CommandHandler)
        for name in h.commands
    }
    assert handled == {name for name, _d, _cb in telegram.COMMANDS}


async def test_a_menu_that_cannot_be_published_does_not_stop_the_adapter(monkeypatch, caplog):
    fake = _FakeApplication()

    async def refused(*args, **kwargs):
        raise RuntimeError("Forbidden")

    fake.bot.set_my_commands = refused
    task = await _run_until_polling(fake, monkeypatch)
    assert ("start_polling", {"drop_pending_updates": True}) in fake.calls
    assert not task.done()
    assert "Could not publish the Telegram command menu" in caplog.text
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
