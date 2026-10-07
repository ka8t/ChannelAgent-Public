"""Telegram channel adapter: receives messages via long polling,
normalizes them, and routes them through the shared dispatch pipeline
(app/channels/dispatch.py). Delivery back to the user is a closure over
bot.send_message — this module is the only place that knows Telegram's
API shape; app/graph.py and app/security/auth.py never see it.
"""

import asyncio
import logging
import secrets
import time

from telegram import (
    BotCommand,
    BotCommandScopeAllChatAdministrators,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeDefault,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app import health
from app.channels import confirmations, notify
from app.channels.dispatch import (
    dispatch_event,
    handle_agent_command,
    handle_export_command,
    handle_model_command,
    handle_new_command,
    handle_prompt_command,
    handle_task_command,
)
from app.channels.schema import NormalizedEvent, ReplyProgress
from app.config import get_settings
from app.db.models import Channel
from app.db.session import session_scope

logger = logging.getLogger("channelagent")


# Handlers run with block=False: a turn that waits for a tool confirmation must
# not stop the application from reading the answer (a button, a typed "yes"). This lock
# keeps the turns themselves one at a time, as they were when handlers blocked.
_turns = asyncio.Lock()
BUTTON_PREFIX = "mcpc"
# Answer buttons of the agent builder: `bld:<nonce>:<choice>`.
CHOICE_PREFIX = "bld"


# While a turn runs. Telegram shows "typing" for 5 s after each chat action, and a draft
# (sendMessageDraft, private chats) for 30 s after its last change: both are sent again before
# they fade. A text change is sent at most once a second (Telegram limits a chat to about one
# message per second).
TYPING_EVERY = 4.0
DRAFT_REFRESH_EVERY = 20.0
MIN_UPDATE_INTERVAL = 1.0
MAX_TEXT = 4096


class TelegramProgress(ReplyProgress):
    """Typing, then the answer as it is written: a draft in a private chat (sendMessageDraft,
    open to every bot since python-telegram-bot 22.7), a message edited in place
    elsewhere. The final
    reply is still sent by `reply`: in a private chat as a new message (a draft is only a
    preview), elsewhere as the last edit of the streamed message (`take_message`)."""

    def __init__(self, bot, chat_id: int, private: bool, clock=time.monotonic):
        self._bot, self._chat_id, self._private, self._clock = bot, chat_id, private, clock
        self._draft_id = secrets.randbelow(2**31 - 1) + 1
        self._shown = ""
        self._last_sent = None
        self._message_id = None
        self._task = None
        self._broken = False

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._keep_alive())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def update(self, text: str) -> None:
        text = text[:MAX_TEXT]
        if self._broken or not text.strip() or text == self._shown:
            return
        now = self._clock()
        if self._last_sent is not None and now - self._last_sent < MIN_UPDATE_INTERVAL:
            return
        self._last_sent = now
        await self._show(text)

    @property
    def shown(self) -> str:
        return self._shown

    def take_message(self) -> int | None:
        """The id of the message streamed in a group, handed once to the final reply."""
        message_id, self._message_id = self._message_id, None
        return message_id

    async def _show(self, text: str) -> None:
        try:
            if self._private:
                await self._bot.send_message_draft(
                    chat_id=self._chat_id, draft_id=self._draft_id, text=text
                )
            elif self._message_id is None:
                sent = await self._bot.send_message(chat_id=self._chat_id, text=text)
                self._message_id = sent.message_id
            else:
                await self._bot.edit_message_text(
                    chat_id=self._chat_id, message_id=self._message_id, text=text
                )
            self._shown = text
        except Exception:
            # A display: the turn goes on and the final reply is sent as usual.
            logger.warning("Showing the reply as it is written failed", exc_info=True)
            self._broken = True

    async def _keep_alive(self) -> None:
        last_draft = None
        while True:
            try:
                await self._bot.send_chat_action(chat_id=self._chat_id, action="typing")
            except Exception:
                logger.debug("Sending the typing action failed", exc_info=True)
            now = self._clock()
            if self._private and not self._broken and (
                last_draft is None or now - last_draft >= DRAFT_REFRESH_EVERY
            ):
                last_draft = now
                try:
                    # Empty text: Telegram shows "Thinking..." until the first words.
                    await self._bot.send_message_draft(
                        chat_id=self._chat_id, draft_id=self._draft_id, text=self._shown
                    )
                except Exception:
                    logger.debug("Refreshing the draft failed", exc_info=True)
            await asyncio.sleep(TYPING_EVERY)


def _event(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> NormalizedEvent:
    chat_id = update.effective_chat.id
    user_id = str(update.effective_user.id)
    progress = TelegramProgress(
        context.bot, chat_id, private=getattr(update.effective_chat, "type", None) == "private"
    )

    async def reply(response_text: str) -> None:
        streamed = progress.take_message()
        if streamed is not None:
            # A group: the streamed message becomes the reply (Telegram refuses an edit that
            # changes nothing). If the edit fails, the reply is sent as a new message.
            if response_text == progress.shown:
                return
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=streamed, text=response_text
                )
                return
            except Exception:
                logger.warning("Editing the streamed message failed, sending anew", exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text=response_text)

    async def confirm(question: str, timeout: float) -> bool | None:
        async def send(nonce: str) -> None:
            buttons = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("Yes", callback_data=f"{BUTTON_PREFIX}:{nonce}:y"),
                        InlineKeyboardButton("No", callback_data=f"{BUTTON_PREFIX}:{nonce}:n"),
                    ]
                ]
            )
            await context.bot.send_message(chat_id=chat_id, text=question, reply_markup=buttons)

        return await confirmations.ask(Channel.TELEGRAM, user_id, send, timeout)

    async def send_file(name: str, data: bytes) -> None:
        await context.bot.send_document(chat_id=chat_id, document=data, filename=name)

    async def reply_choices(response_text: str, choices: list[str], nonce: str) -> None:
        buttons = InlineKeyboardMarkup(
            [[InlineKeyboardButton(c.capitalize(), callback_data=f"{CHOICE_PREFIX}:{nonce}:{c}")
              for c in choices]]
        )  # fmt: skip
        await context.bot.send_message(chat_id=chat_id, text=response_text, reply_markup=buttons)

    return NormalizedEvent(
        user_id=user_id, channel=Channel.TELEGRAM, text=text, reply=reply, confirm=confirm,
        send_file=send_file, reply_choices=reply_choices, progress=progress,
    )  # fmt: skip


async def _on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None or update.effective_user is None:
        return
    # A typed yes/no answers a question this user's running turn is waiting for.
    if confirmations.answer_text(
        Channel.TELEGRAM, str(update.effective_user.id), update.message.text
    ):
        return
    event = _event(update, context, update.message.text)
    async with _turns, session_scope() as session:
        await dispatch_event(session, event)


async def _on_agent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/agent lists the sender's agents, /agent <name> switches."""
    if update.message is None or update.effective_user is None:
        return
    argument = " ".join(context.args or [])
    event = _event(update, context, f"/agent {argument}".strip())
    async with _turns, session_scope() as session:
        await handle_agent_command(session, event, argument)


async def _on_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/model lists the engine's models; /model <name> <message> answers that message with
    that model. The message is taken from the text as typed, line breaks kept."""
    if update.message is None or update.message.text is None or update.effective_user is None:
        return
    parts = update.message.text.split(None, 2)
    name = parts[1] if len(parts) > 1 else ""
    message = parts[2] if len(parts) > 2 else ""
    event = _event(update, context, f"/model {name}".strip())
    async with _turns, session_scope() as session:
        await handle_model_command(session, event, name, message)


async def _on_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/prompt lists the MCP prompts available; /prompt <name> key=value ... runs one."""
    if update.message is None or update.message.text is None or update.effective_user is None:
        return
    parts = update.message.text.split(None, 2)
    name = parts[1] if len(parts) > 1 else ""
    arguments = parts[2] if len(parts) > 2 else ""
    event = _event(update, context, f"/prompt {name}".strip())
    async with _turns, session_scope() as session:
        await handle_prompt_command(session, event, name, arguments)


async def _on_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/task: the sender's scheduled tasks. The prompt is taken as typed."""
    if update.message is None or update.message.text is None or update.effective_user is None:
        return
    parts = update.message.text.split(None, 1)
    text = parts[1] if len(parts) > 1 else ""
    event = _event(update, context, f"/task {text}".strip())
    async with _turns, session_scope() as session:
        await handle_task_command(session, event, text)


async def _on_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/new: a fresh conversation with the current agent."""
    if update.message is None or update.effective_user is None:
        return
    async with _turns, session_scope() as session:
        await handle_new_command(session, _event(update, context, "/new"))


async def _on_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/export: the sender's own conversation as a Markdown file."""
    if update.message is None or update.effective_user is None:
        return
    async with _turns, session_scope() as session:
        await handle_export_command(session, _event(update, context, "/export"))


async def _on_builder_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/newagent <request> opens the agent builder, /cancel ends it. The text is taken
    as typed: the dialogue is handled by the shared pipeline, like on every channel."""
    if update.message is None or update.message.text is None or update.effective_user is None:
        return
    event = _event(update, context, update.message.text.strip())
    async with _turns, session_scope() as session:
        await dispatch_event(session, event)


async def _on_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """An answer button of the agent builder: sent as the user's typed answer, only
    while the question it was shown with is the one waiting for that user."""
    from app import builder
    from app.security.hashing import channel_identifier_key

    query = update.callback_query
    if query is None or query.from_user is None or not query.data:
        return
    _prefix, _sep, rest = query.data.partition(":")
    nonce, _sep, choice = rest.partition(":")
    user_id = str(query.from_user.id)
    thread = builder.thread_id(Channel.TELEGRAM, channel_identifier_key(Channel.TELEGRAM, user_id))
    pending = await builder.waiting(thread)
    current = pending is not None and pending.get("nonce") == nonce and choice in builder.CHOICES
    await query.answer(choice if current else "Expired")
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        logger.debug("Could not remove the builder's buttons", exc_info=True)
    if not current:
        return
    event = _event(update, context, choice)
    async with _turns, session_scope() as session:
        await dispatch_event(session, event)


async def _on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A Yes/No button under a confirmation question. It answers only the
    question it was sent with, and only for the user it was sent to."""
    query = update.callback_query
    if query is None or query.from_user is None or not query.data:
        return
    _prefix, _sep, rest = query.data.partition(":")
    nonce, _sep, choice = rest.partition(":")
    value = choice == "y"
    answered = confirmations.answer_nonce(
        Channel.TELEGRAM, str(query.from_user.id), nonce, value
    )
    await query.answer("Allowed" if answered and value else "Declined" if answered else "Expired")
    try:
        verdict = ("allowed" if value else "declined") if answered else "expired, not used"
        await query.edit_message_text(f"{query.message.text}\n\n({verdict})")
    except Exception:
        logger.debug("Could not mark the confirmation message", exc_info=True)


# The commands this adapter handles, and the menu Telegram shows for them: one list, so
# the menu cannot offer a command that is not handled. The bot's menu is stored by
# Telegram, not by the application: a token used before kept the previous deployment's
# 60 commands (/agents, /help, /new ...), which the menu offered and nothing answered
# (measured with getMyCommands, 2026-09-28).
COMMANDS = (
    ("agent", "List your agents, or /agent <name> to switch", _on_agent),
    ("model", "List the models, or /model <name> <message> for one answer", _on_model),
    ("prompt", "List the MCP prompts, or /prompt <name> key=value to run one", _on_prompt),
    ("task", "Your scheduled tasks, or /task add daily 08:30 <prompt>", _on_task),
    ("new", "Start a new conversation with your current agent", _on_new),
    ("export", "Your conversation with your current agent, as a file", _on_export),
    ("newagent", "Create an agent: /newagent <what it should do and when>", _on_builder_command),
    ("editagent", "Change one of your agents: /editagent <name> <what to change>",
     _on_builder_command),
    ("myagents", "Your agents; /myagents pause|resume|run|delete <name>", _on_builder_command),
    ("cancel", "Stop creating an agent", _on_builder_command),
)
# Scopes a previous deployment may have filled; Telegram shows the most specific one,
# so they are cleared and the default scope carries the menu.
_OTHER_SCOPES = (
    BotCommandScopeAllPrivateChats(),
    BotCommandScopeAllGroupChats(),
    BotCommandScopeAllChatAdministrators(),
)


async def publish_commands(bot) -> None:
    """Replaces the bot's command menu with COMMANDS. A failure is only logged: the
    commands work without the menu."""
    try:
        for scope in _OTHER_SCOPES:
            await bot.delete_my_commands(scope=scope)
        await bot.set_my_commands(
            [BotCommand(name, description) for name, description, _ in COMMANDS],
            scope=BotCommandScopeDefault(),
        )
    except Exception:
        logger.warning("Could not publish the Telegram command menu", exc_info=True)


def build_application() -> Application:
    token = get_settings().telegram_bot_token
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    application = Application.builder().token(token).build()
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message, block=False)
    )
    for name, _description, callback in COMMANDS:
        application.add_handler(CommandHandler(name, callback, block=False))
    application.add_handler(
        CallbackQueryHandler(_on_button, pattern=rf"^{BUTTON_PREFIX}:", block=False)
    )
    application.add_handler(
        CallbackQueryHandler(_on_choice, pattern=rf"^{CHOICE_PREFIX}:", block=False)
    )
    return application


LIVENESS_INTERVAL_SECONDS = 60
LIVENESS_MAX_AGE_SECONDS = 300


async def _liveness_loop(bot, interval: float = LIVENESS_INTERVAL_SECONDS) -> None:
    """Proves the path to Telegram works (`getMe`) and tells the healthcheck.
    A failure is only logged: the container turns unhealthy when none succeeds for
    LIVENESS_MAX_AGE_SECONDS.
    """
    while True:
        try:
            await bot.get_me()
            health.mark("telegram")
        except Exception:
            logger.warning("Telegram getMe failed", exc_info=True)
        await asyncio.sleep(interval)


async def run_telegram_adapter() -> None:
    """Runs inside the caller's own asyncio event loop (app/main.py) —
    deliberately not Application.run_polling(), which manages its own
    loop and isn't meant to be awaited alongside other adapters (Email,
    Matrix) in the same process. Runs until cancelled.
    """
    application = build_application()
    async with application:
        await application.start()
        await publish_commands(application.bot)
        await application.updater.start_polling(drop_pending_updates=True)

        async def send_to_admin(external_id: str, text: str) -> None:
            await application.bot.send_message(chat_id=int(external_id), text=text)

        notify.register_sender(Channel.TELEGRAM, send_to_admin)
        logger.info("Telegram adapter started (long polling).")
        health.register("telegram", LIVENESS_MAX_AGE_SECONDS)
        liveness = asyncio.create_task(_liveness_loop(application.bot))
        try:
            await asyncio.Event().wait()  # runs until this task is cancelled
        finally:
            liveness.cancel()
            health.unregister("telegram")
            notify.unregister_sender(Channel.TELEGRAM)
            await application.updater.stop()
            await application.stop()
