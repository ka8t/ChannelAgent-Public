"""Shared event pipeline every channel adapter calls: authorize,
run the graph, deliver the reply. The one place that connects the Auth
Node and app/graph.py to a NormalizedEvent's reply() callback, so
neither has to know which channel a message came from.

Also the one place that writes ActionLog rows and creates
AccessRequests for denied identities — both via app.admin.service,
the shared layer the interactive CLI and Admin API also use.
"""

import enum
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import (
    ensure_access_request,
    find_agent_by_name,
    find_undelivered_answer,
    get_or_create_default_agent,
    list_agents,
    record_action,
    record_request_message,
    resolve_agent,
)
from app.channels.limits import MESSAGES as LIMIT_MESSAGES
from app.channels.limits import limiter
from app.channels.notify import describe_request, notify_admins
from app.channels.schema import NormalizedEvent
from app.config import get_settings
from app.db.models import ActionStatus, Channel, Direction
from app.graph import last_turn_stats, llm_timeout, reply_stream, run_turn
from app.mcp.confirm import current_confirmer
from app.replies import EmptyReplyError
from app.security.auth import authorize
from app.security.hashing import channel_identifier_key

logger = logging.getLogger("channelagent")

DENIED_MESSAGE = (
    "You're not authorized to use this bot yet. Your request has been recorded "
    "and an admin will review it."
)

# Channels where an unauthorized sender gets an AccessRequest but no
# reply. An email's From address can be forged, so answering it would
# send mail to a third party, and the mailbox is shared with ordinary
# customer mail.
SILENT_DENIAL_CHANNELS = frozenset({Channel.EMAIL})


class DispatchOutcome(enum.StrEnum):
    OK = "ok"  # answered and delivered
    DENIED = "denied"  # not authorized, nothing to retry
    FAILED = "failed"  # the turn failed
    # The answer was produced and kept, but could not be delivered. A
    # retry sends the kept text again instead of running the turn a second time.
    UNDELIVERED = "undelivered"
    # Over the user's rate or turn limit: answered with a short notice, not run.
    LIMITED = "limited"


APOLOGY_MESSAGE = "Sorry, I cannot answer right now. Please try again in a few minutes."
NO_REPLY_NOTE = "(the turn failed, no reply was sent)"
AGENT_DISABLED_MESSAGE = "This agent is currently disabled. Please contact an administrator."


async def dispatch_event(
    session: AsyncSession,
    event: NormalizedEvent,
    *,
    apologize: bool = True,
    retry: bool = False,
) -> DispatchOutcome:
    """Authorize, run the graph, deliver the reply, and record every step.

    Failures never escape: the inbound message is committed to the
    audit trail *before* the LLM is called, so a failed turn cannot lose
    it; a failed turn is recorded with status "failed" and, when
    `apologize` is set, answered with a fixed apology (never an error
    text). The caller learns the outcome and decides about retrying.

    `retry` is for a message that is being processed again: the inbound
    entry already exists and is not written twice. If that message had been
    answered but the answer could not be delivered, only the delivery is
    retried, with the text kept in the audit trail: the model is not
    called again and the conversation gains no second turn. `apologize=False`
    is for channels that retry silently (email).
    """
    decision = await authorize(session, event.channel, event.user_id)

    if decision.allowed:
        if event.text.strip().split(" ", 1)[0].lower() == MYAGENTS_COMMAND:
            return await handle_myagents(session, event, decision)
        outcome = await handle_builder(session, event, decision)
        if outcome is not None:
            return outcome
    if not decision.allowed:
        logger.info("Denied %s/%s: not authorized", event.channel.value, event.user_id)
        key = channel_identifier_key(event.channel, event.user_id)
        request, is_new = await ensure_access_request(session, event.channel, key, event.text)
        # Every message of a sender who is not a user yet: the request keeps only the
        # first, and a stranger's later messages were recorded nowhere. A known user without
        # the permission is logged below, with their user.
        if decision.user is None:
            await record_request_message(session, request, event.text)
        if decision.user is not None:
            agent = await get_or_create_default_agent(session, decision.user.id)
            await record_action(
                session, user_id=decision.user.id, agent_id=agent.id, channel=event.channel,
                direction=Direction.INBOUND, text=event.text, status=ActionStatus.DENIED,
            )
        await session.commit()
        if is_new:
            # Once per request, after it is safely stored. Never raises.
            await notify_admins(session, describe_request(request))
        if event.channel not in SILENT_DENIAL_CHANNELS:
            await event.reply(DENIED_MESSAGE)
        return DispatchOutcome.DENIED

    agent = await resolve_agent(session, decision.user.id, decision.identity.active_agent_id)
    user_id, agent_id = decision.user.id, agent.id
    if not agent.is_active:
        # Deactivating an agent has to stop it answering: no LLM call.
        logger.info(
            "Agent %s of %s/%s is deactivated", agent_id, event.channel.value, event.user_id
        )
        await record_action(
            session, user_id=user_id, agent_id=agent_id, channel=event.channel,
            direction=Direction.INBOUND, text=event.text, status=ActionStatus.DENIED,
        )
        await session.commit()
        try:
            await event.reply(AGENT_DISABLED_MESSAGE)
        except Exception:
            logger.exception("Sending the disabled-agent notice to %s failed", event.user_id)
        return DispatchOutcome.DENIED
    if not retry:
        refusal = limiter.refusal(user_id)
        if refusal is not None:
            logger.info(
                "Limited %s/%s (%s): message not run", event.channel.value, event.user_id, refusal
            )
            await record_action(
                session, user_id=user_id, agent_id=agent_id, channel=event.channel,
                direction=Direction.INBOUND, text=event.text, status=ActionStatus.LIMITED,
            )
            await session.commit()
            if event.channel not in SILENT_DENIAL_CHANNELS:
                try:
                    await event.reply(LIMIT_MESSAGES[refusal])
                except Exception:
                    logger.exception("Sending the limit notice to %s failed", event.user_id)
            return DispatchOutcome.LIMITED
        await record_action(
            session, user_id=user_id, agent_id=agent_id, channel=event.channel,
            direction=Direction.INBOUND, text=event.text,
        )
    await session.commit()

    if retry:
        pending = await find_undelivered_answer(
            session, user_id, agent_id, event.channel, event.text,
            not_answers=(APOLOGY_MESSAGE, NO_REPLY_NOTE),
        )
        if pending is not None:
            return await _redeliver(session, event, pending)

    # A "confirm" tool asks through this event's channel. Reset after the turn,
    # so the next event handled in this same task never inherits it.
    confirmer = current_confirmer.set(event.confirm)
    # What the channel shows while the turn runs: typing, then the answer as it is
    # written. Stopped in the `finally` below, before the final reply is delivered.
    progress = event.progress
    streamer = reply_stream.set(progress.update if progress is not None else None)
    # Nobody waits for an email reply in real time. With the chat limit (120 s) an answer
    # of about 1000 tokens on the owner's model ended at 120 s and was run again in full by the
    # email retry; an email turn gets the task limit (LLM_TASK_TIMEOUT_SECONDS) instead.
    # A terminal turn is not streamed either (the client follows a job), so the chat limit
    # cut a whole answer at 120 s (measured 2026-10-04 with the engine busy with an email).
    patience = llm_timeout.set(
        get_settings().llm_task_timeout_seconds
        if event.channel in (Channel.EMAIL, Channel.TERMINAL)
        else None
    )
    try:
        if progress is not None:
            await progress.start()
        async with limiter.turn(user_id):
            reply_text = await run_turn(
                event.channel, event.user_id, agent_id, event.text, retry=retry, model=event.model
            )
        if not (reply_text or "").strip():
            # A reasoning model can end its turn with no visible text (2026-09-28: 66 s,
            # empty content, no tool call). Sent as is, Telegram refused it ("Message text
            # is empty") and the user got nothing, not even the apology.
            raise EmptyReplyError("the model returned an empty reply")
    except Exception as exc:
        logger.exception("Turn failed for %s/%s", event.channel.value, event.user_id)
        if progress is not None:
            await progress.stop()  # before the apology; stop() is idempotent
        if isinstance(exc, ValueError) and "decrypt" in str(exc).lower():
            # The stored history cannot be read, so every message on this
            # conversation will fail until an admin resets it.
            logger.error(
                "The stored conversation of user %s, agent %s cannot be read. An admin can "
                "reset it: console, Users > reset-conversation, or "
                "POST /users/%s/conversations/reset?agent_id=%s",
                user_id, agent_id, user_id, agent_id,
            )
        sent = False
        if apologize:
            try:
                await event.reply(APOLOGY_MESSAGE)
                sent = True
            except Exception:
                logger.exception(
                    "Sending the apology to %s/%s failed", event.channel.value, event.user_id
                )
        await record_action(
            session, user_id=user_id, agent_id=agent_id, channel=event.channel,
            direction=Direction.OUTBOUND, text=APOLOGY_MESSAGE if sent else NO_REPLY_NOTE,
            status=ActionStatus.FAILED,
        )
        await session.commit()
        return DispatchOutcome.FAILED
    finally:
        current_confirmer.reset(confirmer)
        reply_stream.reset(streamer)
        llm_timeout.reset(patience)
        if progress is not None:
            await progress.stop()

    outbound = await record_action(
        session, user_id=user_id, agent_id=agent_id, channel=event.channel,
        direction=Direction.OUTBOUND, text=reply_text, stats=last_turn_stats.get(),
    )
    await session.commit()
    try:
        await event.reply(reply_text)
    except Exception:
        logger.exception("Delivering the reply to %s/%s failed", event.channel.value, event.user_id)
        outbound.status = ActionStatus.FAILED
        await session.commit()
        return DispatchOutcome.UNDELIVERED
    return DispatchOutcome.OK


async def _builder_step(event: NormalizedEvent, thread: str, user_id: int, identity_id: int):
    """The builder's reply to this message, or None when the message is not for it."""
    from app import builder

    text = event.text.strip()
    command, _sep, rest = text.partition(" ")
    command = command.lower()
    if command == builder.START_COMMAND:
        if not rest.strip():
            return {"say": builder.USAGE, "choices": []}
        return await builder.start(thread, user_id, identity_id, rest.strip())
    if command == builder.EDIT_COMMAND:
        from app.admin.service import find_agent_by_name
        from app.db.session import session_scope

        name, _sep, change = rest.strip().partition(" ")
        if not name or not change.strip():
            return {"say": builder.EDIT_USAGE, "choices": []}
        async with session_scope() as session:
            agent = await find_agent_by_name(session, user_id, name)
        if agent is None:
            return {"say": f"No agent named {name!r}.", "choices": []}
        return await builder.start_edit(thread, user_id, identity_id, agent.id, change.strip())
    if command == builder.CANCEL_COMMAND:
        if await builder.waiting(thread) is None:
            return {"say": builder.NOTHING_TO_CANCEL, "choices": []}
        await builder.drop(thread)
        return {"say": builder.CANCELLED, "choices": []}
    pending = await builder.waiting(thread)
    if pending is None:
        return None
    if builder.is_stale(pending):
        # An abandoned dialogue: dropped, and this message goes to the agent as usual.
        await builder.drop(thread)
        return None
    return await builder.answer(thread, text)


async def handle_builder(session: AsyncSession, event: NormalizedEvent, decision):
    """The agent builder: `/newagent <request>` opens a dialogue, `/cancel` ends it, and
    while it waits for an answer the identity's messages go to it instead of the agent. Returns
    None when the message is not for the builder. Logged, limited and apologised for like a
    turn; the builder acts only as this user (app.admin.agent_spec)."""
    from app import builder

    thread = builder.thread_id(event.channel, decision.identity.external_id)
    text = event.text.strip()
    command = text.split(" ", 1)[0].lower() if text else ""
    if command not in (builder.START_COMMAND, builder.EDIT_COMMAND, builder.CANCEL_COMMAND):
        if await builder.waiting(thread) is None:
            return None
    user_id = decision.user.id
    agent = await resolve_agent(session, user_id, decision.identity.active_agent_id)
    agent_id, identity_id = agent.id, decision.identity.id
    log = {"session": session, "user_id": user_id, "agent_id": agent_id, "channel": event.channel}
    refusal = limiter.refusal(user_id)
    if refusal is not None:
        await record_action(**log, direction=Direction.INBOUND, text=event.text,
                            status=ActionStatus.LIMITED)  # fmt: skip
        await session.commit()
        if event.channel not in SILENT_DENIAL_CHANNELS:
            await event.reply(LIMIT_MESSAGES[refusal])
        return DispatchOutcome.LIMITED
    await record_action(**log, direction=Direction.INBOUND, text=event.text)
    await session.commit()
    try:
        async with limiter.turn(user_id):
            step = await _builder_step(event, thread, user_id, identity_id)
    except Exception:
        logger.exception("Agent builder step failed for %s/%s", event.channel.value, event.user_id)
        await event.reply(APOLOGY_MESSAGE)
        await record_action(**log, direction=Direction.OUTBOUND, text=APOLOGY_MESSAGE,
                            status=ActionStatus.FAILED)  # fmt: skip
        await session.commit()
        return DispatchOutcome.FAILED
    if step is None:
        # The dialogue was abandoned and dropped: the message goes to the agent. Its inbound
        # row is already written, so the turn runs as a retry of it.
        return await dispatch_event(session, event, retry=True)
    say = step["say"]
    await record_action(**log, direction=Direction.OUTBOUND, text=say)
    await session.commit()
    try:
        if step.get("choices") and step.get("nonce") and event.reply_choices is not None:
            await event.reply_choices(say, list(step["choices"]), step["nonce"])
        else:
            await event.reply(say)
    except Exception:
        logger.exception("Delivering the builder's reply to %s failed", event.user_id)
        return DispatchOutcome.UNDELIVERED
    return DispatchOutcome.OK


MYAGENTS_COMMAND = "/myagents"
MYAGENTS_USAGE = (
    "Your agents: /myagents lists them. /myagents pause|resume|run|delete <name>; "
    "/editagent <name> <what to change>."
)
DELETE_CONFIRM_SECONDS = 120


def _when(value, timezone: str | None) -> str:
    from zoneinfo import ZoneInfo

    if value is None:
        return "-"
    return value.astimezone(ZoneInfo(timezone or "UTC")).strftime("%a %Y-%m-%d %H:%M")


def _overview_text(rows: list[dict], timezone: str | None) -> str:
    if not rows:
        return "You have no agent."
    lines = [f"Your agents (times in {timezone or 'UTC'}):"]
    for row in rows:
        state = "on" if row["is_active"] else "disabled"
        purpose = f": {row['purpose']}" if row["purpose"] else ""
        lines.append(f"- {row['name']} [{state}]{purpose}")
        for task in row["tasks"]:
            last = (f", last {task['last_status']} {_when(task['last_run_at'], timezone)}"
                    if task["last_run_at"] else "")  # fmt: skip
            lines.append(
                f"  task #{task['task_id']} {task['kind']} {task['expr']} "
                f"({'on' if task['enabled'] else 'paused'}), "
                f"next {_when(task['next_run_at'], timezone)}{last}"
            )
    return "\n".join([*lines, MYAGENTS_USAGE])


async def _myagents_answer(session: AsyncSession, decision, event: NormalizedEvent, words):
    """The reply to one /myagents command. Every operation carries the sender's user
    id: another user's agent is "No agent named ..." like a missing one."""
    from app import tasks
    from app.admin import my_agents
    from app.admin.service import ConflictError, InvalidInputError, NotFoundError

    user = decision.user
    actor = f"user:{user.id}"
    if not words:
        return _overview_text(await my_agents.overview(session, user.id), user.timezone)
    action = words[0].lower()
    if action not in ("pause", "resume", "run", "delete") or len(words) < 2:
        return MYAGENTS_USAGE
    name = words[1]
    agent = await find_agent_by_name(session, user.id, name)
    if agent is None:
        return f"No agent named {name!r}. " + MYAGENTS_USAGE
    try:
        if action in ("pause", "resume"):
            changed = await my_agents.set_paused(session, user.id, agent.id, action == "pause",
                                                 actor=actor)  # fmt: skip
            done = "paused" if action == "pause" else "resumed"
            return f"Agent {agent.name!r}: {len(changed)} scheduled task(s) {done}."
        if action == "run":
            ids = await my_agents.task_ids(session, user.id, agent.id)
            if not ids:
                return f"Agent {agent.name!r} has no scheduled task to run."
            await session.commit()
            results = [await tasks.execute(task_id, trigger="manual") for task_id in ids]
            return f"Agent {agent.name!r} ran: " + ", ".join(
                f"task #{r['task_id']} {r['status']}" for r in results
            )
        count = len(await my_agents.task_ids(session, user.id, agent.id))
        question = (f"Delete agent {agent.name!r}? Its {count} scheduled task(s) are deleted and "
                    "it is disabled (its history is kept).")  # fmt: skip
        if event.confirm is None:
            if words[2:3] != ["confirm"]:
                return f"{question} Send /myagents delete {agent.name} confirm to do it."
        else:
            await session.commit()
            if await event.confirm(question, DELETE_CONFIRM_SECONDS) is not True:
                return f"Agent {agent.name!r} kept."
        await my_agents.retire(session, user.id, agent.id, actor=actor)
        return f"Agent {agent.name!r} disabled and its {count} scheduled task(s) deleted."
    except (InvalidInputError, NotFoundError, ConflictError) as exc:
        await session.rollback()
        return str(exc)


async def handle_myagents(session: AsyncSession, event: NormalizedEvent, decision):
    """`/myagents`: the sender's own agents; by email, a message whose text starts with
    it. Both log rows and the reply, like /task."""
    words = event.text.strip().split()[1:]
    agent = await resolve_agent(session, decision.user.id, decision.identity.active_agent_id)
    user_id, agent_id = decision.user.id, agent.id
    answer = await _myagents_answer(session, decision, event, words)
    for direction, entry in ((Direction.INBOUND, event.text.strip()), (Direction.OUTBOUND, answer)):
        await record_action(
            session, user_id=user_id, agent_id=agent_id, channel=event.channel,
            direction=direction, text=entry,
        )  # fmt: skip
    await session.commit()
    try:
        await event.reply(answer)
    except Exception:
        logger.exception("Replying to /myagents for %s failed", event.user_id)
        return DispatchOutcome.UNDELIVERED
    return DispatchOutcome.OK


async def _redeliver(session: AsyncSession, event: NormalizedEvent, pending) -> DispatchOutcome:
    """Send again an answer that was generated and recorded but not delivered.
    Success turns its entry from failed to ok; no second entry is written.
    """
    try:
        await event.reply(pending.text)
    except Exception:
        logger.exception(
            "Delivering the kept reply to %s/%s failed again", event.channel.value, event.user_id
        )
        return DispatchOutcome.UNDELIVERED
    pending.status = ActionStatus.OK
    await session.commit()
    logger.info(
        "Kept reply delivered to %s/%s without a new turn", event.channel.value, event.user_id
    )
    return DispatchOutcome.OK


def _agent_list_text(agents, current_id: int) -> str:
    parts = []
    for agent in agents:
        label = agent.name + ("*" if agent.id == current_id else "")
        parts.append(label if agent.is_active else f"{label} (disabled)")
    return "Your agents: " + ", ".join(parts) + ". * is the one you are talking to."


async def handle_agent_command(
    session: AsyncSession, event: NormalizedEvent, argument: str
) -> DispatchOutcome:
    """`/agent` lists the sender's agents, `/agent <name>` switches to one.
    The choice is stored on the channel identity, so it survives a restart and
    the conversation of each agent stays separate. A sender who is not
    authorized goes through the normal denial and request flow instead.
    """
    decision = await authorize(session, event.channel, event.user_id)
    if not decision.allowed:
        return await dispatch_event(session, event)

    user_id = decision.user.id
    current = await resolve_agent(session, user_id, decision.identity.active_agent_id)
    agents = await list_agents(session, user_id)
    name = argument.strip()
    if not name:
        answer = _agent_list_text(agents, current.id) + " Use /agent <name> to switch."
    else:
        chosen = await find_agent_by_name(session, user_id, name)
        if chosen is None:
            answer = f"No agent named {name!r}. " + _agent_list_text(agents, current.id)
        elif not chosen.is_active:
            answer = f"Agent {chosen.name!r} is disabled. " + _agent_list_text(agents, current.id)
        else:
            decision.identity.active_agent_id = chosen.id
            current = chosen
            answer = f"You are now talking to agent {chosen.name!r}."

    command = f"/agent {name}".strip()
    await record_action(
        session, user_id=user_id, agent_id=current.id, channel=event.channel,
        direction=Direction.INBOUND, text=command,
    )
    await record_action(
        session, user_id=user_id, agent_id=current.id, channel=event.channel,
        direction=Direction.OUTBOUND, text=answer,
    )
    await session.commit()
    try:
        await event.reply(answer)
    except Exception:
        logger.exception("Replying to /agent for %s/%s failed", event.channel.value, event.user_id)
        return DispatchOutcome.UNDELIVERED
    return DispatchOutcome.OK


ENGINE_MODELS_TIMEOUT_SECONDS = 5.0


async def engine_models() -> list[str] | None:
    """The models the inference engine offers (`/v1/models`): one in single-model mode, every
    file of the models directory in router mode. None when the engine does not answer."""
    import httpx

    from app.config import engine_headers, get_settings

    url = get_settings().llama_server_url.rstrip("/") + "/v1/models"
    try:
        async with httpx.AsyncClient(
            timeout=ENGINE_MODELS_TIMEOUT_SECONDS, headers=engine_headers()
        ) as client:
            response = await client.get(url)
        response.raise_for_status()
        return sorted(str(m["id"]) for m in response.json().get("data", []) if m.get("id"))
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _models_text(models: list[str]) -> str:
    return "Models: " + ", ".join(models) + ". Use /model <name> <message>."


async def handle_model_command(
    session: AsyncSession, event: NormalizedEvent, name: str, message: str
) -> DispatchOutcome:
    """`/model` lists the engine's models; `/model <name> <message>` answers that one message
    with that model (a one-shot choice, nothing is stored). The name
    must be one the engine offers. The message then goes through the normal pipeline, so an
    unauthorized sender gets the usual denial and access request, and the conversation keeps
    the message, not the command."""
    decision = await authorize(session, event.channel, event.user_id)
    if not decision.allowed:
        return await dispatch_event(session, event)
    name, message = name.strip(), message.strip()
    if name and message:
        models = await engine_models()
        if models is not None and name in models:
            chosen = NormalizedEvent(
                user_id=event.user_id,
                channel=event.channel,
                text=message,
                reply=event.reply,
                confirm=event.confirm,
                model=name,
            )
            return await dispatch_event(session, chosen)
    else:
        models = await engine_models()
    if models is None:
        answer = "The model list is not available right now (the engine does not answer)."
    elif not name:
        answer = _models_text(models) if models else "The engine offers no model."
    elif name not in models:
        answer = f"No model named {name!r}. " + _models_text(models)
    else:
        answer = f"Add the message after the name: /model {name} <message>."

    user_id = decision.user.id
    agent = await resolve_agent(session, user_id, decision.identity.active_agent_id)
    command = f"/model {name}".strip()
    await record_action(
        session, user_id=user_id, agent_id=agent.id, channel=event.channel,
        direction=Direction.INBOUND, text=command,
    )  # fmt: skip
    await record_action(
        session, user_id=user_id, agent_id=agent.id, channel=event.channel,
        direction=Direction.OUTBOUND, text=answer,
    )  # fmt: skip
    await session.commit()
    try:
        await event.reply(answer)
    except Exception:
        logger.exception("Replying to /model for %s/%s failed", event.channel.value, event.user_id)
        return DispatchOutcome.UNDELIVERED
    return DispatchOutcome.OK


async def _prompts_of(session: AsyncSession, user_id: int, agent) -> list[tuple]:
    """(server, prompt) of the MCP servers this user may reach through this agent, by the rule
    of its tools: the server is enabled, an administrator approved its definitions, the agent
    uses one of its tools, and the user holds a grant on it."""
    from app.admin import mcp as mcp_service
    from app.mcp.catalogue import NAME_PREFIX
    from app.mcp.manager import McpServerError
    from app.mcp.manager import manager as mcp_manager

    used = {
        name[len(NAME_PREFIX) :].split("__", 1)[0]
        for name in agent.tools or []
        if name.startswith(NAME_PREFIX)
    }
    if not used:
        return []
    grants = await mcp_service.grants_for(session, user_id, agent.id)
    granted = {server for server, _tool in grants}
    enabled = await mcp_service.enabled_configs(session)
    mcp_manager.configure(enabled)
    configs = [c for c in enabled if c.name in used & granted and c.approved_hashes]
    found = []
    for config in configs:
        server = mcp_manager.get(config.name)
        try:
            for prompt in await server.list_prompts():
                found.append((config.name, prompt))
        except McpServerError:
            logger.info("MCP server %s lists no prompts", config.name)
    return found


def _prompt_list_text(prompts: list[tuple]) -> str:
    if not prompts:
        return "No prompt is available to you."
    lines = [f"/prompt {p.name} — {p.description or ''}".strip() for _s, p in prompts]
    return "Prompts:\n" + "\n".join(lines)


async def handle_prompt_command(
    session: AsyncSession, event: NormalizedEvent, name: str, arguments: str
) -> DispatchOutcome:
    """`/prompt` lists the prompts of the MCP servers this user may reach through their agent;
    `/prompt <name> [key=value ...]` runs one: its text becomes the
    message of a normal turn. A prompt of a server the user has no grant for is not listed and
    cannot be run."""
    import shlex

    from app.mcp.manager import McpServerError
    from app.mcp.manager import manager as mcp_manager

    decision = await authorize(session, event.channel, event.user_id)
    if not decision.allowed:
        return await dispatch_event(session, event)
    user_id = decision.user.id
    agent = await resolve_agent(session, user_id, decision.identity.active_agent_id)
    prompts = await _prompts_of(session, user_id, agent)
    name = name.strip()
    answer = None
    if not name:
        answer = _prompt_list_text(prompts)
    else:
        matches = [(s, p) for s, p in prompts if p.name == name or f"{s}/{p.name}" == name]
        if len(matches) != 1:
            answer = (f"No prompt named {name!r}. " if not matches else
                      f"{name!r} is ambiguous, use server/name. ") + _prompt_list_text(prompts)
        else:
            server_name, prompt = matches[0]
            try:
                pairs = shlex.split(arguments)
            except ValueError:
                pairs = ["?"]
            values = dict(p.split("=", 1) for p in pairs if "=" in p)
            wanted_args = prompt.arguments or []
            missing = [a.name for a in wanted_args if a.required and a.name not in values]
            if len(values) != len(pairs) or missing:
                wanted = " ".join(f"{a.name}=..." for a in wanted_args)
                answer = f"Usage: /prompt {prompt.name} {wanted}".strip()
            else:
                try:
                    text = await mcp_manager.get(server_name).get_prompt(prompt.name, values)
                except McpServerError as exc:
                    logger.warning("Prompt %s/%s failed: %s", server_name, prompt.name, exc)
                    answer = "That prompt could not be run right now."
                else:
                    chosen = NormalizedEvent(
                        user_id=event.user_id, channel=event.channel, text=text,
                        reply=event.reply, confirm=event.confirm,
                    )  # fmt: skip
                    return await dispatch_event(session, chosen)
    command = f"/prompt {name}".strip()
    await record_action(
        session, user_id=user_id, agent_id=agent.id, channel=event.channel,
        direction=Direction.INBOUND, text=command,
    )  # fmt: skip
    await record_action(
        session, user_id=user_id, agent_id=agent.id, channel=event.channel,
        direction=Direction.OUTBOUND, text=answer,
    )  # fmt: skip
    await session.commit()
    try:
        await event.reply(answer)
    except Exception:
        logger.exception("Replying to /prompt for %s/%s failed", event.channel.value, event.user_id)
        return DispatchOutcome.UNDELIVERED
    return DispatchOutcome.OK


TASK_USAGE = (
    "Scheduled tasks: /task lists yours. /task add daily 08:30 <prompt>, "
    "/task add every 2h <prompt>, /task add cron 0 8 * * 1-5 <prompt>, or in words: "
    "/task add every Thursday at 9 give me the AI news; "
    "/task pause|resume|delete|run <id>; /task tz Europe/Paris sets your timezone."
)


def _task_line(task, timezone: str | None) -> str:
    from zoneinfo import ZoneInfo

    from app.tasks import _as_utc

    state = "on" if task.enabled else "off"
    upcoming = _as_utc(task.next_run_at)
    when = "-"
    if upcoming is not None:
        when = upcoming.astimezone(ZoneInfo(timezone or "UTC")).strftime("%Y-%m-%d %H:%M")
    prompt = task.prompt if len(task.prompt) <= 60 else task.prompt[:57] + "..."
    return f"#{task.id} [{state}] {task.kind} {task.expr}, next {when}: {prompt}"


async def _task_answer(session: AsyncSession, decision, event: NormalizedEvent, text: str) -> str:
    """The reply to one /task command. Every lookup carries the sender's user id, so
    another user's task is "not found"."""
    from app import tasks
    from app.admin.service import ConflictError, InvalidInputError, NotFoundError, update_user

    user = decision.user
    actor = f"user:{user.id}"
    words = text.split()
    action = words[0].lower() if words else "list"
    try:
        if action == "list":
            rows = await tasks.list_tasks(session, user.id)
            lines = [_task_line(t, user.timezone) for t in rows] or ["You have no task."]
            return "\n".join([*lines, f"Timezone: {user.timezone or 'UTC'}.", TASK_USAGE])
        if action == "add":
            kind = words[1].lower() if len(words) > 1 else ""
            fields = 5 if kind == "cron" else 1
            parts = text.split(None, 2 + fields)
            if len(words) > 1 and not _is_form(kind, parts[2 : 2 + fields]):
                # Words: "every Thursday at 9 ..." starts like the form "every 2h".
                words_text = text.split(None, 1)[1]
                return await _task_from_words(session, decision, event, words_text, actor)
            if len(parts) < 3 + fields:
                return TASK_USAGE
            task = await tasks.create_task(
                session,
                user_id=user.id,
                prompt=parts[-1],
                kind=kind,
                expr=" ".join(parts[2 : 2 + fields]),
                channel_identity_id=decision.identity.id,
                actor=actor,
                self_service=True,
            )
            return "Task created. " + _task_line(task, user.timezone)
        if action == "tz" and len(words) == 2:
            await update_user(session, user.id, timezone=words[1], actor=actor)
            return f"Your timezone is now {words[1]}; your tasks' next times follow it."
        if action in ("pause", "resume", "delete", "run") and len(words) == 2:
            if not words[1].isdigit():
                return TASK_USAGE
            task_id = int(words[1])
            if action == "delete":
                await tasks.delete_task(session, task_id, actor=actor, user_id=user.id)
                return f"Task #{task_id} deleted."
            if action == "run":
                await tasks.get_task(session, task_id, user.id)
                await session.commit()
                result = await tasks.execute(task_id, trigger="manual")
                return f"Task #{task_id} ran: {result['status']}."
            task = await tasks.update_task(
                session, task_id, {"enabled": action == "resume"}, actor=actor, user_id=user.id
            )
            return _task_line(task, user.timezone)
        return TASK_USAGE
    except (InvalidInputError, NotFoundError, ConflictError) as exc:
        await session.rollback()
        return str(exc)


def _is_form(kind: str, fields: list[str]) -> bool:
    """Whether `/task add` was given one of the three forms (daily HH:MM, every 2h, cron with
    five fields) rather than words."""
    from app import tasks
    from app.admin.service import InvalidInputError

    if kind not in tasks.KINDS or not fields:
        return False
    try:
        tasks.check_schedule(kind, " ".join(fields))
    except InvalidInputError:
        return False
    return True


TASK_CONFIRM_SECONDS = 120


async def _task_from_words(
    session: AsyncSession, decision, event: NormalizedEvent, words: str, actor: str
) -> str:
    """`/task add <phrase>`: the schedule and the prompt read from the words by the
    model, checked by the task parser (app.schedule_phrase, shared with the API), shown back
    with the next run times, and created only on the user's yes (the yes/no: buttons on
    Telegram, a reply code by email). A channel that cannot ask gets the exact form to send."""
    from app import schedule_phrase, tasks
    from app.admin.service import InvalidInputError

    user = decision.user
    try:
        parsed = await schedule_phrase.parse(words, user.timezone)
    except InvalidInputError:
        raise
    except Exception:
        # The engine is down or answered something unreadable: no task, a short reply.
        logger.exception("Reading a schedule in words failed for user %s", user.id)
        return "The schedule could not be read right now; try again, or use daily/every/cron."
    # Before asking: a schedule the user may not have is refused now, not after their yes.
    tasks.check_self_service_interval(parsed["kind"], parsed["expr"])
    if not parsed["prompt"]:
        return ("I found the schedule " f"{parsed['kind']} {parsed['expr']} but not what to do "
                "at each run: add it after the schedule words.")  # fmt: skip
    runs = schedule_phrase.describe_runs(parsed["next_runs"])
    zone = user.timezone or "UTC: set your timezone with /task tz Europe/Paris"
    reading = (f"I read: {parsed['kind']} {parsed['expr']}, next runs {runs} ({zone}), "
               f"prompt: {parsed['prompt']}")  # fmt: skip
    form = f"/task add {parsed['kind']} {parsed['expr']} {parsed['prompt']}"
    if event.confirm is None:
        return f"{reading}. To create it, send: {form}"
    answer = await event.confirm(f"{reading}.\nCreate this task?", TASK_CONFIRM_SECONDS)
    if answer is not True:
        return "Task not created." + (" (no answer in time)" if answer is None else "")
    task = await tasks.create_task(
        session, user_id=user.id, prompt=parsed["prompt"], kind=parsed["kind"],
        expr=parsed["expr"], channel_identity_id=decision.identity.id, actor=actor,
        self_service=True,
    )  # fmt: skip
    return "Task created. " + _task_line(task, user.timezone) + f"\nNext runs: {runs}"


async def handle_task_command(
    session: AsyncSession, event: NormalizedEvent, text: str
) -> DispatchOutcome:
    """`/task`: the sender's own scheduled tasks. A sender who is not authorized
    goes through the normal denial and request flow instead."""
    decision = await authorize(session, event.channel, event.user_id)
    if not decision.allowed:
        return await dispatch_event(session, event)
    agent = await resolve_agent(session, decision.user.id, decision.identity.active_agent_id)
    user_id, agent_id = decision.user.id, agent.id
    answer = await _task_answer(session, decision, event, text.strip())
    await record_action(
        session, user_id=user_id, agent_id=agent_id, channel=event.channel,
        direction=Direction.INBOUND, text=f"/task {text}".strip(),
    )  # fmt: skip
    await record_action(
        session, user_id=user_id, agent_id=agent_id, channel=event.channel,
        direction=Direction.OUTBOUND, text=answer,
    )  # fmt: skip
    await session.commit()
    try:
        await event.reply(answer)
    except Exception:
        logger.exception("Replying to /task for %s/%s failed", event.channel.value, event.user_id)
        return DispatchOutcome.UNDELIVERED
    return DispatchOutcome.OK


async def _own_command(session, event: NormalizedEvent, text: str, answer_for) -> DispatchOutcome:
    """The frame of a command on the sender's own data: authorization (an unknown
    sender goes through the normal denial and request flow), the agent this identity talks
    to, the answer, both log rows, the reply."""
    decision = await authorize(session, event.channel, event.user_id)
    if not decision.allowed:
        return await dispatch_event(session, event)
    agent = await resolve_agent(session, decision.user.id, decision.identity.active_agent_id)
    user_id, agent_id, agent_name = decision.user.id, agent.id, agent.name
    answer = await answer_for(decision.user.id, agent_id, agent_name)
    for direction, entry in ((Direction.INBOUND, text), (Direction.OUTBOUND, answer)):
        await record_action(
            session, user_id=user_id, agent_id=agent_id, channel=event.channel,
            direction=direction, text=entry,
        )  # fmt: skip
    await session.commit()
    try:
        await event.reply(answer)
    except Exception:
        logger.exception("Replying to %s for %s/%s failed", text, event.channel.value,
                         event.user_id)  # fmt: skip
        return DispatchOutcome.UNDELIVERED
    return DispatchOutcome.OK


async def handle_new_command(session: AsyncSession, event: NormalizedEvent) -> DispatchOutcome:
    """`/new`: a fresh conversation with the agent this identity talks to. Only this
    channel's conversation is forgotten; the audit log keeps every message."""
    from app.graph import build_thread_id, delete_threads, threads_with_history

    async def answer(user_id, agent_id, agent_name):
        thread = build_thread_id(event.channel, event.user_id, agent_id)
        had = await threads_with_history([thread])
        await delete_threads([thread])
        start = "New conversation" if had else "This is already a new conversation"
        return f"{start} with agent {agent_name!r}: earlier messages are no longer in its view."

    return await _own_command(session, event, "/new", answer)


async def handle_export_command(session: AsyncSession, event: NormalizedEvent) -> DispatchOutcome:
    """`/export`: the sender's own conversation with their current agent, as a text file
    holding Markdown; never another user's (the export is keyed on the sender's id)."""
    from app.admin.service import conversation_markdown, export_conversation

    async def answer(user_id, agent_id, agent_name):
        messages, about = await export_conversation(
            session, user_id, agent_id=agent_id, include_text=True, fmt="markdown",
            actor=f"user:{user_id}",
        )  # fmt: skip
        if not messages:
            return f"There is no message to export with agent {agent_name!r} yet."
        if event.send_file is None:
            return (f"{len(messages)} messages with agent {agent_name!r}; this channel cannot "
                    "send a file: ask an administrator for the export.")  # fmt: skip
        # .txt, not .md: the owner's Telegram app could not open a .md file (2026-09-28); the
        # content is the same Markdown, readable as plain text everywhere.
        name = f"conversation-{agent_name}.txt"
        await event.send_file(name, conversation_markdown(messages, about).encode("utf-8"))
        return f"{len(messages)} messages with agent {agent_name!r} exported in {name}."

    return await _own_command(session, event, "/export", answer)


async def dispatch_text(session: AsyncSession, event: NormalizedEvent) -> DispatchOutcome | None:
    """A channel without its own command menu (the terminal): the commands Telegram
    routes by its menu (/agent, /model, /prompt, /task, /new, /export) are recognised from the
    text as typed, the same handlers; anything else, the agent builder's commands included,
    goes to `dispatch_event`."""
    text = event.text.strip()
    command, _sep, rest = text.partition(" ")
    command = command.lower()
    if command == "/agent":
        return await handle_agent_command(session, event, rest.strip())
    if command in ("/model", "/prompt"):
        name, _sep, tail = rest.strip().partition(" ")
        handler = handle_model_command if command == "/model" else handle_prompt_command
        event.text = f"{command} {name}".strip()
        return await handler(session, event, name, tail.strip())
    if command == "/task":
        return await handle_task_command(session, event, rest.strip())
    if command == "/new" and not rest.strip():
        return await handle_new_command(session, event)
    if command == "/export" and not rest.strip():
        return await handle_export_command(session, event)
    return await dispatch_event(session, event)
