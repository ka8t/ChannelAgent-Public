"""The LangGraph orchestrator: the single agent loop every
channel routes through, replacing the earlier agent loop, with
conversation state isolated per user via LangGraph's native
checkpointer.

Call build_thread_id(channel, user_id, agent_id) to get the checkpointer
key for a given identity+agent, then invoke the compiled graph with it
in config["configurable"]. See for wiring the response back to the
originating channel adapter (not this module's job).
"""

import asyncio
import contextvars
import logging
import os
import time
from collections.abc import Iterable
from typing import Annotated, TypedDict

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    convert_to_openai_messages,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import checkpoints, engine_stream, memory, recall, replies, slot_cache
from app.admin import mcp as mcp_service
from app.admin import routing as routing_service
from app.admin import skills
from app.config import engine_headers, get_settings
from app.db.models import Agent, Channel
from app.db.session import session_scope
from app.mcp import catalogue as mcp_catalogue
from app.mcp.manager import manager as mcp_manager
from app.security.hashing import channel_identifier_key
from app.tools import run_tool_loop, tools_are_supported

logger = logging.getLogger("channelagent")


class GraphState(TypedDict):
    # add_messages appends each node's returned messages to the
    # checkpointed history instead of replacing it — without this
    # reducer, every turn would overwrite the prior conversation
    # instead of continuing it, defeating the point of the checkpointer.
    messages: Annotated[list[BaseMessage], add_messages]
    # Running summary of the turns that fell out of the model's window, and how
    # many messages it covers. Absent in older checkpoints.
    summary: str
    summary_covers: int


def build_thread_id(channel: Channel, user_id: str, agent_id: int) -> str:
    """thread_id scheme, extended for multiple agents per user:
    {channel}_{identity_key}_{agent_id} — e.g. telegram_123_4,
    email_{hash}_4. Same deterministic identity key as the Auth Node
    (app/security/auth.py) so a given (identity, agent) pair always
    maps to the same conversation thread, and two agents belonging to
    the same user never share one.
    """
    return thread_id_from_key(channel, channel_identifier_key(channel, user_id), agent_id)


def thread_id_from_key(channel: Channel, identity_key: str, agent_id: int) -> str:
    """The same id from an identity key already computed (ChannelIdentity.
    external_id is exactly that key), so code that only has the database row,
    such as a user purge, builds the identical thread id.
    """
    return f"{channel.value}_{identity_key}_{agent_id}"


# Share of the model's context window the conversation may fill. The rest is
# left for the reply and for any error of the token count.
HISTORY_CONTEXT_SHARE = 0.75

# Tokens added around each message by the chat template (role markers), counted
# on top of the message text when the server's tokenizer is used.
MESSAGE_OVERHEAD_TOKENS = 4

# The turns that fall out of the window are summarized, but only once at
# least this many are new since the last summary, so a long conversation does
# not pay one extra model call per turn.
SUMMARY_BATCH = 6
SUMMARY_MAX_TOKENS = 300
SUMMARY_PROMPT = (
    "You maintain the memory of a conversation. Write a summary of at most 200 words. "
    "Keep every concrete fact the user gave (names, numbers, identifiers, dates, "
    "preferences, decisions) exactly as written, and never drop an item of the earlier "
    "summary. Leave out small talk. Write only the summary."
)
SUMMARY_HEADER = "Summary of the earlier part of this conversation: "

# Exact token counts already obtained from the server, by message id, and the
# time until which a server without a usable tokenizer is not asked again.
_token_cache: dict[str, int] = {}
_TOKENIZER_RETRY_SECONDS = 300
_tokenizer_down_until = 0.0


def history_token_budget(ctx_size: int) -> int:
    return int(ctx_size * HISTORY_CONTEXT_SHARE)


# The window is cut by steps of this many messages, so its start stays the same from one
# turn to the next until the next step and the engine reuses its cache of the conversation;
# cut one turn at a time, every message after the system one changed at each turn. The
# summary folds the same steps.
WINDOW_STEP = SUMMARY_BATCH


def stepped_start(messages: list[BaseMessage], minimal: int) -> int:
    """The first message sent: `minimal` (the start the budget needs) rounded up to the next
    step, then moved to a human message; never past the latest message."""
    if minimal <= 0:
        return 0
    start = -(-minimal // WINDOW_STEP) * WINDOW_STEP
    while start < len(messages) - 1 and not isinstance(messages[start], HumanMessage):
        start += 1
    return min(start, len(messages) - 1)


def window_messages(
    messages: list[BaseMessage], budget: int, token_counter=count_tokens_approximately
) -> list[BaseMessage]:
    """The most recent turns that fit in `budget` tokens.

    Starts on a human message so a reply is never sent without its question,
    and always keeps the latest message, even alone above the budget. Only
    what is sent to the model is cut: the checkpoint keeps the whole history
    and the audit trail (ActionLog) is untouched. `token_counter` defaults to
    the characters-per-token estimate of langchain; call_llm passes exact counts
    from the server's tokenizer when it answers.
    """
    kept = trim_messages(
        messages,
        max_tokens=budget,
        token_counter=token_counter,
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
    return kept or messages[-1:]


async def _exact_counter(client: httpx.AsyncClient, messages: list[BaseMessage]):
    """A token counter using the server's own tokenizer (`/tokenize`), or the
    estimate when the server does not offer it. Each message is tokenized once,
    then remembered by its id.
    """
    global _tokenizer_down_until
    if time.monotonic() < _tokenizer_down_until:
        return count_tokens_approximately
    counts: dict[str, int] = {}
    try:
        for message in messages:
            key = message.id
            if key is not None and key in _token_cache:
                counts[key] = _token_cache[key]
                continue
            response = await client.post("/tokenize", json={"content": str(message.content)})
            response.raise_for_status()
            n_tokens = len(response.json()["tokens"]) + MESSAGE_OVERHEAD_TOKENS
            counts[key or str(id(message))] = n_tokens
            if key is not None:
                _token_cache[key] = n_tokens
    except Exception:
        _tokenizer_down_until = time.monotonic() + _TOKENIZER_RETRY_SECONDS
        logger.debug("The server's tokenizer is not usable, estimating token counts", exc_info=True)
        return count_tokens_approximately

    def counter(batch: list[BaseMessage]) -> int:
        return sum(
            counts.get(m.id or str(id(m)), 0) or count_tokens_approximately([m]) for m in batch
        )

    return counter


def _prefix_cost(counter, prefix: str) -> int:
    return counter([SystemMessage(content=prefix)]) if prefix else 0


async def _complete(client: httpx.AsyncClient, request: dict, on_text) -> dict:
    if on_text is not None:
        body = await engine_stream.complete(client, request, on_text)
        _add_usage(body)
        return body
    response = await client.post("/v1/chat/completions", json={**request, "stream": False})
    response.raise_for_status()
    return response.json()


async def _chat(
    client: httpx.AsyncClient, messages: list[dict], on_text=None, **extra
) -> str:
    """One completion. With `on_text`It is streamed: the visible text is handed to
    `on_text` while it is written, and the usage is counted here (the response hook does not
    read a stream)."""
    body = await _complete(client, {"messages": messages, **extra}, on_text)
    if replies.thought_to_the_cap(body) and not replies.thinking_is_off(extra):
        logger.warning("The model thought until the token cap: asking again without thinking")
        request = {"messages": messages, **extra, **replies.NO_THINKING}
        body = await _complete(client, request, on_text)
    choice = body["choices"][0]
    content = choice["message"].get("content")
    if not (content or "").strip():
        # Why a reply came back empty: the engine's own figures, never the text.
        logger.warning(
            "The model returned no visible text: finish_reason=%s, completion_tokens=%s, "
            "reasoning_chars=%s",
            choice.get("finish_reason"),
            (body.get("usage") or {}).get("completion_tokens"),
            len(choice["message"].get("reasoning_content") or ""),
        )
    return content


def thinking_fields(setting: str | None) -> dict:
    """The request fields of LLM_THINKING: `auto` sends nothing (the model's own
    default), `on` and `off` ask the chat template to think or not. Measured on the installed
    model: 7.1 s and 50 tokens by default, 1.2 s and 4 tokens with `off`, the same answer."""
    if setting in ("on", "off"):
        return {"chat_template_kwargs": {"enable_thinking": setting == "on"}}
    return {}


async def _chat_with_fallback(
    client: httpx.AsyncClient,
    messages: list[dict],
    model: str | None,
    default_model: str | None,
    fields: dict | None = None,
    on_text=None,
) -> str:
    """The turn's reply: a model that the engine no longer has (deleted,
    renamed, or never installed) answers 400, in router mode as well as single-
    model mode; retried once against the routing default, and the fallback is
    logged, never silent. Nothing to fall back to (no default, or it *is* the
    one that failed) lets the error surface as any other failed turn.
    """
    fields = fields or {}
    try:
        return await _chat(
            client, messages, on_text, **fields, **({"model": model} if model else {})
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 400 and model and default_model and model != default_model:
            logger.warning(
                "Model %r is unavailable, falling back to the routing default %r",
                model, default_model,
            )
            return await _chat(client, messages, on_text, **fields, model=default_model)
        raise


async def _summarize(
    client: httpx.AsyncClient, previous: str, dropped: list[BaseMessage], **extra
) -> str | None:
    """Fold the dropped turns into the running summary with one model call.
    Returns None when the call fails: the turn then goes on with the window alone.
    """
    transcript = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in dropped
    )
    text = (f"Earlier summary: {previous}\n\n" if previous else "") + transcript
    try:
        return (
            await _chat(
                client,
                [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": text}],
                max_tokens=SUMMARY_MAX_TOKENS,
                **extra,
            )
        ).strip()
    except Exception:
        logger.warning(
            "Summarizing the earlier conversation failed, using the window alone", exc_info=True
        )
        return None


# Telemetry of the running turn: the model that answered and the tokens the engine
# reported for every completion of the turn. A dict held by a context variable and mutated in
# place, so what the node adds is seen by run_turn even if the node runs in a copied context;
# run_turn adds the latency, and `last_turn_stats` hands the whole to the caller (dispatch).
_turn_stats: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "turn_stats", default=None
)
last_turn_stats: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "last_turn_stats", default=None
)


async def _count_usage(response: httpx.Response) -> None:
    """Response hook of the engine client: adds a completion's usage to the turn's stats.
    A streamed completion is left alone: reading it here would consume the stream
    before anyone sees it; engine_stream's caller counts it with `_add_usage`."""
    if not response.url.path.endswith("/chat/completions") or response.status_code != 200:
        return
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        return
    if _turn_stats.get() is None:
        return
    try:
        await response.aread()
        _add_usage(response.json())
    except (ValueError, TypeError, AttributeError):
        pass


def _add_usage(body: dict) -> None:
    """Adds one completion's usage and model to the running turn's stats."""
    stats = _turn_stats.get()
    if stats is None:
        return
    try:
        usage = body.get("usage") or {}
        stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        # How much of the prompt the engine took from its cache, and how long it spent
        # reading the rest (llama-server `timings`, field names checked on build b10735).
        timings = body.get("timings") or {}
        cached = timings.get("cache_n")
        if cached is None:
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        stats["cached_tokens"] = stats.get("cached_tokens", 0) + int(cached or 0)
        stats["prefill_ms"] = stats.get("prefill_ms", 0.0) + float(timings.get("prompt_ms") or 0)
        if body.get("model"):
            # The file name only: in single-model mode the engine reports the full path, and
            # the directory layout is kept out of the logs, as in GET /status (measured).
            stats["model"] = os.path.basename(str(body["model"]))[:200]
    except (ValueError, TypeError, AttributeError):
        pass


# Where the visible text of the running turn's answer goes while it is written: set by
# the channel (app.channels.dispatch) when it can show progress, None otherwise. Only the
# answer is streamed, never a summary or a capability probe.
reply_stream: contextvars.ContextVar = contextvars.ContextVar("reply_stream", default=None)


# The configuration of the agent whose turn is running: its system prompt and model.
# Handed to the node through a context variable and not through the graph's `configurable`,
# because LangGraph copies those values into the checkpoint's metadata, and the prompt is
# encrypted at rest for a reason.
# How long one request to the engine may take. A chat turn keeps CHAT_LLM_TIMEOUT: a
# user waits for it. A scheduled task's turn (app.tasks.execute) sets LLM_TASK_TIMEOUT_SECONDS:
# nobody waits, and a digest of a feed took 94 to 509 s on the owner's model (measured).
CHAT_LLM_TIMEOUT = 120
llm_timeout: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "llm_timeout", default=None
)

_agent_settings: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "agent_settings", default=None
)


def _prefix(system_prompt: str | None, summary: str, memory: str = "") -> str:
    """The one system message in front of the window: the agent's prompt, its memory block,
    then the running summary. One message, because several chat templates refuse a
    second system message."""
    parts = [system_prompt] if system_prompt else []
    if memory:
        parts.append(memory)
    if summary:
        parts.append(SUMMARY_HEADER + summary)
    return "\n\n".join(parts)


async def call_llm(state: GraphState) -> GraphState:
    """Send the conversation so far to the configured LLM gateway and
    return its reply as a new message (add_messages appends it).

    Talks to llama-server's OpenAI-compatible /v1/chat/completions
    endpoint — works unchanged against the native Mac llama-server or a
    containerized Ollama/vLLM backend, since only LLAMA_SERVER_URL
    differs between them.

    The request holds the most recent turns that fit the budget (counted
    with the server's tokenizer when it has one) and, in front of them, a
    running summary of the turns that fell out of the window. Both are
    only what is sent: the checkpoint keeps every message.
    """
    settings = get_settings()
    messages = state["messages"]
    summary = state.get("summary", "")
    covers = state.get("summary_covers", 0)
    update: dict = {}
    agent = _agent_settings.get() or {}
    system_prompt, model = agent.get("system_prompt"), agent.get("model")
    default_model = agent.get("default_model")
    agent_id = agent.get("agent_id")
    owner_id = agent.get("user_id")
    tools_allowed = agent.get("tools_allowed") or set()
    tool_grants = agent.get("tool_grants") or frozenset()
    memory_mode = agent.get("memory_mode") or "off"
    memory_block = agent.get("memory_block") or ""
    memory_turn = agent.get("memory_turn") or ""
    granted_skills = agent.get("skills") or []
    # A model's own context size only ever narrows the engine-wide cap
    # LLAMA_CTX_SIZE, never widens it: that cap is what every router-loaded model
    # is actually started with.
    ctx_size = min(agent.get("ctx_size") or settings.llama_ctx_size, settings.llama_ctx_size)
    budget = history_token_budget(ctx_size)
    extra = {"model": model} if model else {}
    thinking = thinking_fields(settings.llm_thinking)
    on_text = reply_stream.get()

    async with httpx.AsyncClient(
        base_url=settings.llama_server_url,
        headers=engine_headers(settings),
        timeout=llm_timeout.get() or CHAT_LLM_TIMEOUT,
        event_hooks={"response": [_count_usage]},
    ) as client:
        counter = await _exact_counter(client, messages)
        memory_tools = memory.tool_definitions(memory_mode) if owner_id is not None else []
        wants_tools = bool(tools_allowed or memory_tools or granted_skills)
        supported = wants_tools and await tools_are_supported(client)
        # Skills: their names and descriptions, only when the agent can load one.
        memory_block = "\n\n".join(
            part
            for part in (memory_block, skills.index(granted_skills) if supported else "")
            if part
        )
        room = budget - _prefix_cost(counter, _prefix(system_prompt, summary, memory_block))
        room -= counter([HumanMessage(content=memory_turn)]) if memory_turn else 0
        window = window_messages(messages, max(room, 1), counter)
        # Never before what the summary already covers: a shorter new summary left room, and
        # the window moved back over summarised messages, sent twice (measured 2026-10-04: 37 of
        # 43 sent with 12 summarised), which also changed the start the engine had cached.
        floor = min(covers, len(messages) - 1) if summary else 0  # covered by a real summary
        dropped = max(stepped_start(messages, len(messages) - len(window)), floor)
        window = messages[dropped:]
        if dropped > covers and dropped - covers >= SUMMARY_BATCH:
            folded = await _summarize(client, summary, messages[covers:dropped], **extra)
            if folded:
                summary, covers = folded, dropped
                update = {"summary": summary, "summary_covers": covers}
                room = budget - _prefix_cost(counter, _prefix(system_prompt, summary, memory_block))
                room -= counter([HumanMessage(content=memory_turn)]) if memory_turn else 0
                window = window_messages(messages, max(room, 1), counter)
                start = stepped_start(messages, len(messages) - len(window))
                window = messages[max(start, min(covers, len(messages) - 1)) :]  # folded now
        if dropped:
            logger.info(
                "Conversation trimmed for the model: %d of %d messages sent (%d summarized)",
                len(window), len(messages), covers,
            )
        # The recall archive: offered when part of this conversation has left the
        # window, so the model can look up what it no longer sees.
        out_of_window = messages[: len(messages) - len(window)]
        wants_recall = bool(out_of_window) and owner_id is not None and agent_id is not None
        if wants_recall and not supported:
            supported = await tools_are_supported(client)
        turn_messages = list(window)
        if memory_turn and window and isinstance(window[-1], HumanMessage):
            # The memory entries of this turn, in the copy of the new message only.
            latest = window[-1]
            turn_messages[-1] = HumanMessage(content=f"{memory_turn}\n\n{latest.content}")
        sent = turn_messages
        prefix = _prefix(system_prompt, summary, memory_block)
        if prefix:
            sent = [SystemMessage(content=prefix), *turn_messages]
        catalogue = (
            await mcp_catalogue.build_tools(mcp_manager, tools_allowed, tool_grants)
            if supported and tools_allowed
            else mcp_catalogue.Catalogue()
        )
        # The tool budget: only the tools this message is most likely to need.
        mcp_catalogue.narrow(
            catalogue,
            routing_service.select_tools(
                text=agent.get("turn_text") or "",
                tools=catalogue.tools,
                rules=agent.get("tool_rules") or [],
                max_tools=agent.get("max_tools") or 0,
            ),
        )
        skill_tools = [skills.tool_definition()] if supported and granted_skills else []
        recall_tools = [recall.tool_definition()] if supported and wants_recall else []
        if recall_tools:
            # Told once in the system message, as the memory tools are (measured on the real
            # engine: without it the model answered "I don't have it" in 1 run of 3).
            prefix = "\n\n".join(part for part in (prefix, recall.GUIDANCE) if part)
            sent = [SystemMessage(content=prefix), *turn_messages]
        tools = [*memory_tools, *skill_tools, *recall_tools, *catalogue.tools] if supported else []
        # The turn is pinned to an engine slot, whose saved prefix (the conversation
        # without this turn's memory hits) it extends; `stable` is that conversation as the
        # next turn will send it.
        request = {**extra, **thinking, **({"tools": tools} if tools else {})}
        stable = [
            *([SystemMessage(content=prefix)] if prefix else []), *window,
            AIMessage(content="ok"), HumanMessage(content="next"),
        ]  # fmt: skip
        async with slot_cache.pinned(
            client,
            agent.get("thread_id"),
            {**request, "messages": convert_to_openai_messages(sent)},
            {**request, "messages": convert_to_openai_messages(stable)},
        ) as pin:
            if tools:
                mcp_executor = mcp_catalogue.make_executor(
                    mcp_manager, catalogue, agent_id, owner_id
                )
                memory_executor = memory.make_executor(owner_id, agent_id) if memory_tools else None
                skill_executor = skills.make_executor(agent_id) if skill_tools else None
                recall_executor = (
                    recall.make_executor(
                        owner_id, agent_id, agent.get("thread_id") or "", out_of_window,
                        len(messages),
                    )  # fmt: skip
                    if recall_tools
                    else None
                )

                async def executor(name: str, raw_arguments: str) -> str:
                    if skill_executor is not None and name == skills.LOAD_TOOL:
                        return await skill_executor(name, raw_arguments)
                    if recall_executor is not None and name == recall.TOOL_NAME:
                        return await recall_executor(name, raw_arguments)
                    # Memory tools are this application's own, never an MCP name.
                    if memory_executor is not None and memory.is_memory_tool(name):
                        return await memory_executor(name, raw_arguments)
                    return await mcp_executor(name, raw_arguments)

                reply, _rounds = await run_tool_loop(
                    client, convert_to_openai_messages(sent), tools, executor,
                    on_text=on_text, on_usage=_add_usage, **extra, **thinking, **pin,
                )  # fmt: skip
            else:
                reply = await _chat_with_fallback(
                    client, convert_to_openai_messages(sent), model, default_model,
                    {**thinking, **pin}, on_text,
                )

    # A thinking block is removed; a reply that loops fails the turn instead of
    # being stored and sent (DegenerateReplyError).
    reply = replies.check(reply)
    if not reply.strip():
        # Raised here, before the checkpoint stores the step: an empty AI message stored after
        # the user's message made the retry append that message again (2026-10-04: three
        # attempts left question, empty, question, empty, question in the conversation).
        raise replies.EmptyReplyError("the model returned an empty reply")
    return {"messages": [AIMessage(content=reply)], **update}


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("call_llm", call_llm)
    graph.add_edge(START, "call_llm")
    graph.add_edge("call_llm", END)
    return graph


# The compiled graph and its checkpointer are created on first use: the
# checkpointer owns an aiosqlite connection, which belongs to the event loop
# that opened it. Reopened when the configured file changes, closed by
# close_graph() (application shutdown, and between tests).
_state: dict = {}
_lock = asyncio.Lock()


async def get_graph():
    path = checkpoints.checkpoint_db_path()
    async with _lock:
        if _state.get("path") != path:
            await _close_locked()
            saver = await checkpoints.open_saver(path)
            graph = build_graph().compile(checkpointer=saver)
            _state.update(path=path, saver=saver, graph=graph)
            logger.info("Conversation checkpoints stored in %s", path)
        return _state["graph"]


async def _close_locked() -> None:
    saver = _state.get("saver")
    _state.clear()
    if saver is not None:
        try:
            await saver.conn.close()
        except Exception:
            logger.debug("Closing the checkpoint database raised, ignoring", exc_info=True)


async def close_graph() -> None:
    async with _lock:
        await _close_locked()


async def threads_with_history(thread_ids: Iterable[str]) -> int:
    """How many of these conversations have a stored checkpoint. A checkpoint
    that cannot be read counts: it exists, and that is the case an admin
    resets.
    """
    ids = list(thread_ids)
    if not ids:
        return 0
    await get_graph()
    saver = _state["saver"]
    found = 0
    for thread_id in ids:
        try:
            found += await saver.aget_tuple({"configurable": {"thread_id": thread_id}}) is not None
        except Exception:
            found += 1
    return found


async def delete_threads(thread_ids: Iterable[str]) -> int:
    """Delete the checkpoints of these conversations. Returns how many thread
    ids were processed (a thread with no checkpoint is not an error).
    """
    ids = list(thread_ids)
    if not ids:
        return 0
    await get_graph()
    saver = _state["saver"]
    for thread_id in ids:
        await saver.adelete_thread(thread_id)
    # The engine cache of these conversations, on disk and in the engine's slots.
    settings = get_settings()
    async with httpx.AsyncClient(
        base_url=settings.llama_server_url, headers=engine_headers(settings), timeout=10
    ) as client:
        await slot_cache.forget(client, ids)
    return len(ids)


async def run_turn(
    channel: Channel,
    user_id: str,
    agent_id: int,
    text: str,
    *,
    retry: bool = False,
    model: str | None = None,
    thread_id: str | None = None,
) -> str:
    """Entry point channel adapters call after a message passes
    authorization (app/security/auth.py). Returns the assistant's reply.

    `retry` marks a message that is processed again after a failed turn:
    a failed turn leaves its user message in the checkpoint, so when the last
    stored message is that same user message it is not appended a second time.

    `model` is the sender's choice for this one message (`/model`): it comes before
    the agent's own model.

    `thread_id` replaces the conversation of (channel, user, agent): a scheduled task runs
    in a conversation of its own (app/tasks.py).
    """
    thread_id = thread_id or build_thread_id(channel, user_id, agent_id)
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    async with session_scope() as session:
        agent = await session.get(Agent, agent_id)
        if agent is not None:
            routing = await routing_service.get_routing(session)
            # Only queried when the agent might actually use one: most agents
            # have no tools, and every turn otherwise pays a query for nothing.
            tool_grants = frozenset()
            granted_skills = await skills.granted(session, agent)
            memory_block, memory_turn = await memory.injection_parts(
                session, agent.user_id, agent.id, agent.memory_mode, text
            )
            if agent.tools:
                mcp_manager.configure(await mcp_service.enabled_configs(session))
                tool_grants = await mcp_service.grants_for(session, agent.user_id, agent.id)
    if agent is None:
        _agent_settings.set({})
    else:
        # Decision order: the sender's choice for this message (`/model`), then the
        # agent's own model, then the model of tool turns for an agent that uses MCP
        # tools; only when all are empty are the routing rules, then the default
        # model, consulted.
        uses_mcp = any(name.startswith(mcp_catalogue.NAME_PREFIX) for name in agent.tools or [])
        model = (
            model
            or agent.model
            or (routing["tool_model"] if uses_mcp else None)
            or routing_service.select_model(
                text=text, rules=routing["rules"], default_model=routing["default_model"]
            )
        )
        _agent_settings.set(
            {
                "system_prompt": agent.system_prompt,
                "model": model,
                "default_model": routing["default_model"],
                "ctx_size": routing["model_ctx_sizes"].get(model) if model else None,
                "agent_id": agent.id,
                "user_id": agent.user_id,
                "tools_allowed": set(agent.tools),
                "tool_grants": tool_grants,
                "memory_mode": agent.memory_mode,
                "memory_block": memory_block,
                "memory_turn": memory_turn,
                "skills": granted_skills,
                "max_tools": routing["max_tools"],
                "tool_rules": routing["tool_rules"],
                "turn_text": text,
                "thread_id": thread_id,
            }
        )
    new_messages = [HumanMessage(content=text)]
    if retry:
        state = await graph.aget_state(config)
        stored = state.values.get("messages", []) if state and state.values else []
        if stored and isinstance(stored[-1], HumanMessage) and stored[-1].content == text:
            new_messages = []
    stats = {"model": None, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
             "prefill_ms": 0.0}  # fmt: skip
    _turn_stats.set(stats)
    last_turn_stats.set(None)
    started = time.monotonic()
    result = await graph.ainvoke({"messages": new_messages}, config=config)
    stats["latency_ms"] = int((time.monotonic() - started) * 1000)
    if stats["model"] is None:
        stats["model"] = model if agent is not None else None
    last_turn_stats.set(stats)
    return result["messages"][-1].content
