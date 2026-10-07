"""A completion read from the engine while it is generated.

`complete(client, body, on_text)` sends `body` to `/v1/chat/completions` with `stream: true` and
assembles the server-sent events into the same shape as a non-streamed response
(`{"choices": [{"message": ..., "finish_reason": ...}], "usage": ..., "model": ...}`), so the
callers read one shape either way. While it runs, `on_text(text)` receives the visible text so
far: the `content` deltas only, never `reasoning_content`, with any `<think>` block removed.
Tool calls arrive as fragments (an index, then pieces of the arguments) and are joined here.

`on_text` is a display: it must not fail the turn. An exception it raises is logged once and
the display stops; the completion goes on.
"""

import json
import logging
from collections.abc import Awaitable, Callable

import httpx

from app import replies

logger = logging.getLogger("channelagent")

TextSink = Callable[[str], Awaitable[None]]


async def complete(client: httpx.AsyncClient, body: dict, on_text: TextSink | None) -> dict:
    payload = {**body, "stream": True, "stream_options": {"include_usage": True}}
    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        if response.status_code >= 400:
            await response.aread()
            response.raise_for_status()
        return await _assemble(response, on_text)


async def _assemble(response: httpx.Response, on_text: TextSink | None) -> dict:
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict] = {}
    finish_reason = None
    usage = None
    timings = None
    model = None
    shown = ""
    display = on_text
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        usage = chunk.get("usage") or usage
        timings = chunk.get("timings") or timings  # The engine's cache figures
        model = chunk.get("model") or model
        for choice in chunk.get("choices") or []:
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            for fragment in delta.get("tool_calls") or []:
                call = calls.setdefault(
                    fragment.get("index", len(calls)),
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                call["id"] = fragment.get("id") or call["id"]
                function = fragment.get("function") or {}
                call["function"]["name"] += function.get("name") or ""
                call["function"]["arguments"] += function.get("arguments") or ""
            if delta.get("content"):
                content.append(delta["content"])
                visible = replies.strip_thinking("".join(content))
                if display is not None and visible and visible != shown:
                    shown = visible
                    try:
                        await display(visible)
                    except Exception:
                        logger.warning("Showing the reply as it is written failed", exc_info=True)
                        display = None
    message: dict = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    body: dict = {"choices": [{"message": message, "finish_reason": finish_reason}]}
    if usage is not None:
        body["usage"] = usage
    if model is not None:
        body["model"] = model
    if timings is not None:
        body["timings"] = timings
    return body
