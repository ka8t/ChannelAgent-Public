"""Tests: the tool-calling foundation — a capability probe (does the
engine return a parsed `tool_calls` at all) and the loop that runs while it keeps
asking for more of them (rounds cap, identical-call guard, timeout, truncation,
cancellation). No MCP yet: the loop is exercised directly, not through a
conversation turn.
"""

import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from app import tools


class _Scripted(BaseHTTPRequestHandler):
    """Replies from a queue, one per POST, so a test can script a multi-round
    exchange. A response is either a dict (200, that JSON body) or an int (that
    status code, an empty JSON error body).
    """

    responses: list = []
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        type(self).requests.append(body)
        reply = type(self).responses.pop(0)
        if isinstance(reply, int):
            data = json.dumps({"error": "scripted failure"}).encode()
            self.send_response(reply)
        else:
            data = json.dumps(reply).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    _Scripted.responses = []
    _Scripted.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), _Scripted)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    tools.reset_capability_check()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    tools.reset_capability_check()


def _message(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _call(name, arguments, call_id="call-1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# --- the capability probe ---


async def test_the_probe_detects_a_supported_engine(server):
    _Scripted.responses = [_message(tool_calls=[_call("_capability_probe", "{}")])]
    async with httpx.AsyncClient(base_url=server) as client:
        assert await tools.tools_are_supported(client) is True
    body = _Scripted.requests[0]
    assert body["tool_choice"] == "required"
    assert body["tools"][0]["function"]["name"] == "_capability_probe"
    # A reasoning model needs room to think before the call (2 of 10 probes of the owner's
    # model failed with 50 tokens, 0 of 10 with 200).
    assert body["max_tokens"] >= 200


async def test_the_probe_detects_an_unsupported_engine_and_logs_once(server, caplog):
    _Scripted.responses = [_message(content='{"name": "_capability_probe"}')]
    async with httpx.AsyncClient(base_url=server) as client:
        with caplog.at_level(logging.WARNING, logger="channelagent"):
            assert await tools.tools_are_supported(client) is False
        # Cached: a second call makes no second request and logs nothing again.
        assert await tools.tools_are_supported(client) is False
    assert len(_Scripted.requests) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


async def test_an_engine_failure_is_not_kept_and_the_probe_runs_again_later(server, monkeypatch):
    """Live, 2026-10-04: a probe made while the engine was busy failed and turned tools off for
    the whole session. A failure is now tried again after PROBE_RETRY_SECONDS."""
    clock = [1000.0]
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    _Scripted.responses = [500, _message(tool_calls=[_call("_capability_probe", "{}")])]
    async with httpx.AsyncClient(base_url=server) as client:
        assert await tools.tools_are_supported(client) is False
        assert await tools.tools_are_supported(client) is False, "not probed again at once"
        assert len(_Scripted.requests) == 1
        clock[0] += tools.PROBE_RETRY_SECONDS + 1
        assert await tools.tools_are_supported(client) is True
        assert await tools.tools_are_supported(client) is True
    assert len(_Scripted.requests) == 2, "a verdict is kept"
    assert _Scripted.requests[1]["chat_template_kwargs"] == {"enable_thinking": False}


# --- the tool loop ---


async def test_the_loop_stops_immediately_when_no_tool_is_called(server):
    _Scripted.responses = [_message(content="no tool needed")]
    calls = []

    async def executor(name, arguments):
        calls.append((name, arguments))
        return "unused"

    async with httpx.AsyncClient(base_url=server) as client:
        reply, rounds = await tools.run_tool_loop(client, [], [], executor)
    assert reply == "no tool needed"
    assert rounds == 1
    assert calls == []


async def test_a_tool_is_executed_and_its_result_reaches_the_next_round(server):
    _Scripted.responses = [
        _message(tool_calls=[_call("get_time", '{"timezone": "Europe/Paris"}')]),
        _message(content="it is noon"),
    ]
    calls = []

    async def executor(name, arguments):
        calls.append((name, arguments))
        return "12:00"

    async with httpx.AsyncClient(base_url=server) as client:
        reply, rounds = await tools.run_tool_loop(client, [], [], executor)
    assert reply == "it is noon"
    assert rounds == 2
    assert calls == [("get_time", '{"timezone": "Europe/Paris"}')]
    second_request = _Scripted.requests[1]
    tool_message = next(m for m in second_request["messages"] if m.get("role") == "tool")
    assert "12:00" in tool_message["content"]
    assert "untrusted data" in tool_message["content"]


async def test_at_most_max_rounds_requests_are_sent(server):
    _Scripted.responses = [
        _message(tool_calls=[_call("get_time", f'{{"n": {i}}}')])
        for i in range(tools.MAX_TOOL_ROUNDS + 3)
    ]

    async def executor(name, arguments):
        return "ok"

    async with httpx.AsyncClient(base_url=server) as client:
        reply, rounds = await tools.run_tool_loop(client, [], [], executor)
    assert rounds == tools.MAX_TOOL_ROUNDS
    assert reply == tools.TOOL_LIMIT_REACHED_MESSAGE
    assert len(_Scripted.requests) == tools.MAX_TOOL_ROUNDS


async def test_an_identical_call_is_executed_only_once(server):
    # The model asks for the exact same call 3 times, then answers.
    same_call = [_call("get_time", '{"timezone": "Europe/Paris"}')]
    _Scripted.responses = [
        _message(tool_calls=same_call),
        _message(tool_calls=same_call),
        _message(tool_calls=same_call),
        _message(content="done"),
    ]
    counter = {"n": 0}

    async def executor(name, arguments):
        counter["n"] += 1
        return "12:00"

    async with httpx.AsyncClient(base_url=server) as client:
        reply, rounds = await tools.run_tool_loop(client, [], [], executor)
    assert reply == "done"
    assert counter["n"] == 1

    # `messages` accumulates every round, so the *last* tool message of each request
    # is the one that round added. Round 1's call is the real, only execution;
    # rounds 2 and 3 repeat the same call and each still gets a tool message back
    # (skipped, not silently dropped).
    def _last_tool_message(request):
        return [m for m in request["messages"] if m.get("role") == "tool"][-1]

    assert "12:00" in _last_tool_message(_Scripted.requests[1])["content"]
    for request in _Scripted.requests[2:]:
        assert "already made" in _last_tool_message(request)["content"]


async def test_a_hung_tool_is_cut_off_after_the_timeout(server, monkeypatch):
    monkeypatch.setattr(tools, "TOOL_CALL_TIMEOUT_SECONDS", 0.05)
    _Scripted.responses = [
        _message(tool_calls=[_call("slow_tool", "{}")]),
        _message(content="gave up"),
    ]

    async def executor(name, arguments):
        await asyncio.sleep(10)
        return "too late"

    async with httpx.AsyncClient(base_url=server) as client:
        reply, rounds = await tools.run_tool_loop(client, [], [], executor)
    assert reply == "gave up"
    tool_message = next(m for m in _Scripted.requests[1]["messages"] if m.get("role") == "tool")
    assert "did not answer within" in tool_message["content"]


async def test_a_failing_tool_returns_an_error_string_not_a_crash(server):
    _Scripted.responses = [
        _message(tool_calls=[_call("broken_tool", "{}")]),
        _message(content="handled"),
    ]

    async def executor(name, arguments):
        raise ValueError("boom")

    async with httpx.AsyncClient(base_url=server) as client:
        reply, rounds = await tools.run_tool_loop(client, [], [], executor)
    assert reply == "handled"
    tool_message = next(m for m in _Scripted.requests[1]["messages"] if m.get("role") == "tool")
    assert "error: boom" in tool_message["content"]


async def test_a_large_result_is_summarised_by_a_call_of_its_own(server):
    """A result above TOOL_RESULT_MAX_CHARS goes to a separate call (no history, the
    question of the turn, thinking off); the model gets the summary, marked untrusted."""
    page = "Read https://example.org/a-b/c?d=1 now. " + "x" * (tools.TOOL_RESULT_MAX_CHARS * 3)
    _Scripted.responses = [
        _message(tool_calls=[_call("fetch_page", "{}")]),
        _message(content="The page points to https://example.org/a-b/c?d=1"),
        _message(content="ok"),
    ]

    async def executor(name, arguments):
        return page

    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "a"},
               {"role": "user", "content": "what does the page say?"}]  # fmt: skip
    async with httpx.AsyncClient(base_url=server) as client:
        assert (await tools.run_tool_loop(client, history, [], executor))[0] == "ok"
    delegated = _Scripted.requests[1]
    assert [m["role"] for m in delegated["messages"]] == ["system", "user"], "no history"
    assert delegated["messages"][1]["content"].startswith("Question: what does the page say?")
    assert page in delegated["messages"][1]["content"], "the whole result, not the cut one"
    assert delegated["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tools" not in delegated and delegated["temperature"] == 0
    tool_message = next(m for m in _Scripted.requests[2]["messages"] if m.get("role") == "tool")
    assert tool_message["content"].startswith("[tool result, untrusted data")
    assert "https://example.org/a-b/c?d=1" in tool_message["content"]
    assert "x" * 100 not in tool_message["content"]


async def test_a_failed_summary_falls_back_to_the_cut(server):
    _Scripted.responses = [
        _message(tool_calls=[_call("big_tool", "{}")]),
        500,
        _message(content="ok"),
    ]

    async def executor(name, arguments):
        return "x" * (tools.TOOL_RESULT_MAX_CHARS * 3)

    async with httpx.AsyncClient(base_url=server) as client:
        await tools.run_tool_loop(client, [], [], executor)
    tool_message = next(m for m in _Scripted.requests[2]["messages"] if m.get("role") == "tool")
    assert tool_message["content"].count("x") == tools.TOOL_RESULT_MAX_CHARS


async def test_a_small_result_is_not_summarised(server):
    _Scripted.responses = [_message(tool_calls=[_call("t", "{}")]), _message(content="ok")]

    async def executor(name, arguments):
        return "y" * tools.TOOL_RESULT_MAX_CHARS

    async with httpx.AsyncClient(base_url=server) as client:
        await tools.run_tool_loop(client, [], [], executor)
    assert len(_Scripted.requests) == 2


async def test_cancelling_the_turn_stops_a_hung_tool_promptly(server):
    _Scripted.responses = [_message(tool_calls=[_call("slow_tool", "{}")])]
    started = asyncio.Event()

    async def executor(name, arguments):
        started.set()
        await asyncio.sleep(100)  # far longer than TOOL_CALL_TIMEOUT_SECONDS
        return "too late"

    async with httpx.AsyncClient(base_url=server) as client:
        task = asyncio.ensure_future(tools.run_tool_loop(client, [], [], executor))
        await started.wait()
        began = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        elapsed = time.monotonic() - began
    assert elapsed < 1.0, f"cancellation took {elapsed:.3f}s, should be near-instant"


# --- a reply that thought until the token cap is asked for once more without thinking ---


def _thought_to_the_cap():
    message = {"role": "assistant", "content": "", "reasoning_content": "Let me plan... " * 100}
    return {"choices": [{"message": message, "finish_reason": "length"}]}


async def test_the_loop_asks_again_without_thinking_when_the_cap_was_reached(server, caplog):
    _Scripted.responses = [_thought_to_the_cap(), _message(content="the essay")]

    async def executor(name, arguments):
        return "unused"

    async with httpx.AsyncClient(base_url=server) as client:
        with caplog.at_level(logging.WARNING, logger="channelagent"):
            reply, rounds = await tools.run_tool_loop(client, [], [], executor)
    assert (reply, rounds) == ("the essay", 1)
    first, second = _Scripted.requests
    assert "chat_template_kwargs" not in first
    assert second["chat_template_kwargs"] == {"enable_thinking": False}
    assert second["messages"] == first["messages"] and second["tools"] == first["tools"]
    assert "thought until the token cap" in caplog.text


@pytest.mark.parametrize(
    "first_reply, extra",
    [
        (_thought_to_the_cap(), {"chat_template_kwargs": {"enable_thinking": False}}),
        ({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}, {}),
        ({"choices": [{"message": {"content": "cut sh"}, "finish_reason": "length"}]}, {}),
    ],
    ids=["thinking already off", "empty but not at the cap", "visible text at the cap"],
)
async def test_the_loop_does_not_ask_again_otherwise(server, first_reply, extra):
    _Scripted.responses = [first_reply]

    async def executor(name, arguments):
        return "unused"

    async with httpx.AsyncClient(base_url=server) as client:
        await tools.run_tool_loop(client, [], [], executor, **extra)
    assert len(_Scripted.requests) == 1


async def test_a_turn_without_tools_also_asks_again_without_thinking(server):
    from app import graph

    _Scripted.responses = [_thought_to_the_cap(), _message(content="the essay")]
    async with httpx.AsyncClient(base_url=server) as client:
        assert await graph._chat(client, [{"role": "user", "content": "essay"}]) == "the essay"
    first, second = _Scripted.requests
    assert "chat_template_kwargs" not in first
    assert second["chat_template_kwargs"] == {"enable_thinking": False}
    assert second["messages"] == first["messages"]
    _Scripted.responses = [_thought_to_the_cap()]
    async with httpx.AsyncClient(base_url=server) as client:
        off = {"chat_template_kwargs": {"enable_thinking": False}}
        assert await graph._chat(client, [{"role": "user", "content": "essay"}], **off) == ""
    assert len(_Scripted.requests) == 3, "thinking already off: no second request"
