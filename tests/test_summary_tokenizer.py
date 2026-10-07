"""Tests (a running summary of the turns that fell out of the window) and
(token counts from the server's own tokenizer, with the estimate as fallback).

A recording mock of llama-server: /v1/chat/completions (summary requests are told
apart by their system prompt) and /tokenize (one token per word, or 404).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app import graph
from app.db.models import Channel
from app.graph import SUMMARY_HEADER, SUMMARY_PROMPT, history_token_budget

CTX_SIZE = 400  # budget 300 tokens
LONG = "w" * 200 + " x"  # 202 characters (about 50 estimated tokens), 2 words


class _Server(BaseHTTPRequestHandler):
    chat: list = []
    tokenize_calls = 0
    tokenizer = True
    summary_fails = False

    def log_message(self, *a):
        pass

    def _send(self, code, payload=None):
        data = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        if self.path == "/tokenize":
            cls.tokenize_calls += 1
            if not cls.tokenizer:
                return self._send(404)
            return self._send(200, {"tokens": list(range(len(body["content"].split())))})
        messages = body["messages"]
        cls.chat.append(messages)
        if messages[0]["content"] == SUMMARY_PROMPT:
            if cls.summary_fails:
                return self._send(500)
            n = sum(1 for m in cls.chat if m[0]["content"] == SUMMARY_PROMPT)
            return self._send(200, {"choices": [{"message": {"content": f"SUMMARY-{n}"}}]})
        return self._send(200, {"choices": [{"message": {"content": "r" * 150}}]})


@pytest.fixture(autouse=True)
async def _database(fresh_db):
    """A turn reads its agent's settings from the database: a throwaway one."""
    from app.db.session import init_db

    await init_db()


@pytest.fixture
def server(monkeypatch):
    _Server.chat, _Server.tokenize_calls = [], 0
    _Server.tokenizer, _Server.summary_fails = False, False
    httpd = HTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setenv("LLAMA_CTX_SIZE", str(CTX_SIZE))
    from app.config import get_settings

    get_settings.cache_clear()
    yield _Server
    httpd.shutdown()
    get_settings.cache_clear()


def _summaries(server):
    return [m for m in server.chat if m[0]["content"] == SUMMARY_PROMPT]


def _turns(server):
    return [m for m in server.chat if m[0]["content"] != SUMMARY_PROMPT]


async def _run(turns: int, text=LONG):
    for i in range(turns):
        await graph.run_turn(Channel.TELEGRAM, "42", 1, f"message {i} " + text)


async def _state():
    compiled = await graph.get_graph()
    thread = graph.build_thread_id(Channel.TELEGRAM, "42", 1)
    return (await compiled.aget_state({"configurable": {"thread_id": thread}})).values


# --- the summary ---


async def test_a_long_conversation_gets_summarized_in_batches_not_every_turn(server):
    await _run(40)
    assert 0 < len(_summaries(server)) < 40 // 2
    assert len(_turns(server)) == 40


async def test_the_latest_summary_is_in_front_of_the_window_in_every_later_request(server):
    await _run(40)
    last = _turns(server)[-1]
    assert last[0]["role"] == "system"
    assert last[0]["content"].startswith(SUMMARY_HEADER + "SUMMARY-")
    assert last[1]["role"] == "user" and last[-1]["content"].startswith("message 39 ")


async def test_requests_with_a_summary_stay_under_the_budget(server):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.messages.utils import count_tokens_approximately

    await _run(40)
    kinds = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    sizes = [
        count_tokens_approximately([kinds[m["role"]](content=m["content"]) for m in sent])
        for sent in _turns(server)
    ]
    assert max(sizes) <= history_token_budget(CTX_SIZE)


async def test_the_summary_is_checkpointed_and_the_whole_history_is_kept(server):
    await _run(40)
    state = await _state()
    assert len(state["messages"]) == 80
    assert state["summary"].startswith("SUMMARY-")
    assert 0 < state["summary_covers"] < 80


async def test_each_summary_request_carries_the_previous_summary_and_only_new_messages(server):
    await _run(40)
    summaries = _summaries(server)
    assert "Earlier summary" not in summaries[0][1]["content"]
    assert "Earlier summary: SUMMARY-1" in summaries[1][1]["content"]
    first_words = [s[1]["content"].count("message ") for s in summaries]
    assert all(n >= graph.SUMMARY_BATCH // 2 for n in first_words), first_words
    seen = " ".join(s[1]["content"] for s in summaries)
    assert seen.count("message 0 ") == 1, "a dropped message is summarized once"


async def test_a_failing_summary_call_does_not_fail_the_turn(server):
    server.summary_fails = True
    await _run(30)
    assert len(_turns(server)) == 30
    assert all(m[0]["role"] != "system" for m in _turns(server))
    assert "summary" not in await _state()


# --- exact token counts ---


async def test_the_servers_tokenizer_counts_when_it_has_one(server):
    server.tokenizer = True
    await _run(20)
    exact = len(_turns(server)[-1])
    assert exact >= 20, f"exact counts (2 words + 4) keep many more messages, got {exact}"


async def test_the_estimate_is_used_when_the_server_has_no_tokenizer(server):
    server.tokenizer = False
    await _run(20)
    estimate = len(_turns(server)[-1])
    assert estimate <= 8, f"estimated counts keep few messages, got {estimate}"
    assert len(_turns(server)) == 20, "no turn failed"


async def test_a_server_without_a_tokenizer_is_asked_once_not_every_turn(server):
    server.tokenizer = False
    await _run(10)
    assert server.tokenize_calls == 1


async def test_each_message_is_tokenized_once(server, monkeypatch):
    monkeypatch.setenv("LLAMA_CTX_SIZE", "65536")  # nothing dropped, no summary
    from app.config import get_settings

    get_settings.cache_clear()
    server.tokenizer = True
    await _run(5, text="short")
    assert server.tokenize_calls == 1 + 2 * 4, "5 turns: 9 distinct messages, none repeated"


async def test_the_window_size_follows_the_exact_count_including_the_template_overhead(server):
    """Each message is 2 words + 4 overhead = 6 tokens, so a budget of 300 holds at
    most 50 messages (summary header excluded from the arithmetic by a wide margin).
    """
    server.tokenizer = True
    await _run(60)
    sent = _turns(server)[-1]
    body = [m for m in sent if m["role"] != "system"]
    assert 40 <= len(body) <= 50, len(body)
