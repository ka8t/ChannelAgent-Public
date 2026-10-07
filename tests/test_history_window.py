"""Tests: the history sent to the model stays under a token budget.

The window is counted with the same estimator the code uses; the mock LLM
records every request so "under the budget" and "latest message included"
are numbers taken from what was actually sent.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from app import graph
from app.db.models import Channel
from app.graph import HISTORY_CONTEXT_SHARE, history_token_budget, window_messages

CTX_SIZE = 400  # budget 300 estimated tokens


def _history(turns: int, filler: int = 200) -> list:
    messages = []
    for i in range(turns):
        messages.append(HumanMessage(content=f"question {i} " + "x" * filler))
        messages.append(AIMessage(content=f"answer {i} " + "y" * filler))
    return messages


def test_budget_is_a_share_of_the_context_window():
    assert HISTORY_CONTEXT_SHARE == 0.75
    assert history_token_budget(65536) == 49152
    assert history_token_budget(400) == 300


def test_a_thread_far_above_the_budget_is_cut_under_it():
    messages = _history(1000) + [HumanMessage(content="LAST QUESTION")]
    kept = window_messages(messages, 300)
    assert len(messages) == 2001
    assert count_tokens_approximately(messages) > 100000
    assert 0 < len(kept) < 20
    assert count_tokens_approximately(kept) <= 300


def test_the_latest_message_is_always_kept_and_last():
    messages = _history(1000) + [HumanMessage(content="LAST QUESTION")]
    kept = window_messages(messages, 300)
    assert kept[-1].content == "LAST QUESTION"


def test_the_window_starts_on_a_human_message():
    kept = window_messages(_history(50) + [HumanMessage(content="now")], 300)
    assert isinstance(kept[0], HumanMessage)


def test_the_most_recent_turns_are_the_ones_kept():
    kept = window_messages(_history(100) + [HumanMessage(content="now")], 300)
    assert "question 99" in kept[-3].content and "answer 99" in kept[-2].content
    assert not any("question 0 " in m.content for m in kept)


def test_a_short_thread_is_sent_whole():
    messages = _history(2) + [HumanMessage(content="now")]
    assert window_messages(messages, 300) == messages


def test_one_message_alone_above_the_budget_is_still_sent():
    huge = HumanMessage(content="z" * 100000)
    assert window_messages([huge], 100) == [huge]
    older = _history(3)
    assert window_messages(older + [huge], 100) == [huge]


class _Recorder(BaseHTTPRequestHandler):
    requests: list = []
    summaries: list = []  # the texts of the next summaries, when a test sets them

    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":  # no tokenizer on this mock
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body["messages"])
        content = "r" * 150
        if type(self).summaries and body["messages"][0]["content"] == graph.SUMMARY_PROMPT:
            content = type(self).summaries.pop(0)
        data = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(autouse=True)
async def _database(fresh_db):
    """A turn reads its agent's settings from the database: a throwaway one."""
    from app.db.session import init_db

    await init_db()


@pytest.fixture
def llm(monkeypatch):
    _Recorder.requests = []
    _Recorder.summaries = []
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("LLAMA_CTX_SIZE", str(CTX_SIZE))
    from app.config import get_settings

    get_settings.cache_clear()
    yield _Recorder
    server.shutdown()
    get_settings.cache_clear()


async def test_requests_stay_under_the_budget_over_a_long_conversation(llm):
    from app.graph import SUMMARY_PROMPT

    turns = 60
    for i in range(turns):
        await graph.run_turn(Channel.TELEGRAM, "42", 1, f"message {i} " + "q" * 200)

    budget = history_token_budget(CTX_SIZE)
    # Summary calls are separate requests; the turn requests are the others.
    chat = [r for r in llm.requests if r[0]["content"] != SUMMARY_PROMPT]
    assert len(chat) == turns
    sizes = [
        count_tokens_approximately(
            [
                (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
                for m in sent
            ]
        )
        for sent in chat
    ]
    assert max(sizes) <= budget
    for i, sent in enumerate(chat):
        assert sent[-1]["content"].startswith(f"message {i} ")
        body = sent[1:] if sent[0]["role"] == "system" else sent
        assert body[0]["role"] == "user"
    assert len(chat[-1]) < 2 * turns - 1


async def test_the_checkpoint_keeps_the_whole_history(llm):
    turns = 30
    for i in range(turns):
        await graph.run_turn(Channel.TELEGRAM, "42", 1, f"message {i} " + "q" * 200)

    compiled = await graph.get_graph()
    thread = graph.build_thread_id(Channel.TELEGRAM, "42", 1)
    state = await compiled.aget_state({"configurable": {"thread_id": thread}})
    assert len(state.values["messages"]) == 2 * turns
    assert len(llm.requests[-1]) < 2 * turns - 1


# --- a window cut by steps keeps its start, so the engine reuses its cache ---


def test_the_start_is_rounded_up_to_a_step_and_lands_on_a_human_message():
    from app.graph import WINDOW_STEP, stepped_start

    messages = _history(20) + [HumanMessage(content="now")]
    assert WINDOW_STEP == graph.SUMMARY_BATCH == 6
    assert stepped_start(messages, 0) == 0
    assert [stepped_start(messages, m) for m in (1, 5, 6, 7, 12)] == [6, 6, 6, 12, 12]
    odd = [AIMessage(content="orphan")] + messages  # a step boundary on an answer moves on
    assert stepped_start(odd, 1) == 7 and isinstance(odd[7], HumanMessage)
    assert stepped_start(messages, len(messages) + 50) == len(messages) - 1


async def test_between_steps_each_request_extends_the_previous_one(llm, monkeypatch):
    """What makes the engine's cache work: once the window is cut, a turn's request is the
    previous request plus its answer and the new message, except at a step (6 messages, so
    every 3 turns). The budget is several steps wide, as in real use (12,288 tokens)."""
    from app.config import get_settings
    from app.graph import SUMMARY_PROMPT

    monkeypatch.setenv("LLAMA_CTX_SIZE", "2000")
    get_settings.cache_clear()
    turns = 60
    for i in range(turns):
        await graph.run_turn(Channel.TELEGRAM, "42", 1, f"message {i} " + "q" * 200)
    chat = [r for r in llm.requests if r[0]["content"] != SUMMARY_PROMPT]
    bodies = [[m["content"] for m in sent if m["role"] != "system"] for sent in chat]
    extended = changed = 0
    for before, after in zip(bodies, bodies[1:], strict=False):
        if after[: len(before) + 1] == before + ["r" * 150]:
            extended += 1
        else:
            changed += 1
    assert changed <= turns // 3 + 1, (extended, changed)
    assert extended >= turns - turns // 3 - 2
    assert len(chat[-1]) < 2 * turns - 1, "the window was cut"


async def test_the_turns_memory_entries_go_with_the_new_message_not_the_system_one(llm):
    from app import memory
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as session:
        user = await service.create_user(session, "Sam")
        agent = await service.create_agent(session, user.id, "mem")
        agent.memory_mode = "always"
        await session.commit()
        user_id, agent_id = user.id, agent.id
    run = memory.make_executor(user_id, agent_id)
    await run("memory_add", json.dumps({"title": "Pet", "content": "a dog named Biscuit"}))
    await graph.run_turn(Channel.TELEGRAM, "42", agent_id, "first question")
    await graph.run_turn(Channel.TELEGRAM, "42", agent_id, "second question")
    first, second = llm.requests[-2], llm.requests[-1]
    system = [m["content"] for m in second if m["role"] == "system"][0]
    assert memory.GUIDANCE in system and "#1 Pet" not in system
    assert second[-1]["content"].endswith("second question")
    assert "#1 Pet" in second[-1]["content"] and "#1 Pet" in first[-1]["content"]
    assert [m["content"] for m in second if m["role"] == "user"][0] == "first question", (
        "the stored question carries no memory block"
    )
    assert first[0] == second[0], "the system message is the same from one turn to the next"


async def test_the_turns_memory_entries_count_in_the_budget(llm, monkeypatch):
    """The same 30 turns with and without a memory index of about 600 tokens (more than a
    step): with it, fewer history messages fit the budget."""
    from app import memory
    from app.admin import service
    from app.config import get_settings
    from app.db.session import session_scope

    monkeypatch.setenv("LLAMA_CTX_SIZE", "2000")
    get_settings.cache_clear()
    async with session_scope() as session:
        user = await service.create_user(session, "Sam")
        plain = await service.create_agent(session, user.id, "plain")
        mem = await service.create_agent(session, user.id, "mem")
        mem.memory_mode = "always"
        await session.commit()
        user_id, ids = user.id, {"plain": plain.id, "mem": mem.id}
    run = memory.make_executor(user_id, ids["mem"])
    for i in range(20):
        await run("memory_add", json.dumps({"title": f"entry {i} " + "t" * 110, "content": "c"}))
    from app.graph import SUMMARY_PROMPT

    sizes: dict[str, list[int]] = {}
    for name, agent_id in ids.items():
        sizes[name] = []
        for i in range(30):
            await graph.run_turn(Channel.TELEGRAM, name, agent_id, f"message {i} " + "q" * 200)
            turn = [r for r in llm.requests if r[0]["content"] != SUMMARY_PROMPT][-1]
            sizes[name].append(len([m for m in turn if m["role"] != "system"]))
        if name == "mem":
            assert "entry 19" in turn[-1]["content"]
    cut = [i for i, n in enumerate(sizes["plain"]) if n < 2 * i + 1]
    assert cut, "the plain agent's window was cut"
    assert all(sizes["mem"][i] < sizes["plain"][i] for i in cut), sizes


async def test_the_window_never_goes_back_over_what_the_summary_covers(llm, monkeypatch):
    """A new summary shorter than the old one left room, and the window moved back over
    summarised messages: sent twice, and a start the engine had not cached (2026-10-04)."""
    from app.config import get_settings
    from app.graph import SUMMARY_PROMPT

    monkeypatch.setenv("LLAMA_CTX_SIZE", "2000")
    get_settings.cache_clear()
    llm.summaries = ["s" * 2400] + ["s" * 40] * 50  # a long summary, then short ones
    compiled = await graph.get_graph()
    thread = {"configurable": {"thread_id": graph.build_thread_id(Channel.TELEGRAM, "42", 1)}}
    starts = []
    for i in range(60):
        await graph.run_turn(Channel.TELEGRAM, "42", 1, f"message {i} " + "q" * 200)
        state = (await compiled.aget_state(thread)).values
        sent = [r for r in llm.requests if r[0]["content"] != SUMMARY_PROMPT][-1]
        first_user = next(m["content"] for m in sent if m["role"] == "user")
        index = next(n for n, m in enumerate(state["messages"]) if m.content == first_user)
        assert index >= state.get("summary_covers", 0), (i, index, state.get("summary_covers"))
        starts.append(index)
    assert starts == sorted(starts), "the start never moves back"
