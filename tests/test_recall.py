"""Tests: the recall archive. Lexical search (accents, rare words, ties), the archive
per (user, agent) across the user's channels and never beyond, the tool offered only when part
of the conversation has left the window, and a full turn where a fact stated 200 turns earlier
is found through the tool. Tool results are never stored in the history (the reason gap A7
has nothing to shed): checked after a tool turn.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app import recall
from app import tools as tools_module
from app.db.models import Channel

FACT = "My passport number is K7731-QZ, keep it in mind."


def _archive(*texts):
    return [("this conversation", i, len(texts), HumanMessage(t)) for i, t in enumerate(texts, 1)]


# --- search ---


def test_the_message_holding_the_words_is_found():
    archive = _archive("hello there", FACT, "the weather is fine", "passport photos are ugly")
    found = recall.search(archive, "what is my passport number?")
    assert found[0][3].content == FACT


def test_accents_and_case_are_ignored():
    archive = _archive("Le numéro de CAFÉ est 12", "rien à voir")
    assert recall.search(archive, "cafe numero")[0][3].content.startswith("Le numéro")


def test_a_rare_word_outweighs_a_common_one_and_the_newer_wins_a_tie():
    archive = _archive("meeting on monday", "meeting on tuesday", "the zeppelin meeting")
    assert recall.search(archive, "zeppelin meeting")[0][3].content == "the zeppelin meeting"
    tie = recall.search(archive, "meeting")
    assert [m.content for *_r, m in tie][:2] == ["the zeppelin meeting", "meeting on tuesday"]


@pytest.mark.parametrize("query", ["", "a an", "   ", "!!"])
def test_a_query_without_words_finds_nothing(query):
    assert recall.search(_archive(FACT), query) == []


def test_results_say_where_they_come_from_and_long_messages_are_cut_around_the_hit():
    long = "x " * 600 + "the vault code is 9981 " + "y " * 600
    found = recall.search(_archive("short", long), "vault code")
    text = recall.format_results(found, "vault code")
    assert text.startswith("[this conversation, message 2 of 2, user] ...")
    assert "vault code is 9981" in text and len(text) < recall.SNIPPET_CHARS + 80
    assert recall.format_results([], "x") == "nothing found"


# --- the world ---


class _Scripted(BaseHTTPRequestHandler):
    responses: list = []
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.loads(raw)
        if body.get("tool_choice") == "required":
            call = {"id": "p", "type": "function",
                    "function": {"name": "_capability_probe", "arguments": "{}"}}  # fmt: skip
            payload = {"choices": [{"message": {"role": "assistant", "content": "",
                                                "tool_calls": [call]}}]}  # fmt: skip
        else:
            type(self).requests.append(body)
            payload = type(self).responses.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _reply(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _recall_call(query):
    function = {"name": recall.TOOL_NAME, "arguments": json.dumps({"query": query})}
    return [{"id": "c1", "type": "function", "function": function}]


@pytest.fixture
async def world(fresh_db, monkeypatch):
    """Sam: Telegram 111 and an email identity, agents "main" and "other". Alex: Telegram 222."""
    from app.admin import service
    from app.config import get_settings
    from app.db.session import init_db, session_scope

    _Scripted.requests, _Scripted.responses = [], []
    server = HTTPServer(("127.0.0.1", 0), _Scripted)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("LLAMA_CTX_SIZE", "2048")
    get_settings.cache_clear()
    tools_module.reset_capability_check()
    await init_db()
    async with session_scope() as session:
        sam = await service.create_user(session, "Sam")
        main = await service.create_agent(session, sam.id, "main")
        other = await service.create_agent(session, sam.id, "other")
        await service.add_channel_identity(session, sam.id, Channel.TELEGRAM, "111")
        await service.add_channel_identity(session, sam.id, Channel.EMAIL, "sam@example.org")
        alex = await service.create_user(session, "Alex")
        alex_agent = await service.create_agent(session, alex.id, "main")
        await service.add_channel_identity(session, alex.id, Channel.TELEGRAM, "222")
        await session.commit()
        ids = {"sam": sam.id, "main": main.id, "other": other.id, "alex": alex.id,
               "alex_agent": alex_agent.id}  # fmt: skip
    yield ids
    from app.graph import close_graph

    await close_graph()
    server.shutdown()
    get_settings.cache_clear()
    tools_module.reset_capability_check()


async def _store(channel, external, agent_id, messages):
    """Write a conversation into its checkpoint, the summary marked as covering it all
    (so no summary call is made)."""
    from app.graph import build_thread_id, get_graph

    graph = await get_graph()
    config = {"configurable": {"thread_id": build_thread_id(channel, external, agent_id)}}
    await graph.aupdate_state(
        config, {"messages": messages, "summary": "", "summary_covers": 10**6},
        as_node="call_llm",
    )  # fmt: skip


def _turns(n, fact_at=None):
    messages = []
    for i in range(1, n + 1):
        said = FACT if i == fact_at else f"turn {i}: tell me about item {i} of the inventory"
        messages += [HumanMessage(said), AIMessage(f"item {i} is on shelf {i % 7}")]
    return messages


# --- the archive per (user, agent) ---


async def test_the_archive_reads_the_users_other_channel_with_the_same_agent_only(world):
    await _store(Channel.EMAIL, "sam@example.org", world["main"],
                 [HumanMessage("my locker code is 4471"), AIMessage("noted")])  # fmt: skip
    await _store(Channel.TELEGRAM, "111", world["other"],
                 [HumanMessage("my locker code is 0000 for the other agent")])  # fmt: skip
    await _store(Channel.TELEGRAM, "222", world["alex_agent"],
                 [HumanMessage("my locker code is 5555, says Alex")])  # fmt: skip
    from app.graph import build_thread_id

    current = build_thread_id(Channel.TELEGRAM, "111", world["main"])
    run = recall.make_executor(world["sam"], world["main"], current, [], 0)
    answer = await run(recall.TOOL_NAME, json.dumps({"query": "locker code"}))
    assert "4471" in answer and "conversation email" in answer
    assert "0000" not in answer and "5555" not in answer
    assert await run(recall.TOOL_NAME, "{not json") == "error: invalid arguments: not JSON"
    assert (await run(recall.TOOL_NAME, json.dumps({"query": 3}))).startswith("error: query")


# --- a turn ---


async def test_a_fact_stated_200_turns_earlier_is_found_through_the_tool(world):
    from app.graph import run_turn

    history = _turns(200, fact_at=1)
    await _store(Channel.TELEGRAM, "111", world["main"], history)
    _Scripted.responses = [
        _reply(tool_calls=_recall_call("passport number")),
        _reply("Your passport number is K7731-QZ."),
    ]
    reply = await run_turn(Channel.TELEGRAM, "111", world["main"], "What is my passport number?")
    assert reply == "Your passport number is K7731-QZ."

    first, second = _Scripted.requests
    offered = [t["function"]["name"] for t in first.get("tools", [])]
    assert offered == [recall.TOOL_NAME]
    assert first["messages"][0]["role"] == "system"
    assert recall.GUIDANCE in first["messages"][0]["content"]
    assert FACT not in json.dumps(first["messages"]), "the fact is outside the window"
    result = next(m for m in second["messages"] if m.get("role") == "tool")["content"]
    assert FACT in result and "message 1 of 401, user" in result


async def test_the_tool_is_not_offered_while_the_whole_conversation_fits(world):
    from app.graph import run_turn

    await _store(Channel.TELEGRAM, "111", world["main"], _turns(2))
    _Scripted.responses = [_reply("fine")]
    assert await run_turn(Channel.TELEGRAM, "111", world["main"], "hello") == "fine"
    assert "tools" not in _Scripted.requests[0]
    assert recall.GUIDANCE not in json.dumps(_Scripted.requests[0]["messages"])


async def test_tool_results_are_never_stored_in_the_history(world):
    """Why gap A7 (shedding old tool results) has nothing to shed: a turn keeps only the
    user's message and the final reply; the tool's 20 KB result stays inside the turn."""
    from app.graph import build_thread_id, get_graph, run_turn

    await _store(Channel.TELEGRAM, "111", world["main"], _turns(200, fact_at=1))
    _Scripted.responses = [
        _reply(tool_calls=_recall_call("passport")),
        _reply("done"),
        _reply("next answer"),
    ]
    await run_turn(Channel.TELEGRAM, "111", world["main"], "find my passport")
    graph = await get_graph()
    config = {
        "configurable": {"thread_id": build_thread_id(Channel.TELEGRAM, "111", world["main"])}
    }
    stored = (await graph.aget_state(config)).values["messages"]
    assert len(stored) == 402
    assert {type(m).__name__ for m in stored} == {"HumanMessage", "AIMessage"}
    await run_turn(Channel.TELEGRAM, "111", world["main"], "and now?")
    last = _Scripted.requests[-1]
    assert not any(m.get("role") == "tool" for m in last["messages"])


def test_one_rare_word_beats_two_common_ones():
    """Counting the words would pick "meeting note"; a word most messages hold says little."""
    fillers = [f"meeting note {i}" for i in range(10)]
    archive = _archive("meeting note", "the zeppelin", *fillers)
    assert recall.search(archive, "zeppelin meeting note")[0][3].content == "the zeppelin"
