"""Tests: a thinking block removed from a reply, a degenerate loop failing the turn,
the LLM_THINKING request fields, and the /new and /export commands on the sender's own data.
"""

import pytest

from app import replies
from app.db.models import Channel

SEQ = " ".join(f"word{i}" for i in range(20))


# --- replies ---


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("<think>let me count</think>\n391", "391"),
        ("<THINK>a\nb</THINK>The answer is 391.", "The answer is 391."),
        ("<think>still thinking and never closed", ""),
        ("No thinking here.", "No thinking here."),
        ("Before <think>x</think> after", "Before after"),
    ],
)
def test_a_thinking_block_is_removed(raw, clean):
    assert replies.strip_thinking(raw) == clean


def test_a_20_word_sequence_repeated_5_times_is_a_loop():
    with pytest.raises(replies.DegenerateReplyError, match="20 words 5 times"):
        replies.check("Here is the plan. " + " ".join([SEQ] * 5))
    assert replies.check(" ".join([SEQ] * 4)) == " ".join([SEQ] * 4), "4 copies: not a loop"
    with pytest.raises(replies.DegenerateReplyError, match="1 words"):
        replies.check("ha " * 100)
    ordinary = "Sea otters hold hands. They eat urchins. " * 3
    assert replies.check(ordinary) == ordinary.strip()


def test_the_thinking_setting_becomes_request_fields():
    from app.graph import thinking_fields

    assert thinking_fields("auto") == {} and thinking_fields(None) == {}
    assert thinking_fields("off") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert thinking_fields("on") == {"chat_template_kwargs": {"enable_thinking": True}}


async def test_a_looping_reply_is_a_failed_turn_with_the_apology(fresh_db, monkeypatch):
    from app.admin import service
    from app.channels import dispatch
    from app.channels.schema import NormalizedEvent
    from app.db.models import PermissionKind
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        user = await service.create_user(session, "Sam")
        identity = await service.add_channel_identity(session, user.id, Channel.TELEGRAM, "111")
        await service.grant_identity_permission(session, user.id, identity.id, PermissionKind.CHAT)
        await session.commit()

    async def looping(*a, **k):
        return replies.check(" ".join([SEQ] * 6))

    monkeypatch.setattr(dispatch, "run_turn", looping)
    sent = []

    async def reply(text):
        sent.append(text)

    async with session_scope() as session:
        outcome = await dispatch.dispatch_event(
            session, NormalizedEvent("111", Channel.TELEGRAM, "plan my day", reply)
        )
    assert outcome is dispatch.DispatchOutcome.FAILED
    assert sent == [dispatch.APOLOGY_MESSAGE]


# --- /new and /export ---


@pytest.fixture
async def people(fresh_db):
    from app.admin import service
    from app.db.models import PermissionKind
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        ids = {}
        for name, external in (("sam", "111"), ("alex", "222")):
            user = await service.create_user(session, name)
            identity = await service.add_channel_identity(
                session, user.id, Channel.TELEGRAM, external
            )
            await service.grant_identity_permission(
                session, user.id, identity.id, PermissionKind.CHAT
            )
            agent = await service.get_or_create_default_agent(session, user.id)
            for i in range(3):
                await service.record_action(
                    session, user_id=user.id, agent_id=agent.id, channel=Channel.TELEGRAM,
                    direction=service.Direction.INBOUND, text=f"{name} secret {i}",
                )  # fmt: skip
            ids[name] = (user.id, agent.id)
        # Sam's second agent: its conversation is not the current one, never exported.
        work = await service.create_agent(session, ids["sam"][0], "work")
        await service.record_action(
            session, user_id=ids["sam"][0], agent_id=work.id, channel=Channel.TELEGRAM,
            direction=service.Direction.INBOUND, text="work agent note",
        )  # fmt: skip
        await session.commit()
    return ids


async def _run(handler, external, send_file=None):
    from app.channels.schema import NormalizedEvent
    from app.db.session import session_scope

    sent = []

    async def reply(text):
        sent.append(text)

    event = NormalizedEvent(external, Channel.TELEGRAM, "/x", reply, send_file=send_file)
    async with session_scope() as session:
        await handler(session, event)
    return sent[-1]


async def test_export_sends_the_senders_own_messages_only(people):
    from app.channels.dispatch import handle_export_command

    files = []

    async def send_file(name, data):
        files.append((name, data.decode()))

    answer = await _run(handle_export_command, "111", send_file)
    assert answer == "3 messages with agent 'default' exported in conversation-default.txt."
    ((name, text),) = files
    assert name == "conversation-default.txt", "a .md file did not open in the owner's Telegram"
    assert text.count("sam secret") == 3 and "alex" not in text
    assert "work agent note" not in text, "only the current agent's conversation"
    no_file = await _run(handle_export_command, "222")
    assert no_file.startswith("3 messages") and "cannot send a file" in no_file


async def test_new_starts_a_thread_with_no_prior_message(people):
    from langchain_core.messages import AIMessage, HumanMessage

    from app.channels.dispatch import handle_new_command
    from app.graph import build_thread_id, close_graph, get_graph

    graph = await get_graph()
    thread = {"configurable": {"thread_id": build_thread_id(Channel.TELEGRAM, "111",
                                                             people["sam"][1])}}  # fmt: skip
    other = {"configurable": {"thread_id": build_thread_id(Channel.TELEGRAM, "222",
                                                            people["alex"][1])}}  # fmt: skip
    for config in (thread, other):
        await graph.aupdate_state(config, {"messages": [HumanMessage("hi"), AIMessage("hello")]},
                                  as_node="call_llm")  # fmt: skip
    try:
        answer = await _run(handle_new_command, "111")
        assert answer.startswith("New conversation with agent 'default'")
        state = await graph.aget_state(thread)
        assert len(state.values.get("messages", [])) == 0
        assert len((await graph.aget_state(other)).values["messages"]) == 2, "only the sender's"
        again = await _run(handle_new_command, "111")
        assert again.startswith("This is already a new conversation")
    finally:
        await close_graph()


@pytest.mark.parametrize(
    "message, finish, expected",
    [
        ({"content": "", "reasoning_content": "long plan"}, "length", True),
        ({"content": "<think>still planning"}, "length", True),
        ({"content": "half an ans"}, "length", False),
        ({"content": "", "tool_calls": [{"id": "1"}]}, "length", False),
        ({"content": ""}, "stop", False),
    ],
)
def test_a_reply_that_thought_until_the_cap(message, finish, expected):
    body = {"choices": [{"message": message, "finish_reason": finish}]}
    assert replies.thought_to_the_cap(body) is expected
