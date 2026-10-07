"""The agent builder dialogue (.

A user writes `/newagent <request>` ("an agent that gives me the latest AI news every Thursday
at 9:00") on any channel. The builder turns the request into an agent specification,
asks one question at a time for what it cannot infer, shows a summary with the next run times,
and creates the agent and its task only on the user's yes. "test" creates the agent with its task
stopped, runs it once now (the result arrives like any task result), then asks again.

The dialogue is a LangGraph graph of its own, persisted by the same encrypted checkpointer as
the conversations, in the thread `builder_{channel}_{identity key}`: it survives a restart and
resumes at the same question. While a dialogue waits for an answer, the identity's messages go to
it instead of the agent (app.channels.dispatch). `/cancel` ends it; one idle for IDLE_SECONDS is
dropped at the next message.

The model only proposes. Its output is constrained by a JSON schema whose tool and skill lists
are exactly what the user may attach (so it cannot name another tool), then normalised and
checked by app.admin.agent_spec, which is authoritative: the builder never widens a right.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from datetime import UTC, datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import select

from app import schedule_phrase, tasks
from app.admin import agent_spec
from app.admin import routing as routing_service
from app.admin.service import NotFoundError
from app.config import engine_headers, get_settings
from app.db.models import Agent, ChannelIdentity, McpGrant, McpServer, ScheduledTask, Skill, User
from app.db.session import session_scope
from app.mcp.catalogue import catalogue_name
from app.mcp.policy import Policy, effective_policy

logger = logging.getLogger("channelagent")

START_COMMAND = "/newagent"
EDIT_COMMAND = "/editagent"
EDIT_USAGE = "Say which agent and what to change, for example: /editagent ai-news move it to 8:00"
EDIT_CANCELLED = "Agent builder cancelled: the agent was not changed."
CANCEL_COMMAND = "/cancel"
MAX_QUESTIONS = 5  # questions asked to the user in one dialogue
MAX_CHANGES = 5  # change requests answered at the summary
REPAIRS = 2  # extra model calls fed with the check's errors before a question is asked
IDLE_SECONDS = 24 * 3600
MODEL_TIMEOUT_SECONDS = 300
MAX_REQUEST = 2000
CHOICES = ("yes", "test", "no")
YES_WORDS = frozenset({"yes", "y", "oui", "o", "ok", "create", "go"})
NO_WORDS = frozenset({"no", "n", "non", "cancel", "annuler", "stop"})
TEST_WORDS = frozenset({"test", "tester", "try", "essai", "essayer"})

USAGE = (
    "Describe the agent after the command, for example: /newagent an agent that gives me the "
    "latest AI news every Thursday at 9:00"
)
CANCELLED = "Agent builder cancelled: nothing was created."
NOTHING_TO_CANCEL = "No agent is being built."
ASK_AGAIN = "Answer yes to create the agent, no to cancel it, or tell me what to change."
TIMEZONE_QUESTION = (
    "In which timezone should its schedule run? Give a city or a zone name "
    "(for example Paris or Europe/Paris)."
)


def thread_id(channel, identity_key: str) -> str:
    return f"builder_{channel.value}_{identity_key}"


class BuilderState(TypedDict, total=False):
    user_id: int
    identity_id: int
    notes: list  # the dialogue so far: {"role": "user" | "builder", "text": ...}
    spec: dict
    errors: list
    question: str | None
    questions: int
    changes: int
    draft_agent_id: int | None
    draft_task_id: int | None
    needs_timezone: bool
    timezone_missed: str | None
    edit_agent_id: int | None  # /editagent changes this agent instead of creating one
    original: dict  # Its specification before the change
    nonce: str
    say: str
    updated_at: float
    template: dict | None  # The template followed (its fields), {} for none


# --- what the user may attach ---


async def _choices(session, user_id: int) -> tuple[list[dict], list[dict]]:
    """The MCP tools and the skills this user may attach to an agent they create: tools of an
    enabled server, covered by a grant for all their agents, approved, on, not denied; skills
    marked self-service. Read from the database only: no server is contacted."""
    grants = (
        await session.execute(
            select(McpGrant.server_name, McpGrant.tool_name).where(
                McpGrant.user_id == user_id, McpGrant.agent_id.is_(None)
            )
        )
    ).all()
    by_server: dict[str, set | None] = {}
    for server, tool in grants:
        if tool is None:
            by_server[server] = None
        elif by_server.get(server, set()) is not None:
            by_server.setdefault(server, set()).add(tool)
    tools = []
    if by_server:
        servers = (
            await session.execute(select(McpServer).where(McpServer.name.in_(list(by_server))))
        ).scalars()
        for server in servers:
            if not server.enabled or (server.env_vars and not server.shared_credentials):
                continue
            allowed = by_server[server.name]
            for name, entry in sorted((server.approved_definitions or {}).items()):
                definition = entry.get("definition") or {}
                if allowed is not None and name not in allowed:
                    continue
                if name in (server.disabled_tools or []):
                    continue
                policy = effective_policy(name, definition, server.tool_policies or {})
                if policy == Policy.DENY:
                    continue
                description = " ".join(str(definition.get("description") or "").split())[:200]
                full_name = catalogue_name(server.name, name)
                tools.append({"name": full_name, "description": description, "policy": policy})
    skills = [
        {"name": name, "description": description}
        for name, description in (
            await session.execute(
                select(Skill.name, Skill.description)
                .where(Skill.self_service.is_(True))
                .order_by(Skill.name)
            )
        ).all()
    ]
    return tools, skills


# --- templates ---

NO_TEMPLATE = "none"
PICK_PROMPT = """Pick the template that matches the user's request, or "none" when no template
fits well. Answer with JSON: {"template": "<name or none>"}."""


async def pick_template(request: str, templates: list[dict]) -> dict:
    """The template the request matches ({} for none), by one constrained call: the answer is
    an enumeration of the template names and "none", so the model cannot name another."""
    if not templates:
        return {}
    names = [t["name"] for t in templates]
    listing = "\n".join(f"- {t['name']}: {t['description']}" for t in templates)
    schema = {
        "type": "object",
        "properties": {"template": {"type": "string", "enum": [*names, NO_TEMPLATE]}},
        "required": ["template"],
        "additionalProperties": False,
    }
    messages = [
        {"role": "system", "content": PICK_PROMPT},
        {"role": "user", "content": f"Templates:\n{listing}\n\nRequest: {request}"},
    ]
    choice = (await ask_model(messages, schema)).get("template")
    return next((t for t in templates if t["name"] == choice), {})


def apply_template(spec: dict, template: dict, tools: list[dict]) -> dict:
    """What a template fixes, applied after the model wrote the specification: its memory mode,
    its tools among those the user may attach (a template never widens a right), and its
    instructions at the end of the agent's own."""
    if not template:
        return spec
    if template.get("memory_mode"):
        spec["memory_mode"] = template["memory_mode"]
    wanted = set(template.get("tools") or [])
    if wanted:
        offered = {t["name"] for t in tools if t["name"].rsplit("__", 1)[-1] in wanted}
        spec["tools"] = sorted(set(spec.get("tools") or []) | offered)
    extra = (template.get("agent_instructions") or "").strip()
    prompt = (spec.get("system_prompt") or "").strip()
    if extra and extra not in prompt:
        spec["system_prompt"] = f"{prompt}\n\n{extra}" if prompt else extra
    return spec


def template_question(spec: dict, template: dict) -> str | None:
    """The template's question for the first missing detail, or None: while the builder follows
    a template it asks only the template's questions (the model's own are not asked)."""
    questions = template.get("questions") or {}
    scheduled = bool(spec.get("schedule_kind") and spec.get("schedule_expr"))
    missing = {
        "purpose": not spec.get("purpose"),
        "schedule": bool(template.get("needs_schedule")) and not scheduled,
        "task_prompt": scheduled and not spec.get("task_prompt"),
    }
    for field in ("purpose", "schedule", "task_prompt"):
        if missing[field] and questions.get(field):
            return questions[field]
    return None


# --- the model's part ---


def output_schema(tools: list[dict], skills: list[dict]) -> dict:
    """The JSON the model must write. The tool and skill lists are enumerations of what the
    user may attach, so a grammar-constrained engine cannot name anything else."""

    def names(items):
        values = [i["name"] for i in items]
        if values:
            return {"type": "array", "items": {"type": "string", "enum": values}}
        return {"type": "array", "maxItems": 0}

    text_or_null = {"type": ["string", "null"]}
    fields = {
        "name": {"type": "string"},
        "purpose": {"type": "string"},
        "system_prompt": {"type": "string"},
        "memory_mode": {"type": "string", "enum": ["off", "ondemand", "always", "search"]},
        "tools": names(tools),
        "skills": names(skills),
        "schedule_kind": {"type": ["string", "null"], "enum": ["cron", "every", "daily", None]},
        "schedule_expr": text_or_null,
        "task_prompt": text_or_null,
        "question": text_or_null,
    }
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


SYSTEM_PROMPT = """You configure an autonomous agent for the user from their request.
Write the agent specification as JSON:
- name: short, lowercase letters, digits and dashes (for example ai-news).
- purpose: one sentence saying what the agent is for, in the user's language.
- system_prompt: the agent's instructions, written for the agent: its role, what to do, the
  format of its answers (length, links, language), from what the user said.
- memory_mode: "off" unless the agent must remember things between runs ("ondemand").
- tools: only from the list below, only those the agent needs; [] when none fits.
- skills: only from the list below; [] when none fits.
- A recurring agent needs a schedule, read in the user's timezone:
  schedule_kind "daily" with schedule_expr "HH:MM" (every day at that time),
  "every" with "30m", "2h" or "1d" (an interval),
  or "cron" with five fields "minute hour day-of-month month day-of-week": the day of the
  week is the FIFTH field (0 or 7 = Sunday, 1 = Monday ... 4 = Thursday ... 6 = Saturday), the
  third is a day of the month. Examples: every Thursday at 9:00 is cron "0 9 * * 4"; working
  days at 18:00 is cron "0 18 * * 1-5".
  task_prompt: the message the agent receives at each run (what to produce this time).
  An agent the user only chats with has schedule_kind, schedule_expr and task_prompt null.
- News or updates from the web: attach a read_feed tool when you may, and write in task_prompt
  which feed addresses to read (from the known feeds below that match the topic, or the ones
  the user gives) and what to keep.
- question: null when you can build a sensible agent. Otherwise ONE short question, in the
  user's language, about the single most important missing detail (for example the topic or
  the format). Prefer sensible defaults to questions. The answers go to the channel the user
  writes from, so never ask where to send them.
Keep what the specification already holds unless the user changes it. When the previous
attempt was refused, fix every reason given."""


def build_messages(state: BuilderState, context: dict) -> list[dict]:
    lines = [
        f"Now: {context['now']} ({context['timezone']}).",
        "Tools you may use: "
        + (
            "; ".join(f"{t['name']}: {t['description']}" for t in context["tools"])
            or "none (tools must be [])"
        ),
        "Skills you may use: "
        + (
            "; ".join(f"{s['name']}: {s['description']}" for s in context["skills"])
            or "none (skills must be [])"
        ),
    ]
    if context.get("feeds"):
        lines.append(
            "Known feeds: "
            + "; ".join(f"{f['name']} [{f['topics']}]: {f['url']}" for f in context["feeds"])
        )
    template = context.get("template") or {}
    if template:
        fixed = []
        if template.get("memory_mode"):
            fixed.append(f"memory_mode is {template['memory_mode']}")
        if template.get("tools"):
            fixed.append("the tools ending in " + ", ".join(template["tools"]) + " are attached")
        if template.get("needs_schedule"):
            fixed.append("the agent runs on a schedule")
        lines.append(
            f"Template {template['name']}: {template['description']}"
            + (f" {template['guidance']}" if template.get("guidance") else "")
            + (f" Fixed: {'; '.join(fixed)}." if fixed else "")
        )
    dialogue = "\n".join(
        f"{'User' if n['role'] == 'user' else 'You asked'}: {n['text']}"
        for n in state.get("notes", [])
    )
    parts = ["\n".join(lines), "Dialogue so far:\n" + dialogue]
    if state.get("spec"):
        current = {k: v for k, v in state["spec"].items() if k != "delivery_identity_id"}
        parts.append("Specification so far: " + json.dumps(current, ensure_ascii=False))
    if state.get("errors"):
        parts.append("The previous attempt was refused: " + "; ".join(state["errors"]))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


async def ask_model(messages: list[dict], schema: dict) -> dict:
    """One constrained call to the engine. Thinking off: a specification needs no long
    reasoning, and on the installed model it made a simple answer 6 times slower."""
    settings = get_settings()
    async with session_scope() as session:
        default_model = (await routing_service.get_routing(session))["default_model"]
    body = {
        "messages": messages,
        "stream": False,
        # Reproducible: the same request gives the same specification. At 0.2, one case of 22
        # lost its schedule in a benchmark run and gave it in 3 of 3 reruns.
        "temperature": 0,
        "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "spec", "schema": schema},
        },
        **({"model": default_model} if default_model else {}),
    }
    async with httpx.AsyncClient(
        base_url=settings.llama_server_url,
        headers=engine_headers(settings),
        timeout=MODEL_TIMEOUT_SECONDS,
    ) as client:
        response = await client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
    content = response.json()["choices"][0]["message"].get("content") or ""
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("the model's specification is not an object")
    return data


_NAME_CHARS = re.compile(r"[^a-z0-9-]+")


def normalise(data: dict) -> tuple[dict, str | None]:
    """The model's output as a specification, and its question. Deterministic repairs of
    mistakes measured on the installed model: a cron line given as "daily" (2026-09-28, 3 of 3
    probes), a time given as "cron"; a name with spaces or capitals."""
    spec = {k: data.get(k) for k in agent_spec.SPEC_FIELDS if k in data}
    kind, expr = schedule_phrase.repair(spec.get("schedule_kind"), spec.get("schedule_expr"))
    spec["schedule_kind"], spec["schedule_expr"] = kind, expr
    name = str(spec.get("name") or "").strip().lower().replace(" ", "-")
    spec["name"] = _NAME_CHARS.sub("", name).strip("-")[:100]
    for field in ("purpose", "system_prompt", "task_prompt", "model"):
        if isinstance(spec.get(field), str):
            value = spec[field].strip()
            # The owner's model wrote the word "null" for an empty field (measured).
            spec[field] = None if value.lower() in ("", "null", "none") else value
    if not kind or not expr:
        # No schedule: a task prompt alone means nothing.
        spec["schedule_kind"] = spec["schedule_expr"] = spec["task_prompt"] = None
    # A schedule without a task prompt is kept: the check refuses it with that reason and the
    # repair pass asks the model for the prompt. Dropping the schedule lost a correct one in
    # the benchmark (case 22, "every hour from 9 to 17 on weekdays").
    question = data.get("question")
    question = question.strip() if isinstance(question, str) and question.strip() else None
    return spec, question


# --- what the user reads ---


next_runs = schedule_phrase.next_runs


def summary(spec: dict, timezone: str | None, channel: str | None, now: datetime) -> str:
    def cut(text, size=400):
        text = text or "-"
        return text if len(text) <= size else text[: size - 3] + "..."

    lines = [
        "Here is the agent I will create:",
        f"Name: {spec['name']}",
        f"Purpose: {cut(spec.get('purpose'))}",
        f"Instructions: {cut(spec.get('system_prompt'))}",
        f"Tools: {', '.join(spec.get('tools') or []) or 'none'}",
        f"Skills: {', '.join(spec.get('skills') or []) or 'none'}",
        f"Memory: {spec.get('memory_mode') or 'off'}",
    ]
    if spec.get("schedule_kind"):
        runs = next_runs(spec["schedule_kind"], spec["schedule_expr"], timezone, now)
        when = schedule_phrase.describe_runs(runs)
        lines += [
            f"Schedule: {spec['schedule_kind']} {spec['schedule_expr']} ({timezone or 'UTC'}). "
            f"Next runs: {when}",
            f"Each run asks: {cut(spec['task_prompt'])}",
            f"Delivered on: {channel or '-'}",
        ]
        if spec.get("unattended"):
            lines.append(
                "Used without asking in its scheduled runs (nobody can confirm then): "
                + ", ".join(spec["unattended"])
            )
    else:
        lines.append("Schedule: none (you chat with it: /agent " + spec["name"] + ")")
    lines.append(
        "Answer yes to create it, test to run it once now first, no to cancel, "
        "or tell me what to change."
    )
    return "\n".join(lines)


def classify(text: str) -> str | None:
    word = text.strip().strip(".!?").lower()
    if word in YES_WORDS:
        return "yes"
    if word in NO_WORDS:
        return "no"
    if word in TEST_WORDS:
        return "test"
    return None


# --- the graph ---


async def _user_and_identity(user_id: int, identity_id: int):
    async with session_scope() as session:
        user = await session.get(User, user_id)
        identity = await session.get(ChannelIdentity, identity_id)
        return (user.timezone if user else None), (identity.channel if identity else None)


async def extract(state: BuilderState) -> dict:
    """Ask the model for the specification, repair it, check it; repeat with the check's
    errors up to REPAIRS times before asking the user."""
    user_id = state["user_id"]
    async with session_scope() as session:
        user = await session.get(User, user_id)
        identity = await session.get(ChannelIdentity, state["identity_id"])
        tools, skills = await _choices(session, user_id)
        editing = None
        if state.get("edit_agent_id"):
            # What the agent already has stays choosable (an administrator may have set
            # it); only additions are limited to what the user may attach.
            editing = await agent_spec.own_agent(session, user_id, state["edit_agent_id"])
            names = {t["name"] for t in tools}
            tools += [{"name": n, "description": "already on this agent", "policy": None}
                      for n in editing.tools or [] if n not in names]  # fmt: skip
            skill_names = {s["name"] for s in skills}
            skills += [{"name": n, "description": "already on this agent"}
                       for n in editing.skills or [] if n not in skill_names]  # fmt: skip
        feeds = []
        if any(t["name"].endswith("__read_feed") for t in tools):
            from app.admin.feed_sources import list_sources

            feeds = [{"name": s.name, "url": s.url, "topics": s.topics}
                     for s in await list_sources(session)]  # fmt: skip
        template = state.get("template")
        if template is None and not state.get("edit_agent_id"):
            from app.admin import agent_templates

            enabled = await agent_templates.list_templates(session, enabled_only=True)
            templates = [agent_templates.snapshot(t) for t in enabled]
        else:
            templates = []
    if template is None:
        first = next((n["text"] for n in state.get("notes", []) if n["role"] == "user"), "")
        template = await pick_template(first, templates) if templates else {}
    timezone = user.timezone if user else None
    now = datetime.now(UTC)
    context = {
        "template": template,
        "now": now.astimezone(ZoneInfo(timezone or "UTC")).strftime("%A %Y-%m-%d %H:%M"),
        "timezone": timezone or "UTC",
        "tools": tools,
        "skills": skills,
        "feeds": feeds,  # Offered only to a user who may attach a feeds tool
    }
    schema = output_schema(tools, skills)
    working: BuilderState = dict(state)
    spec, question, errors = working.get("spec") or {}, None, []
    for _attempt in range(1 + REPAIRS):
        data = await ask_model(build_messages(working, context), schema)
        spec, question = normalise(data)
        if template:
            spec = apply_template(spec, template, tools)
            question = template_question(spec, template)
        if identity is not None and identity.channel in tasks.DELIVERY_CHANNELS:
            spec["delivery_identity_id"] = identity.id if spec.get("schedule_kind") else None
        kept_identity = (state.get("original") or {}).get("delivery_identity_id")
        if kept_identity and spec.get("schedule_kind"):
            spec["delivery_identity_id"] = kept_identity  # A change keeps its channel
        # The tools that would ask for a confirmation, which nobody can give in a
        # scheduled run; the summary names them and the user's yes approves them for it.
        confirm = {t["name"] for t in tools if t.get("policy") == Policy.CONFIRM}
        wanted = set(spec.get("tools") or [])
        if editing is not None:
            # A change keeps the approval the user gave, for the tools still there, and
            # proposes only the tools it adds (measured: moving a task to 8:00 had proposed the
            # agent's existing clock tool, which the user never asked for).
            original = state.get("original") or {}
            kept = set(original.get("unattended") or []) & wanted
            added = (wanted - set(original.get("tools") or [])) & confirm
            approved = kept | added
        else:
            approved = wanted & confirm
        spec["unattended"] = sorted(approved) if spec.get("schedule_kind") else []
        async with session_scope() as session:
            _cleaned, errors = await agent_spec.check_spec(
                session, user_id, spec, now, editing=editing
            )
        if question or not errors:
            break
        working["spec"], working["errors"] = spec, errors
    return {
        "spec": spec,
        "errors": errors,
        "question": question,
        "template": template,
        # A schedule is read in the user's timezone: without one it would run in UTC.
        "needs_timezone": bool(spec.get("schedule_kind")) and not timezone,
        "updated_at": time.time(),
    }


def after_extract(state: BuilderState) -> str:
    if not state.get("question") and not state.get("errors"):
        return "timezone" if state.get("needs_timezone") else "confirm"
    if state.get("questions", 0) >= MAX_QUESTIONS:
        return "give_up"
    return "ask"


async def ask(state: BuilderState) -> dict:
    question = state.get("question")
    if not question:
        question = "I cannot create it yet: " + "; ".join(state.get("errors") or [])
        question += ". What should I change?"
    answer = interrupt({"say": question, "choices": []})
    notes = [
        *state.get("notes", []),
        {"role": "builder", "text": question},
        {"role": "user", "text": answer},
    ]
    return {"notes": notes, "questions": state.get("questions", 0) + 1, "updated_at": time.time()}


async def ask_timezone(state: BuilderState) -> Command:
    """The user has no timezone and the agent has a schedule: ask for one, set it on the user
    (their other tasks move with it), then show the summary."""
    missed = state.get("timezone_missed")
    question = (f"I do not know the timezone {missed!r}. " if missed else "") + TIMEZONE_QUESTION
    answer = interrupt({"say": question, "choices": []})
    questions = state.get("questions", 0) + 1
    zone = tasks.guess_timezone(answer)
    if zone is None:
        if questions >= MAX_QUESTIONS:
            update = {"questions": questions, "errors": ["no timezone given"]}
            return Command(goto="give_up", update=update)
        update = {"questions": questions, "timezone_missed": answer.strip()[:64]}
        return Command(goto="timezone", update=update)
    from app.admin.service import update_user

    async with session_scope() as session:
        await update_user(
            session, state["user_id"], timezone=zone, actor=f"user:{state['user_id']}"
        )
        await session.commit()
    return Command(
        goto="confirm",
        update={"questions": questions, "needs_timezone": False, "timezone_missed": None},
    )


async def give_up(state: BuilderState) -> dict:
    reasons = "; ".join(state.get("errors") or []) or "the details are still missing"
    return {"say": f"I could not complete the agent after {MAX_QUESTIONS} questions ({reasons}). "
            "Nothing was created; /newagent starts again."}  # fmt: skip


async def _delivery_channel(user_id: int, spec: dict, asking) -> str | None:
    """Where the task's results go: its delivery identity, else the user's only Telegram or
    email identity (app.tasks), never a channel that cannot receive later (the terminal,
    the summary said "terminal" for a task delivered on Telegram)."""
    if not spec.get("schedule_kind"):
        return asking.value if asking else None
    async with session_scope() as session:
        try:
            identity = await tasks._identity_for(session, user_id, spec.get("delivery_identity_id"))
        except Exception:  # noqa: BLE001 - agent_spec's check gives the reason
            return None
        return identity.channel.value


async def confirm(state: BuilderState) -> Command:
    timezone, asking = await _user_and_identity(state["user_id"], state["identity_id"])
    channel = await _delivery_channel(state["user_id"], state["spec"], asking)
    editing = bool(state.get("edit_agent_id"))
    now = datetime.now(UTC)
    if editing:
        text = changes_text(state["original"], state["spec"], timezone, now)
        choices = ["yes", "no"]
    else:
        text = summary(state["spec"], timezone, channel, now)
        choices = list(CHOICES)
    nonce = secrets.token_hex(4)
    answer = interrupt({"say": text, "choices": choices, "nonce": nonce})
    choice = classify(answer)
    if choice == "yes":
        return Command(goto="update" if editing else "create")
    if choice == "test" and editing:
        choice = None  # a change is applied or not: its runs are the test
    if choice == "no":
        return Command(goto="cancel")
    if choice == "test":
        return Command(goto="test")
    if state.get("changes", 0) >= MAX_CHANGES:
        return Command(goto="cancel", update={"say": CANCELLED + " (too many changes)"})
    notes = [*state.get("notes", []), {"role": "user", "text": f"Change: {answer}"}]
    return Command(
        goto="extract",
        update={"notes": notes, "changes": state.get("changes", 0) + 1, "errors": []},
    )


async def _create(state: BuilderState, *, enabled: bool) -> tuple[int, int | None]:
    async with session_scope() as session:
        agent, task = await agent_spec.create_agent_from_spec(
            session, state["user_id"], state["spec"], actor=f"user:{state['user_id']}"
        )
        if task is not None and not enabled:
            task.enabled, task.next_run_at = False, None
        await session.commit()
        return agent.id, task.id if task else None


def _created_text(spec: dict, task_id: int | None) -> str:
    text = f"Agent {spec['name']!r} created."
    if task_id is not None:
        text += f" Its task is #{task_id} (/task lists it)."
    return text + f" /agent {spec['name']} talks to it."


async def create(state: BuilderState) -> dict:
    try:
        _agent_id, task_id = await _create(state, enabled=True)
    except agent_spec.SpecRefusedError as exc:
        return {"say": f"The agent was not created: {'; '.join(exc.errors)}"}
    return {"say": _created_text(state["spec"], task_id)}


FIELD_LABELS = (
    ("name", "Name"), ("purpose", "Purpose"), ("system_prompt", "Instructions"),
    ("model", "Model"), ("memory_mode", "Memory"), ("tools", "Tools"), ("skills", "Skills"),
    ("task_prompt", "Each run asks"), ("unattended", "Used without asking in scheduled runs"),
)  # fmt: skip


def changes_text(before: dict, after: dict, timezone: str | None, now: datetime) -> str:
    """What a change would do, field by field, with the next runs of a new schedule."""

    def show(value) -> str:
        if isinstance(value, list):
            return ", ".join(value) or "none"
        text = str(value) if value not in (None, "") else "none"
        return text if len(text) <= 200 else text[:197] + "..."

    lines = [f"Changes to agent {before['name']!r}:"]
    for field, label in FIELD_LABELS:
        if (before.get(field) or None) != (after.get(field) or None):
            lines.append(f"{label}: {show(before.get(field))} -> {show(after.get(field))}")
    old = (before.get("schedule_kind"), before.get("schedule_expr"))
    new = (after.get("schedule_kind"), after.get("schedule_expr"))
    if old != new:
        shown = f"{new[0]} {new[1]}" if new[0] else "none (the task is deleted)"
        line = f"Schedule: {old[0] + ' ' + old[1] if old[0] else 'none'} -> {shown}"
        if new[0]:
            runs = next_runs(new[0], new[1], timezone, now)
            line += f" ({timezone or 'UTC'}; next runs {schedule_phrase.describe_runs(runs)})"
        lines.append(line)
    if len(lines) == 1:
        lines.append("Nothing would change.")
    lines.append(
        "Answer yes to apply it, no to keep the agent as it is, or tell me what else to change."
    )
    return "\n".join(lines)


async def update(state: BuilderState) -> dict:
    """Apply a change to the user's agent, within the rights the user has."""
    try:
        async with session_scope() as session:
            agent, task = await agent_spec.update_agent_from_spec(
                session, state["user_id"], state["edit_agent_id"], state["spec"],
                actor=f"user:{state['user_id']}",
            )  # fmt: skip
            await session.commit()
            name, when = agent.name, task.next_run_at if task else None
            timezone = (await session.get(User, state["user_id"])).timezone
    except (agent_spec.SpecRefusedError, NotFoundError) as exc:
        return {"say": f"The agent was not changed: {exc}"}
    text = f"Agent {name!r} updated."
    if when is not None:
        local = tasks._as_utc(when).astimezone(ZoneInfo(timezone or "UTC"))
        text += f" Next run: {local.strftime('%a %Y-%m-%d %H:%M')} ({timezone or 'UTC'})."
    return {"say": text}


async def cancel(state: BuilderState) -> dict:
    if state.get("edit_agent_id") and not state.get("say"):
        return {"say": EDIT_CANCELLED}
    return {"say": state.get("say") or CANCELLED}


async def test(state: BuilderState) -> Command:
    """Create the agent with its task stopped and run the task once now (the result is
    delivered like any task result). No interrupt here: LangGraph runs a node again from its
    start when a dialogue resumes, so the question is asked by `keep`."""
    try:
        agent_id, task_id = await _create(state, enabled=False)
    except agent_spec.SpecRefusedError as exc:
        refused = f"The agent was not created: {'; '.join(exc.errors)}"
        return Command(goto=END, update={"say": refused})
    if task_id is None:
        text = (f"Agent {state['spec']['name']!r} created (it has no schedule to test). "
                f"/agent {state['spec']['name']} talks to it.")  # fmt: skip
        return Command(goto=END, update={"say": text})
    result = await tasks.execute(task_id, trigger="builder-test")
    return Command(
        goto="keep",
        update={"draft_agent_id": agent_id, "draft_task_id": task_id, "say": result["status"]},
    )


async def keep(state: BuilderState) -> dict:
    """After the test run: yes starts the schedule; anything else keeps nothing running."""
    task_id, agent_id, name = state["draft_task_id"], state["draft_agent_id"], state["spec"]["name"]
    answer = interrupt(
        {
            "say": f'Test run: {state.get("say")} (the result is the message "Scheduled task '
            f'#{task_id}"). Keep the agent and start its schedule? Answer yes or no.',
            "choices": ["yes", "no"],
            "nonce": secrets.token_hex(4),
        }
    )
    actor, user_id = f"user:{state['user_id']}", state["user_id"]
    async with session_scope() as session:
        if classify(answer) == "yes":
            await tasks.update_task(session, task_id, {"enabled": True}, actor=actor,
                                    user_id=user_id)  # fmt: skip
            await session.commit()
            return {"say": _created_text(state["spec"], task_id)}
        # Agents are disabled, never deleted: the audit trail of the test run refers to it.
        if await session.get(ScheduledTask, task_id) is not None:
            await tasks.delete_task(session, task_id, actor=actor, user_id=user_id)
        agent = await session.get(Agent, agent_id)
        if agent is not None:
            agent.is_active = False
        await session.commit()
    return {"say": f"Agent {name!r} disabled and its task deleted: nothing runs."}


def build_graph() -> StateGraph:
    graph = StateGraph(BuilderState)
    graph.add_node("extract", extract)
    graph.add_node("ask", ask)
    graph.add_node("give_up", give_up)
    graph.add_node("timezone", ask_timezone, destinations=("confirm", "timezone", "give_up"))
    graph.add_node(
        "confirm", confirm, destinations=("create", "cancel", "test", "extract", "update")
    )
    graph.add_node("update", update)
    graph.add_node("create", create)
    graph.add_node("cancel", cancel)
    graph.add_node("test", test, destinations=("keep", END))
    graph.add_node("keep", keep)
    graph.add_edge(START, "extract")
    graph.add_conditional_edges(
        "extract", after_extract, ["confirm", "ask", "give_up", "timezone"]
    )
    graph.add_edge("ask", "extract")
    for node in ("give_up", "create", "cancel", "keep", "update"):
        graph.add_edge(node, END)
    return graph


_compiled: dict = {}


async def get_graph():
    """Compiled on the conversations' checkpointer (app.graph), reopened with it."""
    from app import graph as conversation_graph

    await conversation_graph.get_graph()
    saver = conversation_graph._state["saver"]
    if _compiled.get("saver") is not saver:
        _compiled.update(saver=saver, graph=build_graph().compile(checkpointer=saver))
    return _compiled["graph"]


# --- the dialogue seen from a channel ---


async def waiting(thread: str) -> dict | None:
    """The question an open dialogue waits on ({"say", "choices", "nonce"}, and "updated_at"),
    or None when no dialogue is open in this thread."""
    graph = await get_graph()
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread}})
    if not snapshot.next:
        return None
    for task in snapshot.tasks:
        for item in task.interrupts:
            return {**item.value, "updated_at": snapshot.values.get("updated_at", 0.0)}
    return None


async def drop(thread: str) -> None:
    from app.graph import delete_threads

    await delete_threads([thread])


def _output(result: dict) -> dict:
    """What to send after a step: the question of the interrupt it stopped on, or the final
    message."""
    pending = result.get("__interrupt__") or []
    if pending:
        return dict(pending[0].value)
    return {"say": result.get("say") or CANCELLED, "choices": []}


async def start(thread: str, user_id: int, identity_id: int, request: str) -> dict:
    """Open a dialogue (an earlier one in this thread is dropped) and run its first step."""
    await drop(thread)
    graph = await get_graph()
    state: BuilderState = {
        "user_id": user_id,
        "identity_id": identity_id,
        "notes": [{"role": "user", "text": request[:MAX_REQUEST]}],
        "questions": 0,
        "changes": 0,
        "updated_at": time.time(),
    }
    result = await graph.ainvoke(state, {"configurable": {"thread_id": thread}})
    return _output(result)


async def start_edit(
    thread: str, user_id: int, identity_id: int, agent_id: int, change: str
) -> dict:
    """Open a dialogue that changes one of the user's agents: the model starts from the
    agent's current specification and the requested change."""
    await drop(thread)
    async with session_scope() as session:
        agent = await agent_spec.own_agent(session, user_id, agent_id)
        original = await agent_spec.spec_of(session, agent)
    graph = await get_graph()
    state: BuilderState = {
        "user_id": user_id,
        "identity_id": identity_id,
        "edit_agent_id": agent_id,
        "original": original,
        "spec": dict(original),
        "notes": [{"role": "user", "text": f"Change to this agent: {change[:MAX_REQUEST]}"}],
        "questions": 0,
        "changes": 0,
        "updated_at": time.time(),
    }
    result = await graph.ainvoke(state, {"configurable": {"thread_id": thread}})
    return _output(result)


async def answer(thread: str, text: str) -> dict:
    graph = await get_graph()
    result = await graph.ainvoke(
        Command(resume=text[:MAX_REQUEST]), {"configurable": {"thread_id": thread}}
    )
    return _output(result)


def is_stale(pending: dict, now: float | None = None) -> bool:
    return (now or time.time()) - float(pending.get("updated_at") or 0.0) > IDLE_SECONDS
