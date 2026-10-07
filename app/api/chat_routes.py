"""The terminal channel: talk to an agent through the API, from `./start.sh --chat` on
this machine or another one (API_URL).

A message is a turn like any other channel's: it goes through `dispatch_event` (authorization,
per-user limits, the audit trail, the commands `/agent`, `/new`, `/newagent`, `/task`), as the
channel `terminal`, from the identity bound to the API account that sent it. The account comes
from the key, never from a header the client chooses: with the one key of today it is `owner`
(named keys are). An administrator binds it to a user like any identity (`add-channel-identity
--channel terminal --external-id owner`) and grants it chat.

A turn may take minutes, longer than one API request may (API_REQUEST_TIMEOUT_SECONDS): the
route starts a job and answers 202; the client follows it with `GET /chat/{job_id}`, which
shows only chat jobs. A tool whose policy is "confirm" asks in the job (`question`), and the
client answers with `POST /chat/{job_id}/answer`; without an answer within the tool's time, the
tool is refused, as on the other channels.
"""

import asyncio

from fastapi import APIRouter, status
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from app.admin.jobs import Job, registry
from app.admin.service import ConflictError, NotFoundError
from app.api.errors import error_responses
from app.api.schemas import JobOut
from app.api.scopes import Scope, require

router = APIRouter()

TERMINAL_ACCOUNT = "owner"  # the account of the one API key (will name keys)
CHAT_KIND = "chat"
MAX_TEXT = 4000

# job id -> (question, future the answer resolves)
_pending: dict[str, tuple[str, asyncio.Future]] = {}


class ChatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=MAX_TEXT, description="the message to the agent")


class AnswerIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: StrictBool = Field(description="true: yes, run the tool; false: no")


def terminal_account() -> str:
    return TERMINAL_ACCOUNT


def _chat_job(job_id: str) -> Job:
    job = registry.get(job_id)
    if job.kind != CHAT_KIND:
        raise NotFoundError(f"No chat job {job_id}")
    return job


@router.post(
    "/chat",
    dependencies=[require(Scope.OPERATE)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["chat"],
    responses=error_responses(409),
)
async def chat(body: ChatIn) -> Job:
    """Send a message to the agent of the terminal identity of this API account; the turn runs
    as a job (follow it with GET /chat/{job_id}). Its result holds the replies."""
    from app.channels.dispatch import dispatch_text
    from app.channels.schema import NormalizedEvent
    from app.db.models import Channel
    from app.db.session import session_scope

    account = terminal_account()

    async def runner(job: Job) -> dict:
        replies: list[str] = []

        async def reply(text: str) -> None:
            replies.append(text)
            job.update(message=f"{len(replies)} reply(s)")

        async def confirm(question: str, timeout: float) -> bool | None:
            future = asyncio.get_running_loop().create_future()
            _pending[job.id] = (question, future)
            job.result = {"replies": list(replies), "question": question}
            job.update(message="waiting for an answer")
            try:
                return await asyncio.wait_for(future, timeout)
            except TimeoutError:
                return None
            finally:
                _pending.pop(job.id, None)
                job.result = {"replies": list(replies), "question": None}
                job.update(message="running")

        event = NormalizedEvent(account, Channel.TERMINAL, body.text, reply, confirm=confirm)
        async with session_scope() as session:
            outcome = await dispatch_text(session, event)
        return {"replies": replies, "question": None, "outcome": outcome.value if outcome else None}

    return registry.start(CHAT_KIND, runner)


@router.get(
    "/chat/{job_id}",
    dependencies=[require(Scope.OPERATE)],
    response_model=JobOut,
    tags=["chat"],
    responses=error_responses(404),
)
async def get_chat(job_id: str) -> Job:
    """One chat job: running (with a `question` in its result while a tool waits for a yes),
    done (its replies) or failed."""
    return _chat_job(job_id)


@router.post(
    "/chat/{job_id}/answer",
    dependencies=[require(Scope.OPERATE)],
    response_model=JobOut,
    tags=["chat"],
    responses=error_responses(404, 409),
)
async def answer_chat(job_id: str, body: AnswerIn) -> Job:
    """Answer the question a tool asks in this chat job (409 when none is waiting)."""
    job = _chat_job(job_id)
    pending = _pending.get(job_id)
    if pending is None or pending[1].done():
        raise ConflictError(f"Chat job {job_id} is not waiting for an answer")
    pending[1].set_result(body.answer)
    await asyncio.sleep(0)
    return job
