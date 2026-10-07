"""The application's side of the host helper: one signed call, after the Admin API's
own authorization. The helper re-checks the signature, the scope claim and its table.

Where: HOST_HELPER_URL, else unix://HOST_HELPER_SOCKET, else http://127.0.0.1:HOST_HELPER_PORT.
The container gets http://host.docker.internal:PORT (Docker Desktop) or a unix:// socket
(Linux Docker Engine) from its compose file.
"""

import json
from urllib.parse import urlencode

import httpx

from app.admin.service import ConflictError, InvalidInputError, NotFoundError
from app.config import get_settings
from app.host.signing import sign

TIMEOUT_SECONDS = 30.0
JOB_PREFIX = "host-"


class HelperError(ConflictError):
    """The helper is off, unreachable or refused the call: a state of the host (409)."""


def enabled() -> bool:
    return get_settings().host_helper_enabled


def helper_url() -> str:
    settings = get_settings()
    if settings.host_helper_url:
        return settings.host_helper_url
    if settings.host_helper_socket:
        return "unix://" + settings.host_helper_socket
    return f"http://127.0.0.1:{settings.host_helper_port}"


def _client() -> httpx.AsyncClient:
    url = helper_url()
    if url.startswith("unix://"):
        transport = httpx.AsyncHTTPTransport(uds=url[len("unix://") :])
        return httpx.AsyncClient(
            transport=transport, base_url="http://helper", timeout=TIMEOUT_SECONDS
        )
    return httpx.AsyncClient(base_url=url.rstrip("/"), timeout=TIMEOUT_SECONDS)


async def call(
    method: str,
    path: str,
    *,
    scope: str,
    actor: str,
    body: dict | None = None,
    params: dict | None = None,
):
    """Send one signed call; the helper's JSON answer, or the service error its status means."""
    settings = get_settings()
    if not settings.host_helper_enabled:
        raise HelperError(
            "The host helper is off: set HOST_HELPER_ENABLED=true on the host and restart "
            "(./start.sh --config HOST_HELPER_ENABLED=true)"
        )
    if not settings.host_helper_secret:
        raise HelperError("HOST_HELPER_SECRET is not set for the application")
    target = path + (f"?{urlencode(params)}" if params else "")
    content = json.dumps(body).encode() if body is not None else b""
    headers = sign(settings.host_helper_secret, method, target, content, scope=scope, actor=actor)
    if body is not None:
        headers["content-type"] = "application/json"
    try:
        async with _client() as client:
            response = await client.request(method, target, content=content, headers=headers)
    except httpx.HTTPError as exc:
        raise HelperError(
            f"The host helper does not answer at {helper_url()} ({type(exc).__name__}): "
            "start the application on the host with HOST_HELPER_ENABLED=true "
            "(the helper starts with it)"
        ) from None
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text[:300]}
    if response.status_code < 400:
        return payload
    detail = str(payload.get("detail", "")) if isinstance(payload, dict) else ""
    if response.status_code == 404:
        raise NotFoundError(detail)
    if response.status_code == 409:
        raise ConflictError(detail or "The host helper refused: conflict")
    if response.status_code == 422:
        raise InvalidInputError(detail or "The host helper refused the arguments")
    if response.status_code in (401, 403):
        raise HelperError(
            f"The host helper refused the call ({detail}): check that HOST_HELPER_SECRET is "
            "the same for the application and the helper"
        )
    raise HelperError(f"The host helper answered {response.status_code}: {detail}")
