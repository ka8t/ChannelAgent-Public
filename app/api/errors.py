"""The error responses every route documents in OpenAPI, so a generated client
and the UI know what to expect. Always 401, 403 and 429; 404 where a route has an
id; 409 where a change can conflict.
"""

_DESCRIPTIONS = {
    401: "Missing or invalid API key",
    403: "The caller's scope is too low for this route",
    404: "The user, agent, request or identity does not exist",
    409: "The change conflicts with the current state",
    429: "Too many failed authentication attempts from this address",
}


def error_responses(*extra: int) -> dict[int | str, dict[str, str]]:
    codes = (401, 403, 429, *extra)
    return {code: {"description": _DESCRIPTIONS[code]} for code in codes}
