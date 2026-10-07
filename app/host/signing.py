"""Signed calls between the Admin API and the host helper (.

Every call carries an HMAC-SHA256 over the method, the path, the SHA-256 of the body, a
timestamp, a nonce, the scope the application granted and the actor, keyed with
HOST_HELPER_SECRET (in `.env`, shared by the application and the helper only). The helper
refuses a call without a valid signature, a call older than MAX_AGE_SECONDS (or dated before
the helper started, so a restart does not reopen the replay window) and a nonce it has
already seen: a request captured on the way cannot be sent again.
"""

import hashlib
import hmac
import secrets
import time

H_TIMESTAMP = "x-host-timestamp"
H_NONCE = "x-host-nonce"
H_SCOPE = "x-host-scope"
H_ACTOR = "x-host-actor"
H_SIGNATURE = "x-host-signature"

MAX_AGE_SECONDS = 30
MAX_NONCES = 10_000


class SignatureError(Exception):
    """A call the helper refuses; the message says why and holds no secret."""


def _canonical(
    method: str, path: str, body: bytes, timestamp: str, nonce: str, scope: str, actor: str
) -> bytes:
    digest = hashlib.sha256(body).hexdigest()
    return "\n".join((method.upper(), path, digest, timestamp, nonce, scope, actor)).encode()


def _mac(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def sign(
    secret: str,
    method: str,
    path: str,
    body: bytes,
    *,
    scope: str,
    actor: str,
    now: float | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """The headers that authenticate one call. `path` includes the query string, if any."""
    timestamp = str(int(time.time() if now is None else now))
    nonce = nonce or secrets.token_hex(16)
    return {
        H_TIMESTAMP: timestamp,
        H_NONCE: nonce,
        H_SCOPE: scope,
        H_ACTOR: actor,
        H_SIGNATURE: _mac(secret, _canonical(method, path, body, timestamp, nonce, scope, actor)),
    }


class Verifier:
    """Checks the signed calls on the helper side, and remembers the nonces it has seen."""

    def __init__(self, secret: str, started_at: float | None = None) -> None:
        self._secret = secret
        self._started_at = int(time.time() if started_at is None else started_at)
        self._seen: dict[str, float] = {}

    def verify(
        self,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
        now: float | None = None,
    ) -> tuple[str, str]:
        """(scope, actor) of a valid call; SignatureError otherwise. The nonce is recorded
        only once the signature is known to be right, so garbage cannot fill the store."""
        now = time.time() if now is None else now
        try:
            timestamp, nonce = headers[H_TIMESTAMP], headers[H_NONCE]
            scope, actor, given = headers[H_SCOPE], headers[H_ACTOR], headers[H_SIGNATURE]
        except KeyError:
            raise SignatureError("unsigned call") from None
        message = _canonical(method, path, body, timestamp, nonce, scope, actor)
        expected = _mac(self._secret, message)
        if not hmac.compare_digest(given, expected):
            raise SignatureError("bad signature")
        if not timestamp.isdigit():
            raise SignatureError("bad timestamp")
        stamp = int(timestamp)
        if stamp < self._started_at or abs(now - stamp) > MAX_AGE_SECONDS:
            raise SignatureError("expired call")
        self._forget(now)
        if nonce in self._seen:
            raise SignatureError("replayed call")
        if len(self._seen) >= MAX_NONCES:
            raise SignatureError("too many calls in the replay window")
        self._seen[nonce] = now
        return scope, actor

    def _forget(self, now: float) -> None:
        # Kept for twice the window: a nonce is refused at least as long as its call is.
        for nonce in [n for n, at in self._seen.items() if now - at > 2 * MAX_AGE_SECONDS]:
            del self._seen[nonce]
