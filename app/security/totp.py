"""Time-based one-time codes (TOTP, RFC 6238) for the second factor (S2).

An authenticator app (on a phone) and the server share a random secret; both compute
HMAC-SHA1(secret, number of 30-second steps since 1970) and keep 6 digits. The server
accepts the current step and one step on each side (clock drift), and never the same step
twice for one account (`last_step`): a code seen by someone else is already spent.

Standard library only. The secret is stored encrypted (`AdminAccount.totp_secret`).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote, urlencode

STEP_SECONDS = 30
DIGITS = 6
DRIFT_STEPS = 1
SECRET_BYTES = 20
ISSUER = "ChannelAgent"


def new_secret() -> str:
    """A random secret in base32, the form authenticator apps take."""
    return base64.b32encode(secrets.token_bytes(SECRET_BYTES)).decode("ascii").rstrip("=")


def provisioning_uri(name: str, secret: str) -> str:
    """The `otpauth://` link an authenticator app reads (as text or as a QR code)."""
    query = urlencode(
        {"secret": secret, "issuer": ISSUER, "digits": DIGITS, "period": STEP_SECONDS}
    )
    return f"otpauth://totp/{quote(ISSUER)}:{quote(name)}?{query}"


def _key(secret: str) -> bytes:
    return base64.b32decode(secret.upper() + "=" * (-len(secret) % 8))


def code_at(secret: str, step: int) -> str:
    digest = hmac.new(_key(secret), struct.pack(">Q", step), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    number = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(number % 10**DIGITS).zfill(DIGITS)


def current_step(now: float | None = None) -> int:
    return int((time.time() if now is None else now) // STEP_SECONDS)


def matching_step(
    secret: str, code: str | None, last_step: int | None, now: float | None = None
) -> int | None:
    """The step this code belongs to, or None when it is wrong, malformed or already used.
    The caller stores the step as the account's `last_step`."""
    if not code or len(code) != DIGITS or not code.isdigit():
        return None
    step = current_step(now)
    found = None
    for candidate in range(step - DRIFT_STEPS, step + DRIFT_STEPS + 1):
        # Every candidate is computed and compared, so the time does not tell which matched.
        if hmac.compare_digest(code_at(secret, candidate), code) and found is None:
            found = candidate
    if found is None or (last_step is not None and found <= last_step):
        return None
    return found
