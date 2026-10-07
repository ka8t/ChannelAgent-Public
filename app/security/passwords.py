"""Password hashing for named administrators (S1).

`hashlib.scrypt` from the standard library: no new dependency (the lock file stays small).
scrypt is a deliberately slow and memory-hard function, so a stolen database does not give
the passwords back cheaply. The stored text carries its parameters and its random salt:

    scrypt$<log2 n>$<r>$<p>$<salt, base64>$<hash, base64>

so raising the cost later keeps the older hashes readable. `verify_password` compares in
constant time, and `dummy_hash()` lets a sign-in with an unknown name pay the same cost as a
wrong password (the time of the answer does not tell which names exist).
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from functools import cache

LOG2_N = 15  # n = 32768: about 32 MiB of memory per check (128 * r * n bytes)
R = 8
P = 1
SALT_BYTES = 16
HASH_BYTES = 32
MAX_MEMORY = 64 * 1024 * 1024
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 256


def password_problem(password: str) -> str | None:
    """Why this password is refused, or None. Length only: a long passphrase beats rules."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"must have at least {MIN_PASSWORD_LENGTH} characters"
    if len(password) > MAX_PASSWORD_LENGTH:
        return f"must have at most {MAX_PASSWORD_LENGTH} characters"
    return None


def _derive(password: str, salt: bytes, log2_n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**log2_n, r=r, p=p, maxmem=MAX_MEMORY,
        dklen=HASH_BYTES,
    )  # fmt: skip


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(password, salt, LOG2_N, R, P)
    return f"scrypt${LOG2_N}${R}${P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        name, log2_n, r, p, salt, digest = stored.split("$")
        if name != "scrypt":
            return False
        expected = base64.b64decode(digest)
        given = _derive(password, base64.b64decode(salt), int(log2_n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(given, expected)


@cache
def dummy_hash() -> str:
    """A valid hash of a random password nobody knows, checked when the name does not exist.
    Made on first use, not at import (one scrypt run)."""
    return hash_password(secrets.token_urlsafe(24))
