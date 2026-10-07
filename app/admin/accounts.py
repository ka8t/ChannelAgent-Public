"""Named administrators and their tokens (S1, S2, S3).

- An account has a name, a scope, a scrypt password hash (app/security/passwords.py) and,
  for an `owner` always and below on request, a TOTP secret (app/security/totp.py). It is
  disabled, never deleted, so the admin events keep naming it.
- Signing in (`check_credentials`) takes a name, a password and a code; it costs one scrypt
  run whether the name exists or not.
- A token is 32 random bytes shown once; only its SHA-256 is stored. Its scope is never above
  its account's, it expires, it can be revoked, and it dies with its account.
- Once one enabled `owner` account exists, the static `API_SERVER_KEY` stops working over
  HTTP (app/api/deps.py): `named_owner_exists`.

Every change is an admin event, in the caller's transaction. No secret ever goes into an
event, a log or an error: the password, the token and the TOTP secret are returned once to
the caller that made them and never stored in clear.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import (
    ConflictError,
    InvalidInputError,
    NotFoundError,
    record_admin_event,
)
from app.db.models import AdminAccount, ApiToken
from app.security import passwords, totp

SCOPES = ("read", "operate", "admin", "owner")
NAME = re.compile(r"[a-z][a-z0-9._-]{1,23}")
TOKEN_PREFIX = "ca_"
TOKEN_BYTES = 32
DEFAULT_TOKEN_HOURS = 30 * 24
MAX_TOKEN_HOURS = 365 * 24
LABEL = re.compile(r"[A-Za-z0-9._:-]{1,32}")
# `last_used_at` is written at most once a minute per token: not a write per request.
LAST_USED_RESOLUTION = timedelta(minutes=1)


class AccountNotFoundError(NotFoundError):
    pass


class TokenNotFoundError(NotFoundError):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    """SQLite gives back naive datetimes: they were stored in UTC."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def scope_rank(scope: str) -> int:
    return SCOPES.index(scope) + 1


def actor_of(account: AdminAccount) -> str:
    return f"adm:{account.name}"


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _check_name(name: str) -> str:
    name = (name or "").strip().lower()
    if not NAME.fullmatch(name):
        raise InvalidInputError(
            "An administrator name is 2 to 24 characters: a lowercase letter, then lowercase "
            "letters, digits, '.', '_' or '-'"
        )
    return name


def _check_scope(scope: str) -> str:
    if scope not in SCOPES:
        raise InvalidInputError(f"The scope is one of: {', '.join(SCOPES)}")
    return scope


def _check_password(password: str) -> str:
    problem = passwords.password_problem(password or "")
    if problem:
        raise InvalidInputError(f"The password {problem}")
    return password


async def _hash(password: str) -> str:
    # scrypt takes tens of milliseconds and 32 MiB: off the event loop.
    return await asyncio.to_thread(passwords.hash_password, password)


async def named_owner_exists(session: AsyncSession) -> bool:
    count = await session.scalar(
        select(func.count())
        .select_from(AdminAccount)
        .where(AdminAccount.scope == "owner", AdminAccount.disabled_at.is_(None))
    )
    return bool(count)


async def get_account(session: AsyncSession, name: str) -> AdminAccount:
    account = await session.scalar(
        select(AdminAccount).where(AdminAccount.name == (name or "").strip().lower())
    )
    if account is None:
        raise AccountNotFoundError(f"No administrator named {name!r}")
    return account


async def list_accounts(session: AsyncSession) -> list[AdminAccount]:
    return list((await session.scalars(select(AdminAccount).order_by(AdminAccount.name))).all())


@dataclass
class NewSecret:
    """What a creation or a reset gives back once: the TOTP secret and its link."""

    account: AdminAccount
    totp_secret: str | None
    totp_uri: str | None


def _new_totp(account: AdminAccount) -> NewSecret:
    secret = totp.new_secret()
    account.totp_secret = secret
    account.totp_last_step = None
    return NewSecret(account, secret, totp.provisioning_uri(account.name, secret))


async def create_account(
    session: AsyncSession,
    *,
    name: str,
    scope: str,
    password: str,
    with_totp: bool = False,
    actor: str,
) -> NewSecret:
    """A new administrator. An `owner` always gets a TOTP secret; below, on request."""
    name, scope, password = _check_name(name), _check_scope(scope), _check_password(password)
    if await session.scalar(select(AdminAccount.id).where(AdminAccount.name == name)):
        raise ConflictError(f"An administrator named {name!r} already exists")
    account = AdminAccount(name=name, scope=scope, password_hash=await _hash(password))
    session.add(account)
    result = (
        _new_totp(account) if scope == "owner" or with_totp else NewSecret(account, None, None)
    )
    await session.flush()
    await record_admin_event(
        session, actor=actor, action="admin.create", target_type="admin_account",
        target_id=account.id,
        details={"name": name, "scope": scope, "totp": result.totp_secret is not None},
    )  # fmt: skip
    return result


async def _enabled_owners(session: AsyncSession) -> int:
    return int(
        await session.scalar(
            select(func.count())
            .select_from(AdminAccount)
            .where(AdminAccount.scope == "owner", AdminAccount.disabled_at.is_(None))
        )
        or 0
    )


async def disable_account(session: AsyncSession, name: str, *, actor: str) -> AdminAccount:
    """Disabled and every token revoked. The last enabled owner cannot be disabled through the
    API: nobody could administer it any more (the in-process command still can, app stopped)."""
    account = await get_account(session, name)
    if account.disabled_at is not None:
        return account
    if account.scope == "owner" and await _enabled_owners(session) <= 1:
        raise ConflictError("The last enabled owner cannot be disabled")
    now = _now()
    account.disabled_at = now
    for token in await session.scalars(
        select(ApiToken).where(ApiToken.account_id == account.id, ApiToken.revoked_at.is_(None))
    ):
        token.revoked_at = now
    await record_admin_event(
        session, actor=actor, action="admin.disable", target_type="admin_account",
        target_id=account.id, details={"name": account.name},
    )  # fmt: skip
    return account


async def set_password(
    session: AsyncSession, name: str, password: str, *, actor: str
) -> AdminAccount:
    account = await get_account(session, name)
    account.password_hash = await _hash(_check_password(password))
    await record_admin_event(
        session, actor=actor, action="admin.password", target_type="admin_account",
        target_id=account.id, details={"name": account.name},
    )  # fmt: skip
    return account


async def reset_totp(session: AsyncSession, name: str, *, actor: str) -> NewSecret:
    account = await get_account(session, name)
    result = _new_totp(account)
    await record_admin_event(
        session, actor=actor, action="admin.totp_reset", target_type="admin_account",
        target_id=account.id, details={"name": account.name},
    )  # fmt: skip
    return result


class SignInError(Exception):
    """A refused sign-in. One message for every reason, so the answer does not tell whether
    the name exists, the password was wrong or the code was missing."""

    MESSAGE = "Invalid name, password or code"


async def check_credentials(
    session: AsyncSession, name: str, password: str, code: str | None
) -> AdminAccount:
    """The enabled account these credentials open. One scrypt run whatever the outcome (an
    unknown name checks a dummy hash); the TOTP step used is stored, so a code works once."""
    account = await session.scalar(
        select(AdminAccount).where(AdminAccount.name == (name or "").strip().lower())
    )
    stored = account.password_hash if account is not None else passwords.dummy_hash()
    good = await asyncio.to_thread(passwords.verify_password, password or "", stored)
    if account is None or not good or account.disabled_at is not None:
        raise SignInError(SignInError.MESSAGE)
    if account.totp_secret is not None:
        step = totp.matching_step(account.totp_secret, code, account.totp_last_step)
        if step is None:
            raise SignInError(SignInError.MESSAGE)
        account.totp_last_step = step
    return account


async def issue_token(
    session: AsyncSession,
    account: AdminAccount,
    *,
    label: str,
    scope: str | None = None,
    hours: int = DEFAULT_TOKEN_HOURS,
    actor: str,
) -> tuple[ApiToken, str]:
    """A new token for `account`: (the row, the token in clear, shown once)."""
    if not LABEL.fullmatch(label or ""):
        raise InvalidInputError("A token label is 1 to 32 letters, digits, '.', '_', ':' or '-'")
    scope = _check_scope(scope or account.scope)
    if scope_rank(scope) > scope_rank(account.scope):
        raise InvalidInputError(f"A token of {account.name!r} cannot go above {account.scope!r}")
    if not 1 <= hours <= MAX_TOKEN_HOURS:
        raise InvalidInputError(f"A token lasts 1 to {MAX_TOKEN_HOURS} hours")
    raw = TOKEN_PREFIX + secrets.token_urlsafe(TOKEN_BYTES)
    token = ApiToken(
        account_id=account.id, label=label, scope=scope, token_hash=hash_token(raw),
        expires_at=_now() + timedelta(hours=hours),
    )  # fmt: skip
    session.add(token)
    await session.flush()
    await record_admin_event(
        session, actor=actor, action="token.create", target_type="api_token",
        target_id=token.id,
        details={"account": account.name, "label": label, "scope": scope, "hours": hours},
    )  # fmt: skip
    return token, raw


async def resolve_token(
    session: AsyncSession, raw: str
) -> tuple[AdminAccount, ApiToken] | None:
    """The account and token behind a bearer token, or None when it is unknown, expired,
    revoked or its account disabled."""
    if not raw.startswith(TOKEN_PREFIX):
        return None
    token = await session.scalar(select(ApiToken).where(ApiToken.token_hash == hash_token(raw)))
    if token is None or token.revoked_at is not None:
        return None
    now = _now()
    if _aware(token.expires_at) <= now:
        return None
    account = await session.get(AdminAccount, token.account_id)
    if account is None or account.disabled_at is not None:
        return None
    if token.last_used_at is None or now - _aware(token.last_used_at) >= LAST_USED_RESOLUTION:
        token.last_used_at = now
        await session.commit()
    return account, token


async def list_tokens(session: AsyncSession, account_id: int | None) -> list[ApiToken]:
    """The tokens of one account, or of every account (`None`, owner only)."""
    query = select(ApiToken).order_by(ApiToken.id)
    if account_id is not None:
        query = query.where(ApiToken.account_id == account_id)
    return list((await session.scalars(query)).all())


async def revoke_token(
    session: AsyncSession, token_id: int, *, account_id: int | None, actor: str
) -> ApiToken:
    """Revoke a token of `account_id`, or any token (`None`, owner only). Someone else's
    token is "not found": its existence is not told."""
    token = await session.get(ApiToken, token_id)
    if token is None or (account_id is not None and token.account_id != account_id):
        raise TokenNotFoundError(f"No token {token_id}")
    if token.revoked_at is None:
        token.revoked_at = _now()
        await record_admin_event(
            session, actor=actor, action="token.revoke", target_type="api_token",
            target_id=token.id, details={"label": token.label},
        )  # fmt: skip
    return token
