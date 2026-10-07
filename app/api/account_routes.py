"""Named administrators and their tokens, service in
app/admin/accounts.py.

Signing in is `POST /auth/token` with the name and password in an `Authorization: Basic` header
and the TOTP code in `X-TOTP`: credentials travel in headers only, so they are checked before
any body is read, like every other call. It answers a token, shown once.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import accounts
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.schemas import SECRET_FIELD
from app.api.scopes import Principal, Scope, get_principal, require

router = APIRouter()

NAME_PATH = Path(min_length=2, max_length=24, description="the administrator's name")


class TokenOut(BaseModel):
    id: int
    account: str
    label: str
    scope: str
    expires_at: datetime
    token: str = Field(description="shown once: keep it, it is never shown again")


class TokenRowOut(BaseModel):
    id: int
    account: str
    label: str
    scope: str
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    last_used_at: datetime | None


class AdminIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=2, max_length=24, description="lowercase letters, digits, . _ -")
    scope: str = Field(description="read, operate, admin or owner")
    password: str = Field(
        min_length=1, max_length=256, description="12 characters or more",
        json_schema_extra=SECRET_FIELD,
    )  # fmt: skip
    with_totp: bool = Field(
        default=False, description="a second factor below owner (an owner always has one)"
    )


class PasswordIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(
        min_length=1, max_length=256, description="12 characters or more",
        json_schema_extra=SECRET_FIELD,
    )  # fmt: skip


class AdminOut(BaseModel):
    name: str
    scope: str
    totp: bool
    created_at: datetime
    disabled_at: datetime | None


class AdminSecretOut(AdminOut):
    totp_secret: str | None = Field(description="shown once: add it to an authenticator app")
    totp_uri: str | None = Field(description="the same secret as an otpauth:// link")


def _admin(account) -> dict:
    return {
        "name": account.name,
        "scope": account.scope,
        "totp": account.totp_secret is not None,
        "created_at": account.created_at,
        "disabled_at": account.disabled_at,
    }


def _with_secret(result: accounts.NewSecret) -> dict:
    return {
        **_admin(result.account),
        "totp_secret": result.totp_secret,
        "totp_uri": result.totp_uri,
    }


async def _token_row(session: AsyncSession, token) -> dict:
    from app.db.models import AdminAccount

    account = await session.get(AdminAccount, token.account_id)
    return {
        "id": token.id,
        "account": account.name if account else "?",
        "label": token.label,
        "scope": token.scope,
        "created_at": token.created_at,
        "expires_at": token.expires_at,
        "revoked_at": token.revoked_at,
        "last_used_at": token.last_used_at,
    }


@router.post(
    "/auth/token",
    dependencies=[require(Scope.READ)],
    response_model=TokenOut,
    status_code=status.HTTP_201_CREATED,
    tags=["administrators"],
    responses=error_responses(409),
)
async def sign_in(
    label: str = Query(default="cli", max_length=32, description="what this token is for"),
    scope: str | None = Query(default=None, description="at most the account's scope"),
    hours: int = Query(
        default=accounts.DEFAULT_TOKEN_HOURS, ge=1, le=accounts.MAX_TOKEN_HOURS,
        description="how long the token lasts",
    ),  # fmt: skip
    principal: Principal = Depends(get_principal),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Sign in: name and password in a Basic header, the code in X-TOTP; answers a token."""
    if principal.account_id is None or principal.token_id is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Sign in with a name and a password (Authorization: Basic) to get a token",
        )
    from app.db.models import AdminAccount

    account = await session.get(AdminAccount, principal.account_id)
    token, raw = await accounts.issue_token(
        session, account, label=label, scope=scope, hours=hours, actor=principal.actor
    )
    await session.commit()
    return {
        "id": token.id, "account": account.name, "label": token.label, "scope": token.scope,
        "expires_at": token.expires_at, "token": raw,
    }  # fmt: skip


@router.get(
    "/auth/tokens",
    dependencies=[require(Scope.READ)],
    response_model=list[TokenRowOut],
    tags=["administrators"],
    responses=error_responses(),
)
async def list_tokens(
    every_account: bool = Query(default=False, description="every account's tokens (owner)"),
    principal: Principal = Depends(get_principal),
    session: AsyncSession = Depends(get_db_session),
) -> list[dict]:
    """The caller's tokens (never the token itself), or every account's for an owner."""
    if every_account and principal.scope < Scope.OWNER:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient scope")
    if not every_account and principal.account_id is None:
        return []  # the static API key has no account, so no token
    rows = await accounts.list_tokens(session, None if every_account else principal.account_id)
    return [await _token_row(session, t) for t in rows]


@router.delete(
    "/auth/tokens/{token_id}",
    dependencies=[require(Scope.READ)],
    response_model=TokenRowOut,
    tags=["administrators"],
    responses=error_responses(404, 409),
)
async def revoke_token(
    token_id: int = Path(ge=1),
    principal: Principal = Depends(get_principal),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Revoke a token: one of the caller's, or any for an owner."""
    owner = principal.scope >= Scope.OWNER
    if not owner and principal.account_id is None:
        raise accounts.TokenNotFoundError(f"No token {token_id}")
    token = await accounts.revoke_token(
        session, token_id, account_id=None if owner else principal.account_id,
        actor=current_actor(),
    )  # fmt: skip
    await session.commit()
    return await _token_row(session, token)


@router.get(
    "/admins",
    dependencies=[require(Scope.OWNER)],
    response_model=list[AdminOut],
    tags=["administrators"],
    responses=error_responses(),
)
async def list_admins(session: AsyncSession = Depends(get_db_session)) -> list[dict]:
    """Every named administrator, disabled ones included."""
    return [_admin(a) for a in await accounts.list_accounts(session)]


@router.post(
    "/admins",
    dependencies=[require(Scope.OWNER)],
    response_model=AdminSecretOut,
    status_code=status.HTTP_201_CREATED,
    tags=["administrators"],
    responses=error_responses(409),
)
async def create_admin(body: AdminIn, session: AsyncSession = Depends(get_db_session)) -> dict:
    """Create a named administrator; an owner's TOTP secret is shown once."""
    result = await accounts.create_account(
        session, name=body.name, scope=body.scope, password=body.password,
        with_totp=body.with_totp, actor=current_actor(),
    )  # fmt: skip
    await session.commit()
    return _with_secret(result)


@router.post(
    "/admins/{name}/disable",
    dependencies=[require(Scope.OWNER)],
    response_model=AdminOut,
    tags=["administrators"],
    responses=error_responses(404, 409),
)
async def disable_admin(
    name: str = NAME_PATH, session: AsyncSession = Depends(get_db_session)
) -> dict:
    """Disable an administrator and revoke their tokens (never the last owner)."""
    account = await accounts.disable_account(session, name, actor=current_actor())
    await session.commit()
    return _admin(account)


@router.post(
    "/admins/{name}/password",
    dependencies=[require(Scope.READ)],
    response_model=AdminOut,
    tags=["administrators"],
    responses=error_responses(404, 409),
)
async def set_admin_password(
    body: PasswordIn,
    name: str = NAME_PATH,
    principal: Principal = Depends(get_principal),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Change a password: one's own, or anyone's for an owner."""
    if principal.scope < Scope.OWNER:
        own = principal.account_id is not None and (
            (await accounts.get_account(session, name)).id == principal.account_id
        )
        if not own:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient scope")
    account = await accounts.set_password(session, name, body.password, actor=current_actor())
    await session.commit()
    return _admin(account)


@router.post(
    "/admins/{name}/totp",
    dependencies=[require(Scope.OWNER)],
    response_model=AdminSecretOut,
    tags=["administrators"],
    responses=error_responses(404, 409),
)
async def reset_admin_totp(
    name: str = NAME_PATH, session: AsyncSession = Depends(get_db_session)
) -> dict:
    """Give an administrator a new TOTP secret, shown once (a lost phone)."""
    result = await accounts.reset_totp(session, name, actor=current_actor())
    await session.commit()
    return _with_secret(result)
