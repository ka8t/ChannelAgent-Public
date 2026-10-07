"""Core schema: User, ChannelIdentity, Permission.

Design notes:

- A `User` is platform-independent. Each `ChannelIdentity` links one
  external account (a Telegram user id, an email address, a Matrix
  user id) on one channel to a `User`.
- `ChannelIdentity.external_id` is the lookup key the Auth Node
  queries by, so it must stay a plain, deterministic value — Fernet
  encryption is non-deterministic (a fresh nonce per call), so an
  encrypted column can never be looked up by equality. For Telegram
  and Matrix, the external id itself isn't the kind of sensitive data
  Flags for encryption (a numeric Telegram id or
  a Matrix user id, not "raw email addresses ... or personal
  metadata"). For email, the lookup key is a SHA-256 hash of the
  lowercased address (see app.security.hashing), matching the
  `email_{email_hash}` thread_id scheme in — the
  raw address itself is only ever stored in the encrypted
  `raw_address` column, used solely to send SMTP replies.
- `Permission` is a separate table, not a `role` column, so it maps
  directly onto the Admin API's grant/revoke semantics (POST/DELETE
  .../permissions) — granting is inserting a row, revoking is deleting
  one, with no separate "no permission" state to reconcile.
"""

import enum
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.db.types import EncryptedString


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _db_enum(enum_cls: type[enum.Enum]) -> Enum:
    # SQLAlchemy's Enum stores the member NAME by default (e.g. "EMAIL").
    # These are (str, Enum) members whose value is the lowercase wire
    # format ("email") used throughout normalized events and .env — store
    # that instead, so the raw DB content matches what the rest of the
    # app (and anyone inspecting the DB file directly) actually expects.
    return Enum(enum_cls, native_enum=False, values_callable=lambda cls: [e.value for e in cls])


class Channel(enum.StrEnum):
    TELEGRAM = "telegram"
    EMAIL = "email"
    MATRIX = "matrix"
    #`./start.sh --chat`, a turn sent through the API (app/api/chat_routes.py). Eight
    # characters, like "telegram": the stored column keeps its width, no migration.
    TERMINAL = "terminal"


class PermissionKind(enum.StrEnum):
    CHAT = "chat"
    ADMIN = "admin"


class Direction(enum.StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class ActionStatus(enum.StrEnum):
    OK = "ok"
    FAILED = "failed"  # the turn or the delivery failed
    DENIED = "denied"  # a known identity without permission wrote in
    LIMITED = "limited"  # an allowed user over the rate limit or the turn limit


class RequestStatus(enum.StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(255), default=None)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # The user's IANA timezone, e.g. "Europe/Paris": the "daily" and "cron" times of
    # their scheduled tasks are read in it. Null means UTC.
    timezone: Mapped[str | None] = mapped_column(String(64), default=None)
    # Set when the MCP guard detected MCP_GUARD_SUSPEND_AFTER threats in 24 hours; every
    # tool call of the user is then refused until an administrator resumes them.
    tools_suspended: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("0")
    )

    channel_identities: Mapped[list[ChannelIdentity]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class ChannelIdentity(Base):
    __tablename__ = "channel_identities"
    __table_args__ = (UniqueConstraint("channel", "external_id", name="uq_channel_external_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    channel: Mapped[Channel] = mapped_column(_db_enum(Channel), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    # Only meaningful for the email channel — see module docstring.
    raw_address: Mapped[str | None] = mapped_column(EncryptedString, default=None)
    # The agent this identity talks to. None means the user's "default"
    # agent. Set by the user with /agent <name> or by an admin.
    active_agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agents.id", name="fk_channel_identities_active_agent"), default=None
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped[User] = relationship(back_populates="channel_identities")
    permissions: Mapped[list[Permission]] = relationship(
        back_populates="channel_identity", cascade="all, delete-orphan"
    )


class Permission(Base):
    __tablename__ = "permissions"
    __table_args__ = (
        UniqueConstraint("channel_identity_id", "kind", name="uq_channel_identity_kind"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    channel_identity_id: Mapped[int] = mapped_column(
        ForeignKey("channel_identities.id"), nullable=False
    )
    kind: Mapped[PermissionKind] = mapped_column(_db_enum(PermissionKind), nullable=False)
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    channel_identity: Mapped[ChannelIdentity] = relationship(back_populates="permissions")


class Agent(Base):
    """A User-owned, channel-agnostic autonomous agent. Admin-editable
    — see app/admin/service.py — not only self-service by the owning user.
    """

    __tablename__ = "agents"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_agent_user_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # Per-agent configuration. The system prompt is what an administrator wrote to
    # steer this agent and can hold private instructions, so it is encrypted at rest like the
    # other free text (registered in app/admin/rekey.py and app/admin/service.py). The model
    # (null = the engine's default), the memory mode and the tool allow-list are names and
    # switches, not sensitive, and stay in clear so they can be listed and searched.
    system_prompt: Mapped[str | None] = mapped_column(EncryptedString, default=None)
    model: Mapped[str | None] = mapped_column(String(200), default=None)
    memory_mode: Mapped[str] = mapped_column(String(16), default="off", server_default="off")
    tools: Mapped[list] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    # Skills granted to this agent: names of `skills` rows; only these are listed in
    # its prompt and loadable by its load_skill tool.
    skills: Mapped[list] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    # What the agent is for, in one sentence: shown in the user's list of agents. Written
    # by the user through the agent builder, so encrypted like the other free text.
    purpose: Mapped[str | None] = mapped_column(EncryptedString, default=None)


class ActionLog(Base):
    """Per-user, per-agent audit trail."""

    __tablename__ = "action_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"), nullable=False)
    channel: Mapped[Channel] = mapped_column(_db_enum(Channel), nullable=False)
    direction: Mapped[Direction] = mapped_column(_db_enum(Direction), nullable=False)
    # Conversation content — same sensitivity class as ChannelIdentity.raw_address.
    text: Mapped[str] = mapped_column(EncryptedString, nullable=False)
    # ok, failed or denied. Rows written before this column
    # existed are ok.
    status: Mapped[ActionStatus] = mapped_column(
        _db_enum(ActionStatus),
        nullable=False,
        default=ActionStatus.OK,
        server_default=ActionStatus.OK.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    # Telemetry of the turn, on the reply of a turn only: the model that answered, the
    # time from the message to the reply, and the tokens the engine reported for every call of
    # the turn (tool rounds and summaries included). Null on other rows and on older rows.
    model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Prompt tokens the engine took from its cache, and its time reading the rest (ms).
    cached_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prefill_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class AccessRequest(Base):
    """An unrecognized identity asking for access — created instead
    of only silently denying (see app/security/auth.py). Upserted, not
    inserted per message: one pending row per (channel, external_id).
    """

    __tablename__ = "access_requests"
    __table_args__ = (
        UniqueConstraint("channel", "external_id", name="uq_request_channel_external_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    channel: Mapped[Channel] = mapped_column(_db_enum(Channel), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    first_message_text: Mapped[str] = mapped_column(EncryptedString, nullable=False)
    status: Mapped[RequestStatus] = mapped_column(
        _db_enum(RequestStatus), nullable=False, default=RequestStatus.PENDING
    )
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    # Who resolved it: "api" or "console" (one shared API key and no admin
    # identity, see the admin-audit issue). None while pending.
    resolved_by: Mapped[str | None] = mapped_column(String(32), default=None)


class RequestMessage(Base):
    """Every message of a sender who is not a user yet: the access request keeps only
    the first one, and a stranger's later messages were recorded nowhere. Encrypted like any
    message; the most recent REQUEST_MESSAGES_KEEP are kept per request; deleted with it."""

    __tablename__ = "request_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(
        ForeignKey("access_requests.id", ondelete="CASCADE", name="fk_request_messages_request"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    text: Mapped[str] = mapped_column(EncryptedString, nullable=False)


class ApiCall(Base):
    """One request to the Admin API: who (the client's label and address), what
    (method and path, never the body nor the query string, which can hold a search keyword),
    and the result. Written for every request, refused ones included, by
    `app.api.audit.AuditMiddleware`. No foreign keys, like AdminEvent."""

    __tablename__ = "api_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    actor: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    method: Mapped[str] = mapped_column(String(8), nullable=False)
    path: Mapped[str] = mapped_column(String(300), nullable=False)
    status: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)


class AdminAccount(Base):
    """A named administrator of the Admin API: a name, a scope (`read`,
    `operate`, `admin`, `owner`, app/api/scopes.py), a scrypt password hash
    (app/security/passwords.py) and, for the second factor, an encrypted TOTP secret
    and the last time step used (a code is never accepted twice). Disabled, never deleted:
    the admin events keep naming it."""

    __tablename__ = "admin_accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(24), nullable=False, unique=True)
    scope: Mapped[str] = mapped_column(String(8), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    totp_secret: Mapped[str | None] = mapped_column(EncryptedString, default=None)
    totp_last_step: Mapped[int | None] = mapped_column(Integer, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class ApiToken(Base):
    """A bearer token of one administrator for one client: only the SHA-256 of the
    token is stored (the token is shown once, when it is made), with its scope (never above
    the account's), its expiry and its revocation. Looked up by `token_hash`: the token is
    32 random bytes, so its hash cannot be guessed and needs no slow function."""

    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("admin_accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    scope: Mapped[str] = mapped_column(String(8), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class AdminEvent(Base):
    """One action taken by an administrator: who (`api` or `console`,
    there is one shared API key and no admin identity), what, on which
    object. Written by the service functions in the same transaction as the
    change, so the API and the console record identical events.

    No foreign keys on purpose: an event about a deleted user or agent must
    survive it. `details` is encrypted JSON (names, filters, a search
    keyword) and never holds a secret or an email address.
    """

    __tablename__ = "admin_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    actor: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[int | None] = mapped_column(default=None)
    details: Mapped[str | None] = mapped_column(EncryptedString, default=None)
    # The tamper-evident chain (app/admin/audit_chain.py): the previous event's hash,
    # and the SHA-256 of it and of this event's content.
    prev_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    hash: Mapped[str | None] = mapped_column(String(64), default=None)


class RoutingRule(Base):
    """One rule of the model-routing table, evaluated only when a turn's
    agent has no explicit model of its own (Agent.model, decision layer 1).
    `position` orders the list; the first matching rule wins. `match_value` is a
    stringified threshold (`min_length`) or a literal prefix (`command_prefix`),
    both validated by app/admin/routing.py before they ever reach the database.
    """

    __tablename__ = "routing_rules"

    id: Mapped[int] = mapped_column(primary_key=True)
    position: Mapped[int] = mapped_column(nullable=False, index=True)
    match_type: Mapped[str] = mapped_column(String(32), nullable=False)
    match_value: Mapped[str] = mapped_column(String(200), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)


class RoutingConfig(Base):
    """The routing table's default model (decision layer 3) and, per model, the
    context size app/graph.py budgets the conversation history against.
    One row (id=1): admin configuration, not code. LLAMA_CTX_SIZE stays the
    engine-wide cap every router-loaded model is started with; a size here only
    ever narrows a model's own budget below that cap, never widens it.
    """

    __tablename__ = "routing_config"

    id: Mapped[int] = mapped_column(primary_key=True)
    default_model: Mapped[str | None] = mapped_column(String(200), default=None)
    model_ctx_sizes: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
    # Tool budget: at most max_tools MCP tools per turn (0 = no cap), the tool rules
    # ([{"keyword", "tools"}]) that choose them first, and the model of tool turns.
    max_tools: Mapped[int] = mapped_column(default=5, server_default=text("5"))
    tool_rules: Mapped[list] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    tool_model: Mapped[str | None] = mapped_column(String(200), default=None)


class BackupSchedule(Base):
    """The scheduled backup: one row (id=1), set through the Admin API, and what
    the last run did, so the status survives a restart. `last_error` is a short message
    about files, never data. `failing` marks a failure streak: the administrators are
    notified once when it starts, not at every attempt.
    """

    __tablename__ = "backup_schedule"

    id: Mapped[int] = mapped_column(primary_key=True)
    enabled: Mapped[bool] = mapped_column(default=True, server_default=text("1"))
    interval_minutes: Mapped[int] = mapped_column(default=1440, server_default=text("1440"))
    keep: Mapped[int] = mapped_column(default=7, server_default=text("7"))
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    last_success_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    last_error: Mapped[str | None] = mapped_column(String(500), default=None)
    last_files: Mapped[list] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    failing: Mapped[bool] = mapped_column(default=False, server_default=text("0"))


class Skill(Base):
    """A local skill (decision M3): instructions an agent loads on demand. Only the name
    and the description enter an agent's prompt; the body is read by the load_skill tool, and
    only by an agent the skill is granted to. Versioned: every change of the body or the
    description is a new row of `skill_versions`."""

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(String(120), nullable=False)
    body: Mapped[str] = mapped_column(String(20000), nullable=False)
    # Tools the skill expects (informational: granting tools stays per agent).
    tools: Mapped[list] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    # A user may attach this skill to an agent they create themselves; every other skill
    # is granted by an administrator only.
    self_service: Mapped[bool] = mapped_column(default=False, server_default=text("0"))
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source: Mapped[str] = mapped_column(String(200), nullable=False, default="api")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SkillVersion(Base):
    """Every version of a skill: what an agent could load, and when."""

    __tablename__ = "skill_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    skill_id: Mapped[int] = mapped_column(
        ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(String(120), nullable=False)
    body: Mapped[str] = mapped_column(String(20000), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RetentionConfig(Base):
    """Retention of stored data: one row (id=1), set through the Admin API. Each
    period is a number of days, null = kept for ever (the default,). The
    admin events are never deleted by retention (owner). `last_run` is the counts of the
    last real run."""

    __tablename__ = "retention_config"

    id: Mapped[int] = mapped_column(primary_key=True)
    messages_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tool_calls_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    conversations_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The Admin API call trail, `api_calls`.
    api_calls_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    last_run: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class MemoryEntry(Base):
    """A persistent memory entry of one agent of one user: what the agent chose to
    remember, through its memory tools, across conversations and restarts. Scoped to
    (user, agent): an agent is the namespace, and no tool reaches another's entries. The
    title and the content are free text written from conversations, so both are encrypted
    at rest like the other free text (registered for key rotation); searching decrypts
    one agent's entries, bounded by MAX_ENTRIES_PER_AGENT (app/memory.py).
    """

    __tablename__ = "memory_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agents.id", name="fk_memory_entries_agent"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(EncryptedString, nullable=False)
    content: Mapped[str] = mapped_column(EncryptedString, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class McpTransport(enum.StrEnum):
    STDIO = "stdio"
    HTTP = "http"


class McpEgress(enum.StrEnum):
    LOCAL = "local"
    LAN = "lan"
    INTERNET = "internet"


class McpServer(Base):
    """One MCP server an administrator declared: only administrators declare
    servers, never users. `stdio` is restricted
    to a fixed set of vetted built-ins the project ships (`builtin_id`, looked up in
    `app.mcp.builtin.REGISTRY`), never an admin-supplied command — that would be
    arbitrary code execution as a feature. `http` is one exact URL, re-checked by the
    outbound guard on every connection, not only when declared.
    """

    __tablename__ = "mcp_servers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    # Named "protocol", not "transport": the Admin API's own CLI generator
    # already reserves --transport as a global flag (how the CLI reaches the API),
    # unrelated to this field, and the two collided (test_admin_client.py caught it).
    protocol: Mapped[McpTransport] = mapped_column(_db_enum(McpTransport), nullable=False)
    builtin_id: Mapped[str | None] = mapped_column(String(100), default=None)
    url: Mapped[str | None] = mapped_column(String(500), default=None)
    # JSON object, stdio only: extra variables a built-in may read (none needs any yet).
    env_vars: Mapped[str | None] = mapped_column(EncryptedString, default=None)
    egress: Mapped[McpEgress] = mapped_column(_db_enum(McpEgress), default=McpEgress.LOCAL)
    enabled: Mapped[bool] = mapped_column(default=True)
    timeout_seconds: Mapped[int] = mapped_column(default=20)
    concurrency_limit: Mapped[int] = mapped_column(default=2)
    result_max_bytes: Mapped[int] = mapped_column(default=1_000_000)
    # Per-tool switch (Admin UI/API): a tool this server offers but an admin turned off.
    disabled_tools: Mapped[list] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    #. Policy per tool name, "allow" | "confirm" | "deny"; a tool left out gets the
    # default of app.mcp.policy.default_policy (from its annotations, M4).
    tool_policies: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
    #. The tool definitions an administrator approved, by tool name:
    # {"sha256": ..., "definition": {...}}. A live definition whose hash differs, or a
    # tool with no entry, is not offered until approved again (tool poisoning).
    approved_definitions: Mapped[dict] = mapped_column(
        JSON, default=dict, server_default=text("'{}'")
    )
    # M5. A server holding a credential (env_vars) acts with it for every user it is
    # granted to: it is usable only once flagged, and granted only by an owner.
    shared_credentials: Mapped[bool] = mapped_column(default=False, server_default=text("0"))
    #. How long a "confirm" tool waits for the user's answer before it is refused.
    confirm_timeout_seconds: Mapped[int] = mapped_column(default=120, server_default=text("120"))
    #. The administrator's overrides of the derived classes, by tool name:
    # {"tool": {"private": bool, "untrusted": bool, "outbound": bool}} (a class left out keeps
    # the one app.mcp.policy.default_classes derives).
    tool_classes: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class McpGrant(Base):
    """A user reaches an MCP tool only through a grant (default deny). `agent_id`
    None covers every agent of that user; `tool_name` None covers every tool of the
    server. Names, not foreign keys, for the server and the tool, like McpCall: a grant
    names what an administrator allowed, and a server recreated under the same name is
    the one the administrator named.
    """

    __tablename__ = "mcp_grants"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agents.id", name="fk_mcp_grants_agent"), default=None
    )
    server_name: Mapped[str] = mapped_column(String(100), nullable=False)
    tool_name: Mapped[str | None] = mapped_column(String(200), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class McpCall(Base):
    """One tool call, with its confirmation and its definition pinning.
    No foreign keys, like AdminEvent: a call must survive a
    deleted server or agent. `detail` is encrypted and never holds the raw arguments
    or result, only what happened.
    """

    __tablename__ = "mcp_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    agent_id: Mapped[int | None] = mapped_column(default=None)
    server_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    duration_ms: Mapped[int] = mapped_column(nullable=False)
    result_bytes: Mapped[int] = mapped_column(default=0)
    detail: Mapped[str | None] = mapped_column(EncryptedString, default=None)
    # Who asked, what was decided before the call (app.mcp.policy.Decision), and
    # the arguments with secrets redacted (JSON), encrypted like the other free text.
    user_id: Mapped[int | None] = mapped_column(default=None, index=True)
    decision: Mapped[str | None] = mapped_column(String(16), default=None)
    arguments: Mapped[str | None] = mapped_column(EncryptedString, default=None)


class ScheduledTask(Base):
    """A task a user asked to run on a schedule: the prompt is sent to one of the
    user's agents, in a conversation of its own, and the reply is delivered on the channel
    identity the task names. `kind` is "cron", "every" or "daily" (app/tasks.py); the times
    are read in the user's timezone. The prompt is free text, encrypted at rest like the
    other free text (registered for key rotation). `last_error` is a short class name,
    never text from the conversation.
    """

    __tablename__ = "scheduled_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agents.id", name="fk_scheduled_tasks_agent"), nullable=False
    )
    channel_identity_id: Mapped[int] = mapped_column(
        ForeignKey("channel_identities.id", name="fk_scheduled_tasks_identity"), nullable=False
    )
    prompt: Mapped[str] = mapped_column(EncryptedString, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    expr: Mapped[str] = mapped_column(String(100), nullable=False)
    enabled: Mapped[bool] = mapped_column(default=True, server_default=text("1"))
    next_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None, index=True
    )
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    last_status: Mapped[str | None] = mapped_column(String(16), default=None)
    last_error: Mapped[str | None] = mapped_column(String(200), default=None)
    run_count: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    # Standing approval: the MCP tools this task's turns may run without asking (nobody
    # can answer a confirmation in a scheduled run), each bound to the SHA-256 of the definition
    # approved when the user agreed: {"mcp__web__fetch_page": "<sha256>"}. A tool whose approved
    # definition changed since is refused until the user agrees again.
    standing_tools: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class TaskFeedItem(Base):
    """A feed item a scheduled task already delivered: its id (the first 16 hex of the
    SHA-256 of its link, app/feeds.py), so the task's next runs do not offer it again. Deleted
    with the task; at most app.feed_memory.MAX_REMEMBERED per task, the oldest dropped first."""

    __tablename__ = "task_feed_items"
    __table_args__ = (UniqueConstraint("task_id", "item_id", name="uq_task_feed_item"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(
        ForeignKey("scheduled_tasks.id", name="fk_task_feed_items_task"), nullable=False,
        index=True,
    )  # fmt: skip
    item_id: Mapped[str] = mapped_column(String(16), nullable=False)
    delivered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class FeedSource(Base):
    """A feed an administrator lists for the agent builder to propose: a name, its
    https address and the topics it covers. Not a grant: reading any feed still needs the
    feeds tool, its grant and WEB_FETCH_ALLOWED_HOSTS."""

    __tablename__ = "feed_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    topics: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class TaskConfig(Base):
    """The global switch of the scheduled tasks: one row (id=1). While `paused`,
    no task runs; a task that falls due is moved to its next time, not run late."""

    __tablename__ = "task_config"

    id: Mapped[int] = mapped_column(primary_key=True)
    paused: Mapped[bool] = mapped_column(default=False, server_default=text("0"))


class AgentTemplate(Base):
    """A template the agent builder starts from: an administrator's data, never code.
    The builder first picks one from the user's request (or none), then fixes what the template
    fixes and asks only the template's questions. Versioned: every change is a new row of
    `agent_template_versions` holding the whole template."""

    __tablename__ = "agent_templates"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    # What the template is for: the builder picks a template by this text.
    description: Mapped[str] = mapped_column(String(200), nullable=False)
    # Extra instructions for the builder's model when it fills a specification from it.
    guidance: Mapped[str] = mapped_column(String(4000), nullable=False, default="")
    # Added at the end of the agent's own instructions (for example "copy every link exactly").
    agent_instructions: Mapped[str] = mapped_column(String(2000), nullable=False, default="")
    # Fixed memory mode, or null to let the model choose.
    memory_mode: Mapped[str | None] = mapped_column(String(10), nullable=True)
    # Tool names (the part after mcp__<server>__) attached whenever the user may attach them.
    tools: Mapped[list] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    # The agent must run on a schedule.
    needs_schedule: Mapped[bool] = mapped_column(default=False, server_default=text("0"))
    # The questions the builder may ask, by what is missing: "purpose", "schedule",
    # "task_prompt". The builder asks nothing else while it follows the template.
    questions: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
    enabled: Mapped[bool] = mapped_column(default=True, server_default=text("1"))
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source: Mapped[str] = mapped_column(String(200), nullable=False, default="api")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AgentTemplateVersion(Base):
    """Every version of an agent template: the whole template as it was."""

    __tablename__ = "agent_template_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    template_id: Mapped[int] = mapped_column(
        ForeignKey("agent_templates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
