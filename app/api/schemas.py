"""Request/response models for the Admin API.

raw_address (email) is never returned by any of these — see
ChannelIdentityOut. external_id is safe to return for every channel:
for Telegram/Matrix it's just the platform's own id, and for email
it's a one-way hash (app.security.hashing.hash_email), not the address
itself.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from app.db.models import ActionStatus, Channel, Direction, PermissionKind, RequestStatus


class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str | None = None


class UserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str | None = None
    is_active: bool | None = None
    timezone: str | None = Field(default=None, max_length=64)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    display_name: str | None
    is_active: bool
    timezone: str | None = None
    # The MCP guard suspended this user's tools; `resume-tools` lifts it.
    tools_suspended: bool = False


class ChannelIdentityCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel: Channel
    # Raw as an admin would naturally provide it: a Telegram numeric id,
    # a Matrix user id, or — for channel="email" — the actual address.
    # The route computes external_id (and, for email, encrypts the
    # address into raw_address) from this; callers never construct the
    # stored key themselves.
    identifier: str


class ChannelIdentityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    channel: Channel
    external_id: str
    active_agent_id: int | None = None


class IdentityAgentSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """`agent_id: null` goes back to the user's default agent."""

    agent_id: int | None


class PermissionGrant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: PermissionKind


class PermissionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    kind: PermissionKind


class ActionLogOut(BaseModel):
    """One audit-trail entry. `text` is the decrypted message, so
    this is only ever served behind the API_SERVER_KEY dependency.
    """

    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    agent_id: int
    channel: Channel
    direction: Direction
    status: ActionStatus
    text: str
    created_at: datetime


class AdminEventOut(BaseModel):
    """One administrator action. `details` is decrypted JSON text."""

    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    actor: str
    action: str
    target_type: str
    target_id: int | None
    details: str | None


class ConversationResetOut(BaseModel):
    threads_reset: int


class StorageOut(BaseModel):
    """What the database holds. The file path is left out on purpose:
    the admin needs the size, not the server's directory layout.
    """

    db_size_bytes: int | None
    row_counts: dict[str, int]
    oldest_log_at: datetime | None
    newest_log_at: datetime | None
    undecryptable_rows: int
    undecryptable_by_table: dict[str, int]
    checkpoint_size_bytes: int | None = None
    checkpoint_row_counts: dict[str, int] = {}


MemoryMode = Literal["off", "ondemand", "always", "search"]


class AgentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    # The per-agent configuration; only what is given is set, the rest keeps its default.
    system_prompt: str | None = Field(default=None, max_length=20000)
    model: str | None = Field(default=None, max_length=200)
    memory_mode: MemoryMode = "off"
    tools: list[str] = Field(default_factory=list, max_length=100)


class AgentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Only the fields that are given change."""

    name: str | None = None
    is_active: bool | None = None
    # `null` clears the prompt and the model; leave a setting out to keep it.
    system_prompt: str | None = Field(default=None, max_length=20000)
    model: str | None = Field(default=None, max_length=200)
    memory_mode: MemoryMode | None = None
    tools: list[str] | None = Field(default=None, max_length=100)


class AgentOut(BaseModel):
    """The system prompt is only returned to an administrator: everyone else sees
    `has_system_prompt`. The other settings are names and switches."""

    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    name: str
    is_active: bool
    system_prompt: str | None = None
    has_system_prompt: bool = False
    model: str | None = None
    memory_mode: str = "off"
    tools: list[str] = []
    skills: list[str] = []
    # User text, only returned to an administrator, like the system prompt.
    purpose: str | None = None

class AccessRequestOut(BaseModel):
    """`first_message_text` is the decrypted first message of the unknown
    sender, so this is only served behind the API_SERVER_KEY dependency.
    """

    model_config = ConfigDict(from_attributes=True)
    id: int
    channel: Channel
    external_id: str
    first_message_text: str
    status: RequestStatus
    requested_at: datetime
    resolved_at: datetime | None
    resolved_by: str | None


class WhoAmIOut(BaseModel):
    actor: str
    scope: str
    api_version: str


class ComponentStatus(BaseModel):
    healthy: bool
    seconds_since_success: float


class DatabaseStatus(BaseModel):
    kind: str
    revision: str | None
    size_bytes: int | None


class EngineStatus(BaseModel):
    reachable: bool
    model: str | None = None
    n_ctx: int | None = None
    # From the engine's /slots: how many requests it can serve at once, and how many it
    # is serving now. None when the engine does not publish them.
    slots_total: int | None = None
    slots_busy: int | None = None


class MemoryStatus(BaseModel):
    """The machine's memory; `low` below 10 GiB available."""

    total_bytes: int
    available_bytes: int
    low: bool


class StatusOut(BaseModel):
    api_version: str
    started_at: datetime
    uptime_seconds: int
    components: dict[str, ComponentStatus]
    database: DatabaseStatus
    engine: EngineStatus
    memory: MemoryStatus | None = None


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    kind: str
    status: str
    progress: float | None
    message: str | None
    result: dict | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class BackupOut(BaseModel):
    name: str
    kind: str
    size_bytes: int
    created_at: datetime | None


class ConfigEntryOut(BaseModel):
    """One variable. The value of a secret is never returned: `value` is null and
    `is_set` says whether one is in place.
    """

    key: str
    value: str | None
    is_set: bool
    secret: bool


class ConfigSetIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    value: str


class ConfigSetOut(BaseModel):
    """What changed, never the value."""

    key: str
    changed: bool
    backup: str
    applies: str


class ModelOut(BaseModel):
    name: str
    size_bytes: int
    sha256: str | None
    modified_at: datetime
    loaded: bool
    configured: bool


class RoutingRuleIn(BaseModel):
    """One rule of the routing table: match_value is an integer for
    min_length, a string for command_prefix — validated by app.admin.routing
    against match_type, since the two share no single Pydantic type.
    """

    model_config = ConfigDict(extra="forbid")
    match_type: str
    match_value: int | str
    model: str


class RoutingRuleOut(BaseModel):
    match_type: str
    match_value: str
    model: str


class ToolRule(BaseModel):
    """When `keyword` is in the message (any case), the tools it names are shown first.
    A name is `mcp__<server>__<tool>`; `*` matches any text (`mcp__time__*`)."""

    model_config = ConfigDict(extra="forbid")
    keyword: str = Field(min_length=1, max_length=100)
    tools: list[str] = Field(min_length=1, max_length=50)


class RoutingIn(BaseModel):
    """Replaces the whole routing table (PUT semantics): a field left out clears
    to its empty default rather than staying unchanged, since a partial ordered
    list has no obvious meaning to merge.
    """

    model_config = ConfigDict(extra="forbid")
    default_model: str | None = Field(default=None, max_length=200)
    rules: list[RoutingRuleIn] = Field(default_factory=list, max_length=50)
    model_ctx_sizes: dict[str, int] = Field(default_factory=dict)
    max_tools: int = Field(
        default=5, ge=0, le=100, description="MCP tools shown per turn at most (0 = no cap)"
    )
    tool_rules: list[ToolRule] = Field(default_factory=list, max_length=50)
    tool_model: str | None = Field(
        default=None, max_length=200, description="the model of the turns of agents with tools"
    )


class RoutingOut(BaseModel):
    default_model: str | None
    rules: list[RoutingRuleOut]
    model_ctx_sizes: dict[str, int]
    max_tools: int
    tool_rules: list[ToolRule]
    tool_model: str | None


class McpServerIn(BaseModel):
    """`builtin_id` only applies to `stdio` (one of app.mcp.builtin.REGISTRY, never
    an admin-supplied command); `url` only to `http`. Named `protocol`, not
    `transport`: the generated CLI already reserves `--transport` as a
    global flag (how it reaches the API), unrelated to this field.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(max_length=100)
    protocol: Literal["stdio", "http"]
    builtin_id: str | None = None
    url: str | None = Field(default=None, max_length=500)
    env_vars: dict[str, str] | None = None
    egress: Literal["local", "lan", "internet"] = "local"
    enabled: bool = True
    timeout_seconds: int = 20
    concurrency_limit: int = 2
    result_max_bytes: int = 1_000_000
    tool_policies: dict[str, Literal["allow", "confirm", "deny"]] | None = None
    shared_credentials: bool = False
    confirm_timeout_seconds: int = 120


class McpServerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str | None = Field(default=None, max_length=500)
    env_vars: dict[str, str] | None = None
    egress: Literal["local", "lan", "internet"] | None = None
    enabled: bool | None = None
    timeout_seconds: int | None = None
    concurrency_limit: int | None = None
    result_max_bytes: int | None = None
    disabled_tools: list[str] | None = None
    tool_policies: dict[str, Literal["allow", "confirm", "deny"]] | None = None
    # Owner scope only when it changes (M5).
    shared_credentials: bool | None = None
    confirm_timeout_seconds: int | None = None


class McpServerOut(BaseModel):
    """`has_env_vars` replaces `env_vars`: a secret is write-only, like `/config`."""

    id: int
    name: str
    protocol: str
    builtin_id: str | None
    url: str | None
    has_env_vars: bool
    egress: str
    enabled: bool
    timeout_seconds: int
    concurrency_limit: int
    result_max_bytes: int
    disabled_tools: list[str]
    tool_policies: dict[str, str]
    shared_credentials: bool
    confirm_timeout_seconds: int
    approved_tools: list[str]
    tool_classes: dict[str, dict[str, bool]] = {}  # The administrator's overrides
    created_at: datetime


class McpToolOut(BaseModel):
    """`approval`: `approved` (the live definition is the pinned one), `changed` (it
    differs: `approved_definition` next to `definition` is the diff to review), `new`
    (never approved) or `gone` (approved, no longer offered). Only an approved tool is
    offered to a turn."""

    name: str
    description: str
    enabled: bool
    approval: Literal["approved", "changed", "new", "gone"] = "new"
    sha256: str | None = None
    policy: Literal["allow", "confirm", "deny"] | None = None
    default_policy: Literal["allow", "confirm", "deny"] | None = None
    classes: dict[str, bool] | None = None  # In force (derived, then overridden)
    default_classes: dict[str, bool] | None = None  # Derived from annotations and egress
    definition: dict | None = None
    approved_definition: dict | None = None


class McpServerTestOut(BaseModel):
    reachable: bool
    tools: list[McpToolOut]
    error: str | None = None


class McpToolToggleIn(BaseModel):
    """Either or both: turn the tool on or off, set its policy (`default` goes
    back to the policy its annotations give)."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool | None = None
    policy: Literal["allow", "confirm", "deny", "default"] | None = None
    classes: dict[Literal["private", "untrusted", "outbound"], StrictBool | None] | None = Field(
        default=None,
        description="Override a class (true or false), or null to go back to the derived one",
    )


class ExposureTool(BaseModel):
    name: str
    server: str
    tool: str
    counted: bool
    why: str | None
    private: bool | None
    untrusted: bool | None
    outbound: bool | None
    overridden: list[str]
    # Tasks whose scheduled runs may use this tool without asking, and tasks whose
    # standing approval was given for a definition since replaced (refused until agreed again).
    standing_tasks: list[int] = []
    standing_stale: list[int] = []


class ExposureOut(BaseModel):
    """`all_three` is the dangerous combination: private data access, untrusted content
    and outbound or write reach in one agent."""

    agent_id: int
    memory: bool
    private: bool
    untrusted: bool
    outbound: bool
    all_three: bool
    tools: list[ExposureTool]


class McpApproveIn(BaseModel):
    """`tools` null approves every tool the server offers right now."""

    model_config = ConfigDict(extra="forbid")
    tools: list[str] | None = Field(default=None, max_length=200)


class McpGrantItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server_name: str = Field(max_length=100)
    tool_name: str | None = Field(default=None, max_length=200)
    agent_id: int | None = None


class McpGrantsIn(BaseModel):
    """Replaces every grant of `user_id` (an empty list revokes them all)."""

    model_config = ConfigDict(extra="forbid")
    user_id: int
    grants: list[McpGrantItem] = Field(max_length=500)


class McpBuiltinEnableIn(BaseModel):
    """`agent_id` (optional) gets the server's tools added to its list."""

    model_config = ConfigDict(extra="forbid")
    user_id: int = Field(description="the user who gets the server, for every agent of theirs")
    agent_id: int | None = Field(default=None, description="an agent of that user: its tools list")


class McpMissingSettingOut(BaseModel):
    name: str
    needs: str
    command: str


class McpBuiltinEnableOut(BaseModel):
    server_id: int
    server: str
    created: bool
    enabled: bool
    approved: list[str]
    granted: bool
    agent_id: int | None
    added_tools: list[str]
    missing_setting: McpMissingSettingOut | None


class McpGrantOut(BaseModel):
    id: int
    user_id: int
    agent_id: int | None
    server_name: str
    tool_name: str | None
    created_at: datetime


class McpCallOut(BaseModel):
    """`arguments` are stored with secrets redacted."""

    id: int
    created_at: datetime
    user_id: int | None
    agent_id: int | None
    server_name: str
    tool_name: str
    decision: str | None
    status: str
    arguments: str | None
    duration_ms: int
    result_bytes: int


class ModelImportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    name: str | None = None
    force: bool = False


class ModelPullIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: str
    name: str | None = None
    sha256: str | None = Field(default=None, pattern="^[0-9a-fA-F]{64}$")
    force: bool = False


class BackupScheduleIn(BaseModel):
    """A field left out keeps its value."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool | None = None
    interval_minutes: int | None = Field(default=None, ge=1, le=10080)
    keep: int | None = Field(default=None, ge=1, le=365)


class BackupScheduleOut(BaseModel):
    """The scheduled backup's settings and its last run. `failing` is true from a failed
    run until the next successful one; `last_files` are the copies of the last success."""

    enabled: bool
    interval_minutes: int
    keep: int
    last_run_at: datetime | None
    last_success_at: datetime | None
    last_error: str | None
    last_files: list[str]
    failing: bool
    next_run_at: datetime | None


class HostStatusOut(BaseModel):
    """The application as the host helper sees it: the mode start.sh last used, whether
    it runs and what shows it, and when the helper started."""

    mode: str
    running: bool
    reasons: list[str]
    helper_started_at: datetime


class BackupRestoreIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allow_unreadable: bool = Field(
        default=False,
        description="restore even if some values cannot be decrypted with the current key",
    )


class HostRekeyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allow_unreadable: bool = Field(
        default=False, description="go on although some values are readable with neither key"
    )
    dry_run: bool = Field(default=False, description="count what would change, change nothing")


class HostAuditOut(BaseModel):
    """One line of the host helper's audit: `before` and `after` for every call it ran,
    `refused` for a call it did not."""

    model_config = ConfigDict(extra="allow")
    at: datetime
    call: str
    phase: str
    actor: str | None = None
    op: str | None = None
    result: str | None = None


class TelemetryRowOut(BaseModel):
    """One group of the telemetry of a period. `key` is the user id, the agent id or the
    model name (null: no group, or messages whose model is unknown)."""

    group_by: str
    key: int | str | None
    messages: int
    replies: int
    failed_turns: int
    denied: int
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    prompt_tokens: int
    completion_tokens: int
    # Prompt tokens the engine took from its cache, their share of the prompt tokens of
    # the replies that report it (null: none does), and the mean time reading the rest.
    cached_tokens: int
    cache_reuse_rate: float | None
    prefill_mean_ms: int | None


class RetentionIn(BaseModel):
    """A period left out keeps its value; null keeps that data for ever."""

    model_config = ConfigDict(extra="forbid")
    messages_days: int | None = Field(default=None, description="action_logs older than this")
    tool_calls_days: int | None = Field(default=None, description="mcp_calls older than this")
    api_calls_days: int | None = Field(default=None, description="api_calls older than this")
    conversations_days: int | None = Field(
        default=None, description="conversation histories idle for longer than this"
    )


class RetentionOut(BaseModel):
    messages_days: int | None
    tool_calls_days: int | None
    api_calls_days: int | None
    conversations_days: int | None
    last_run_at: datetime | None
    last_run: dict | None


# A secret field: OpenAPI's `format: password`, so the script client never takes it on its
# command line and the UI and the console mask it (app/admin/manifest.py reads the format).
SECRET_FIELD = {"format": "password", "writeOnly": True}


class ExportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(
        min_length=1, max_length=1024, description="protects the export file",
        json_schema_extra=SECRET_FIELD,
    )  # fmt: skip


class ImportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(max_length=200, description="an export file of backups/ (*.caexport)")
    password: str = Field(
        min_length=1, max_length=1024, description="the export's password",
        json_schema_extra=SECRET_FIELD,
    )  # fmt: skip


class RetentionRunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dry_run: bool = Field(default=True, description="count only (the default)")


class SidecarIn(BaseModel):
    """A third-party MCP server to run in its own container."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,39}$", description="lowercase, digits, -")
    image: str = Field(description="an image of MCP_SIDECAR_IMAGES, pinned by digest")
    port: int = Field(default=8000, ge=1, le=65535, description="its HTTP port in the container")
    egress: Literal["local", "lan", "internet"] = Field(
        default="local", description="local: no way out; lan or internet: may go out"
    )
    memory_mb: int = Field(default=256, ge=32, le=4096)
    cpus: float = Field(default=0.5, ge=0.1, le=4.0)


class SidecarOut(BaseModel):
    name: str
    container: str
    status: str
    ports: str
