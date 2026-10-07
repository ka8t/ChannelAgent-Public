"""Application configuration, loaded from environment variables (.env).

Fails fast on missing required settings instead of falling back to
insecure defaults — a missing ENCRYPTION_KEY must never silently
result in unencrypted storage.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env also carries shell-only variables start.sh
    # reads directly (LLAMA_SERVER_BIN, MODELS_DIR, LLAMA_PORT — used to
    # auto-start a native llama-server, never by this Python process).
    # Without this, adding one of those breaks Settings() with "Extra
    # inputs are not permitted" for a variable the app never touches.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application-layer encryption (see app/security/encryption.py)
    encryption_key: str = Field(..., alias="ENCRYPTION_KEY")

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/channelagent.db",
        alias="DATABASE_URL",
    )

    # Local LLM gateway
    # The engine's bearer key: start.sh gives it to llama-server (LLAMA_API_KEY, never on
    # its command line) and every request of the application sends it. Empty: no key.
    llama_server_api_key: str | None = Field(default=None, alias="LLAMA_SERVER_API_KEY")
    llama_server_url: str = Field(
        default="http://host.docker.internal:8080", alias="LLAMA_SERVER_URL"
    )
    llama_ctx_size: int = Field(default=65536, alias="LLAMA_CTX_SIZE")
    # Whether the model thinks before it answers: auto (the model's default), on, off.
    llm_thinking: str = Field(default="auto", alias="LLM_THINKING")
    # Conversation checkpoints. A separate SQLite file, so Alembic keeps
    # owning only the application's tables and the checkpointer its own.
    # Empty: "checkpoints.db" next to the main SQLite database file.
    checkpoint_db_path: str | None = Field(default=None, alias="CHECKPOINT_DB_PATH")
    # How many pre-migration copies of the database to keep in backups/ next to
    # it. 0 turns the backup off.
    migration_backups_keep: int = Field(default=5, alias="MIGRATION_BACKUPS_KEEP")
    model_file: str | None = Field(default=None, alias="MODEL_FILE")
    # The engine cache of conversations saved on disk (data/slots), at most this many
    # MiB, the oldest files deleted first. 0 turns it off.
    slot_cache_max_mb: int = Field(default=4096, alias="SLOT_CACHE_MAX_MB")

    # Telegram adapter
    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    # Bootstrap only — never consulted by the Auth Node after
    # the first admin user has been seeded from it.
    telegram_allowed_users: str | None = Field(default=None, alias="TELEGRAM_ALLOWED_USERS")

    # Email adapter
    email_imap_host: str | None = Field(default=None, alias="EMAIL_IMAP_HOST")
    email_imap_port: int = Field(default=993, alias="EMAIL_IMAP_PORT")
    email_smtp_host: str | None = Field(default=None, alias="EMAIL_SMTP_HOST")
    email_smtp_port: int = Field(default=465, alias="EMAIL_SMTP_PORT")
    email_username: str | None = Field(default=None, alias="EMAIL_USERNAME")
    email_password: str | None = Field(default=None, alias="EMAIL_PASSWORD")
    # The mailbox is shared with ordinary mail (website contact form,
    # customer questions), so the adapter only touches messages whose
    # subject contains this tag. Empty means "process nothing", never
    # "process everything". See ("Email on a shared
    # mailbox").
    email_trigger_tag: str = Field(default="[agent]", alias="EMAIL_TRIGGER_TAG")
    # Where a handled agent message is filed so it leaves the INBOX humans
    # read. Created on first use. The default follows the OVH/Dovecot
    # layout ("." separator under INBOX); change it for another provider.
    # Empty means "leave handled messages in the INBOX".
    email_agent_folder: str = Field(default="INBOX.Agent", alias="EMAIL_AGENT_FOLDER")

    # Matrix adapter (not yet implemented)
    matrix_homeserver_url: str | None = Field(default=None, alias="MATRIX_HOMESERVER_URL")
    matrix_bot_user_id: str | None = Field(default=None, alias="MATRIX_BOT_USER_ID")
    matrix_bot_access_token: str | None = Field(default=None, alias="MATRIX_BOT_ACCESS_TOKEN")

    # Admin API
    api_server_port: int = Field(default=8700, alias="API_SERVER_PORT")
    # Interface the Admin API binds to. Loopback by default: the API
    # serves decrypted conversations behind one static key over plain HTTP,
    # so reaching it from the network must be a deliberate choice. The
    # Docker image sets 0.0.0.0 *inside* the container, where the host-side
    # exposure is decided by docker-compose's published address instead.
    api_server_host: str = Field(default="127.0.0.1", alias="API_SERVER_HOST")
    # "tls-proxy" allows a non-loopback API_SERVER_HOST outside the container:
    # a TLS proxy is in front. Empty: the application refuses to start on one.
    api_remote: str = Field(default="", alias="API_REMOTE")
    api_server_key: str | None = Field(default=None, alias="API_SERVER_KEY")
    # Names the API answers to: a request with another Host is refused (421),
    # which stops DNS rebinding from a web page. Add the name of a TLS proxy here.
    # Empty means the loopback names.
    allowed_hosts: str = Field(default="localhost,127.0.0.1,::1", alias="ALLOWED_HOSTS")
    # A request body larger than this is refused (413), a request slower than this
    # is cut (504).
    api_max_body_bytes: int = Field(default=1_048_576, alias="API_MAX_BODY_BYTES")
    api_request_timeout_seconds: int = Field(default=60, alias="API_REQUEST_TIMEOUT_SECONDS")

    # Per-user limits (D9: the same for everyone): messages per minute and turns running at
    # once for one user. 0 = no limit.
    rate_limit_messages_per_minute: int = Field(default=20, alias="RATE_LIMIT_MESSAGES_PER_MINUTE")
    rate_limit_concurrent_turns: int = Field(default=1, alias="RATE_LIMIT_CONCURRENT_TURNS")
    # Agents a user may have when creating one themselves (agent builder): creation is
    # refused once the user has this many agents, whoever created them. 0 = users cannot create
    # agents themselves. Administrators are not limited.
    self_service_max_agents: int = Field(default=10, alias="SELF_SERVICE_MAX_AGENTS")
    # The shortest time between two runs of a task a user schedules themselves (/task, the agent
    # builder), in minutes: one engine serves everyone, and a task every minute from one user
    # would keep it busy for all the others. 0 = no floor. The Admin API is not limited.
    task_min_interval_minutes: int = Field(default=15, alias="TASK_MIN_INTERVAL_MINUTES")
    # The application's log, written to this file as well as the terminal (rotated, mode 600).
    # Empty = the terminal only.
    log_file: str = Field(default="logs/channelagent.log", alias="LOG_FILE")
    # The MCP guard: refused tool calls of one user in an hour, and outbound calls of one
    # user in an hour, above which the behaviour is flagged; and how many flags in 24 hours
    # suspend the user's tools (0 = never suspend).
    mcp_guard_refusals_per_hour: int = Field(default=5, alias="MCP_GUARD_REFUSALS_PER_HOUR")
    mcp_guard_outbound_per_hour: int = Field(default=30, alias="MCP_GUARD_OUTBOUND_PER_HOUR")
    mcp_guard_suspend_after: int = Field(default=3, alias="MCP_GUARD_SUSPEND_AFTER")
    # How long one engine request of a scheduled task's turn may take, in seconds. A
    # chat turn keeps 120 s (a user waits); a task has nobody waiting, and a feed digest took
    # 94 to 509 s on the owner's model.
    llm_task_timeout_seconds: int = Field(default=600, alias="LLM_TASK_TIMEOUT_SECONDS")
    # Local skills: the folder whose `<name>/SKILL.md` files `POST /skills/import` reads.
    # Inside data/ so the container sees it through the same volume.
    skills_dir: str = Field(default="data/skills", alias="SKILLS_DIR")

    # Host helper: off by default; where the application reaches it, and the secret
    # that signs every call to it.
    host_helper_enabled: bool = Field(default=False, alias="HOST_HELPER_ENABLED")
    host_helper_port: int = Field(default=8701, alias="HOST_HELPER_PORT")
    host_helper_socket: str = Field(default="", alias="HOST_HELPER_SOCKET")
    host_helper_secret: str | None = Field(default=None, alias="HOST_HELPER_SECRET")
    host_helper_url: str = Field(default="", alias="HOST_HELPER_URL")

    # Model management. The directory of the `.gguf` files (start.sh reads the same
    # variable to start the engine), the hub a model is pulled from, the extra hosts a pull
    # may reach, the size and time limits of one pull, and the token for gated models.
    models_dir: str = Field(default="models", alias="MODELS_DIR")
    # The engine program start.sh runs (asks it for the GPU memory and runs a measurement).
    llama_server_bin: str = Field(
        default="./vendor/llama.cpp/llama-server", alias="LLAMA_SERVER_BIN"
    )
    model_hub_url: str = Field(default="https://huggingface.co", alias="MODEL_HUB_URL")
    model_pull_allowed_hosts: str = Field(default="", alias="MODEL_PULL_ALLOWED_HOSTS")
    model_pull_max_bytes: int = Field(default=64 * 1024**3, alias="MODEL_PULL_MAX_BYTES")
    model_pull_timeout_seconds: int = Field(default=6 * 3600, alias="MODEL_PULL_TIMEOUT_SECONDS")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    # Reading a web page, the built-in MCP server "web". The hosts a page may come from
    # (comma separated, a host covers its subdomains, "*" any public host; empty: none), the
    # size and time limits of one page, and how long a page is kept in the cache.
    web_fetch_allowed_hosts: str = Field(default="", alias="WEB_FETCH_ALLOWED_HOSTS")
    web_fetch_max_bytes: int = Field(default=2 * 1024**2, alias="WEB_FETCH_MAX_BYTES")
    web_fetch_timeout_seconds: int = Field(default=15, alias="WEB_FETCH_TIMEOUT_SECONDS")
    web_fetch_cache_seconds: int = Field(default=600, alias="WEB_FETCH_CACHE_SECONDS")
    # The browser name a page request carries (empty: a current desktop Chrome), and the
    # headless browser for a page refused (403) or without text to a plain request.
    web_fetch_user_agent: str = Field(default="", alias="WEB_FETCH_USER_AGENT")
    web_fetch_browser: bool = Field(default=False, alias="WEB_FETCH_BROWSER")

    # The built-in MCP server "notes": the one folder of Markdown notes it may read and
    # write (an absolute path, for example an Obsidian vault; empty: the tools refuse), and the
    # size cap of one note.
    notes_dir: str = Field(default="", alias="NOTES_DIR")
    notes_max_bytes: int = Field(default=200_000, alias="NOTES_MAX_BYTES")

    # The built-in MCP server "search": the owner's own SearXNG instance (empty: the tool
    # refuses) and how many results a search returns (at most 20).
    searxng_url: str = Field(default="", alias="SEARXNG_URL")
    search_max_results: int = Field(default=8, alias="SEARCH_MAX_RESULTS")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def engine_headers(settings: Settings | None = None) -> dict[str, str]:
    """What every request to the inference engine carries: the bearer key when
    LLAMA_SERVER_API_KEY is set (llama-server started with LLAMA_API_KEY), else nothing."""
    key = (settings or get_settings()).llama_server_api_key
    return {"Authorization": f"Bearer {key}"} if key else {}
