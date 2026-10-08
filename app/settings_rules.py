"""The rules a value must satisfy to be written to .env.

One place for them: `./start.sh --config KEY=VALUE` runs this file (through the system
python3, before any virtualenv exists, so it uses the standard library only),
and the application imports `api_key_is_acceptable` from here, so the key rule
of the Admin API is the same in both. A value the application would reject is
refused when it is typed, not at the next start.

Messages never contain the value (it may be a secret).

    printf %s "$VALUE" | python3 app/settings_rules.py KEY
    exit 0 accepted, 1 refused (reason on stderr), 2 unknown variable
"""

from __future__ import annotations

import difflib
import ipaddress
import os
import re
import sys
import time
from urllib.parse import urlparse

# --- rules shared with the application ---

MIN_API_KEY_LENGTH = 16
MIN_DISTINCT_CHARACTERS = 6
MAX_REPEAT_PERIOD = 8
PLACEHOLDER_FRAGMENTS = (
    "changeme",
    "change_me",
    "change-me",
    "password",
    "placeholder",
    "your_",
    "your-",
    "example",
    "0123456789",
    "abcdefghij",
)


def _is_repeating(key: str) -> bool:
    """True when the whole key is one short unit repeated (`abab...`)."""
    return any(
        all(key[i] == key[i - period] for i in range(period, len(key)))
        for period in range(1, MAX_REPEAT_PERIOD + 1)
    )


def api_key_is_acceptable(key: str | None) -> bool:
    """A weak bearer key is guessable, and it is the only thing protecting
    decrypted conversations. The API refuses to start with one
    instead of running with it: too short, a repeated character or short
    repeating pattern, fewer than 6 distinct characters, or a known
    placeholder.
    """
    if not key or len(key) < MIN_API_KEY_LENGTH:
        return False
    lowered = key.lower()
    if any(fragment in lowered for fragment in PLACEHOLDER_FRAGMENTS):
        return False
    if len(set(key)) < MIN_DISTINCT_CHARACTERS:
        return False
    return not _is_repeating(key)


# --- value shapes ---

_FERNET = re.compile(r"[A-Za-z0-9_-]{43}=")
_PORT = re.compile(r"[0-9]{1,5}")
_UINT = re.compile(r"[0-9]{1,9}")
_HOSTNAME = re.compile(
    r"(?=.{1,253}$)[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"
)
_TELEGRAM_TOKEN = re.compile(r"[0-9]{5,}:[A-Za-z0-9_-]{20,}")
_TELEGRAM_USERS = re.compile(r"[0-9]+(,[0-9]+)*")
_MATRIX_USER = re.compile(r"@[^:\s]+:[^\s]+")


def _ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


API_REMOTE_TLS_PROXY = "tls-proxy"


def api_exposure_problem(variable: str, address: str, remote: str) -> str | None:
    """Why the Admin API must not listen or be published on `address`, or None.
    Loopback is always allowed; any other address (a wildcard such as 0.0.0.0
    included) makes the API reachable from other machines, which is allowed only behind a TLS
    proxy, declared with API_REMOTE=tls-proxy. `start.sh` checks API_SERVER_HOST natively and
    API_BIND_ADDRESS for Docker; the application checks API_SERVER_HOST outside its container.
    """
    address = (address or "").strip().strip("[]")
    if _is_loopback(address) or (remote or "").strip() == API_REMOTE_TLS_PROXY:
        return None
    return (
        f"{variable}={address or '(empty)'} makes the Admin API reachable from other machines "
        "over plain HTTP. Keep it on 127.0.0.1, or put a TLS proxy in front "
        "(docker-compose.tls.yml) and set API_REMOTE=tls-proxy "
        "(./start.sh --config API_REMOTE=tls-proxy)"
    )


def api_url_problem(value: str) -> str | None:
    """Why `value` cannot be API_URL, the address the clients (start.sh, the command
    line, a UI on another machine) reach the Admin API at, or None. The API, the
    script and the UI may each run on a different machine. Plain http only to this
    machine: across a network the key would travel in clear, so a remote API is
    reached over https (a TLS proxy) or through an SSH tunnel, whose local end is
    loopback. Shared with app.admin.client, so a value --config KEY=VALUE accepts is one the
    client accepts."""
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError:
        return "must be an http:// or https:// URL with a host and an optional port"
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "must be an http:// or https:// URL with a host and an optional port"
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return "must be scheme://host[:port] only, no credentials, query or fragment"
    if parsed.path not in ("", "/"):
        return "must not have a path: the API's routes start at the root"
    if port is not None and not 1 <= port <= 65535:
        return "must use a port from 1 to 65535"
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
        return (
            "must use https:// for another machine (the API key would cross the network "
            "in clear); http:// is accepted for localhost, 127.0.0.1 and ::1 only, "
            "for example the local end of an SSH tunnel"
        )
    return None


# The inference engine may run on another machine than the application. It is
# "local" when LLAMA_SERVER_URL names this machine: a loopback address, localhost, or
# host.docker.internal (a container's name for the host, which a native run reaches on
# localhost). A local engine is the llama-server start.sh reuses or starts on LLAMA_PORT.
# The helper is reached on this machine's loopback, from a container through Docker
# Desktop's host name, or over a Unix socket: never another machine.
HELPER_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "host.docker.internal"})


def helper_url_problem(value: str) -> str | None:
    """Why `value` cannot be HOST_HELPER_URL, or None."""
    if value.startswith("unix://"):
        path = value[len("unix://") :]
        return None if path.startswith("/") else "unix:// must be followed by an absolute path"
    parts = urlparse(value)
    if parts.scheme != "http" or (parts.hostname or "").lower() not in HELPER_HOSTS:
        return (
            "must be http:// on 127.0.0.1, localhost or host.docker.internal, or "
            "unix:///absolute/path: the helper is never reached over the network"
        )
    return None


LOCAL_ENGINE_HOSTS = frozenset({"localhost", "host.docker.internal"})


def engine_is_local(url: str) -> bool:
    if not url.strip():
        return True
    host = urlparse(url.strip()).hostname or ""
    return host.lower() in LOCAL_ENGINE_HOSTS or _is_loopback(host)


def native_engine_problem(url: str) -> str | None:
    """Why a native run (start.sh --native, --admin, --chat) cannot use this engine URL,
    or None. A remote engine receives every prompt: over plain http they would cross
    the network in clear, so it needs https://, or an SSH tunnel whose local end is
    localhost. (A container reaching an engine on its own Docker network by name is
    the Docker mode, not checked here.)"""
    if not url.strip():
        return None
    if not _url(url.strip()):
        return "must be an http:// or https:// URL with a host"
    if engine_is_local(url):
        return None
    if urlparse(url.strip()).scheme != "https":
        return (
            "names another machine over plain http: the prompts would cross the network "
            "in clear. Use https://, or an SSH tunnel and http://localhost:PORT"
        )
    return None


def _meaning(kind: str, value: str) -> str | None:
    """The reason `value` is not a valid `kind`, or None. Empty is handled by
    the caller: it is valid for every kind except the ones that say otherwise.
    """
    if kind == "free":
        return None
    if kind == "fernet_key":
        if not _FERNET.fullmatch(value):
            return (
                "must be a Fernet key (44 characters, url-safe base64 ending in '='). "
                'Generate one with: python3 -c "import base64, os; '
                'print(base64.urlsafe_b64encode(os.urandom(32)).decode())"'
            )
    elif kind == "port":
        if not _PORT.fullmatch(value) or not 1 <= int(value) <= 65535:
            return "must be a whole number from 1 to 65535"
    elif kind == "uint":
        if not _UINT.fullmatch(value):
            return "must be a whole number, 0 or more"
    elif kind == "posint":
        if not _UINT.fullmatch(value) or int(value) < 1:
            return "must be a whole number, 1 or more"
    elif kind == "bytes":
        if not re.fullmatch(r"[0-9]{1,15}", value) or int(value) < 1:
            return "must be a whole number of bytes, 1 or more"
    elif kind == "host":
        if not (_ip(value) or _HOSTNAME.fullmatch(value)):
            return "must be a host name (letters, digits, dots, hyphens) or an IP address"
    elif kind == "ip":
        if not _ip(value):
            return "must be an IP address such as 127.0.0.1"
    elif kind == "http_url":
        if not _url(value):
            return "must be an http:// or https:// URL with a host"
    elif kind == "api_url":
        return api_url_problem(value)
    elif kind == "abs_path":
        if not value.startswith("/") or len(value) > 100:
            return "must be an absolute path of at most 100 characters"
    elif kind == "helper_url":
        return helper_url_problem(value)
    elif kind == "digest_list":
        pinned = re.compile(r"[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}|sha256:[0-9a-f]{64}")
        bad = [item for item in value.split(",") if not pinned.fullmatch(item.strip())]
        if bad:
            return (
                "must be images pinned by digest, comma separated: repo@sha256:<64 hex> or "
                f"sha256:<64 hex> (not {bad[0].strip()!r})"
            )
    elif kind == "database_url":
        if not value.startswith("sqlite+aiosqlite://"):
            return "must start with sqlite+aiosqlite:// (the only database driver installed)"
    elif kind == "telegram_token":
        if not _TELEGRAM_TOKEN.fullmatch(value):
            return "must look like <digits>:<token> as given by BotFather"
    elif kind == "telegram_users":
        if not _TELEGRAM_USERS.fullmatch(value):
            return "must be Telegram user ids (digits) separated by commas"
    elif kind == "matrix_user":
        if not _MATRIX_USER.fullmatch(value):
            return "must look like @name:server"
    elif kind == "api_key":
        if not api_key_is_acceptable(value):
            return (
                f"is too weak: at least {MIN_API_KEY_LENGTH} characters, not a repeated pattern "
                "or a placeholder. Generate one with: openssl rand -hex 32"
            )
    elif kind == "email_tag":
        # Same rule as the email adapter: the tag goes into an IMAP SEARCH.
        if not (value.strip() and value.isascii() and '"' not in value and "\\" not in value):
            return "must be non-empty ASCII without quotes or backslashes"
    elif kind == "email_folder":
        # Same rule as the email adapter: the folder goes into IMAP commands.
        if value.strip() and not (value.isascii() and not any(c in value for c in '"\\*%')):
            return "must be ASCII without quotes, backslashes or the wildcards * and %"
    elif kind == "thinking":
        if value not in ("auto", "on", "off"):
            return 'must be "auto", "on" or "off"'
    elif kind == "bool":
        if value not in ("true", "false"):
            return 'must be "true" or "false"'
    elif kind == "api_remote":
        if value not in ("", API_REMOTE_TLS_PROXY):
            return f'must be empty (loopback only) or "{API_REMOTE_TLS_PROXY}"'
    else:  # pragma: no cover - guarded by the test that lists every kind
        raise ValueError(f"unknown rule kind {kind!r}")
    return None


# One rule per variable of .env.example (a test fails when one is missing).
# "free" means any single-line text. An empty value is accepted everywhere
# except where REQUIRED says otherwise.
RULES: dict[str, str] = {
    "ENCRYPTION_KEY": "fernet_key",
    "DATABASE_URL": "database_url",
    "LLAMA_SERVER_URL": "http_url",
    "LLAMA_CTX_SIZE": "posint",
    "CHECKPOINT_DB_PATH": "free",
    "MIGRATION_BACKUPS_KEEP": "uint",
    "MODEL_FILE": "free",
    "SLOT_CACHE_MAX_MB": "uint",
    "LLAMA_PORT": "port",
    "LLAMA_SERVER_BIN": "free",
    "MODELS_DIR": "free",
    "LLAMA_SERVER_BIN_DIR": "free",
    "LLAMA_THREADS": "posint",
    "LLAMA_ROUTER_MODE": "bool",
    "LLAMA_MODELS_MAX": "posint",
    "TELEGRAM_BOT_TOKEN": "telegram_token",
    "TELEGRAM_ALLOWED_USERS": "telegram_users",
    "EMAIL_IMAP_HOST": "host",
    "EMAIL_IMAP_PORT": "port",
    "EMAIL_SMTP_HOST": "host",
    "EMAIL_SMTP_PORT": "port",
    "EMAIL_USERNAME": "free",
    "EMAIL_PASSWORD": "free",
    "EMAIL_TRIGGER_TAG": "email_tag",
    "EMAIL_AGENT_FOLDER": "email_folder",
    "MATRIX_HOMESERVER_URL": "http_url",
    "MATRIX_BOT_USER_ID": "matrix_user",
    "MATRIX_BOT_ACCESS_TOKEN": "free",
    "API_SERVER_PORT": "port",
    "API_SERVER_HOST": "host",
    "API_BIND_ADDRESS": "ip",
    "API_REMOTE": "api_remote",
    "API_SERVER_KEY": "api_key",
    "API_URL": "api_url",
    "ALLOWED_HOSTS": "free",
    "API_MAX_BODY_BYTES": "posint",
    "API_REQUEST_TIMEOUT_SECONDS": "posint",
    "MCP_SIDECAR_IMAGES": "digest_list",
    "MCP_SIDECAR_FORWARDER_IMAGE": "free",
    "RATE_LIMIT_MESSAGES_PER_MINUTE": "uint",
    "RATE_LIMIT_CONCURRENT_TURNS": "uint",
    "SELF_SERVICE_MAX_AGENTS": "uint",
    "TASK_MIN_INTERVAL_MINUTES": "uint",
    "LOG_FILE": "free",
    "MCP_GUARD_REFUSALS_PER_HOUR": "posint",
    "MCP_GUARD_OUTBOUND_PER_HOUR": "posint",
    "MCP_GUARD_SUSPEND_AFTER": "uint",
    "LLM_TASK_TIMEOUT_SECONDS": "posint",
    "SKILLS_DIR": "free",
    "HOST_HELPER_ENABLED": "bool",
    "HOST_HELPER_PORT": "port",
    "HOST_HELPER_SOCKET": "abs_path",
    "HOST_HELPER_SECRET": "api_key",
    "LLAMA_SERVER_API_KEY": "api_key",
    "LLM_THINKING": "thinking",
    "HOST_HELPER_URL": "helper_url",
    "HOST_HELPER_GID": "uint",
    "MODEL_HUB_URL": "http_url",
    "MODEL_PULL_ALLOWED_HOSTS": "free",
    "MODEL_PULL_MAX_BYTES": "bytes",
    "MODEL_PULL_TIMEOUT_SECONDS": "posint",
    "HF_TOKEN": "free",
    "WEB_FETCH_ALLOWED_HOSTS": "free",
    "WEB_FETCH_MAX_BYTES": "bytes",
    "WEB_FETCH_TIMEOUT_SECONDS": "posint",
    "WEB_FETCH_CACHE_SECONDS": "uint",
    "WEB_FETCH_USER_AGENT": "free",
    "WEB_FETCH_BROWSER": "bool",
    "NOTES_DIR": "abs_path",
    "NOTES_MAX_BYTES": "bytes",
    "SEARXNG_MANAGED": "bool",
    "SEARXNG_PORT": "port",
    "SEARXNG_URL": "http_url",
    "SEARCH_MAX_RESULTS": "posint",
}

# Variables that cannot be empty: they have a default and an empty value would
# not mean "use the default" (posint/uint/port would be read as an invalid
# number, an empty tag matches every message of the shared mailbox).
REQUIRED = frozenset(
    {
        "DATABASE_URL",
        "LLAMA_SERVER_URL",
        "LLAMA_CTX_SIZE",
        "MIGRATION_BACKUPS_KEEP",
        "SLOT_CACHE_MAX_MB",
        "LLAMA_PORT",
        "LLAMA_THREADS",
        "LLAMA_MODELS_MAX",
        "EMAIL_IMAP_PORT",
        "EMAIL_SMTP_PORT",
        "EMAIL_TRIGGER_TAG",
        "API_SERVER_PORT",
        "API_SERVER_HOST",
        "API_BIND_ADDRESS",
        "API_MAX_BODY_BYTES",
        "API_REQUEST_TIMEOUT_SECONDS",
        "HOST_HELPER_ENABLED",
        "HOST_HELPER_PORT",
        "RATE_LIMIT_MESSAGES_PER_MINUTE",
        "RATE_LIMIT_CONCURRENT_TURNS",
        "SELF_SERVICE_MAX_AGENTS",
        "TASK_MIN_INTERVAL_MINUTES",
        "MCP_GUARD_REFUSALS_PER_HOUR",
        "MCP_GUARD_OUTBOUND_PER_HOUR",
        "MCP_GUARD_SUSPEND_AFTER",
        "LLM_TASK_TIMEOUT_SECONDS",
        "SKILLS_DIR",
        "MODEL_HUB_URL",
        "MODEL_PULL_MAX_BYTES",
        "MODEL_PULL_TIMEOUT_SECONDS",
        "WEB_FETCH_MAX_BYTES",
        "WEB_FETCH_TIMEOUT_SECONDS",
        "WEB_FETCH_CACHE_SECONDS",
        "NOTES_MAX_BYTES",
        "SEARCH_MAX_RESULTS",
        "SEARXNG_MANAGED",
        "SEARXNG_PORT",
    }
)

# How a value is written to .env. One line, read the same way by the
# shell (`source .env` in start.sh), docker compose (`env_file` and `${...}`),
# python-dotenv (the application) and start.sh --config. A value made only
# of these characters is written as it is; any other is written between single
# quotes, where the shell and compose take it literally and python-dotenv too.
_SAFE_UNQUOTED = re.compile(r"[A-Za-z0-9_./:@+,=%^\[\]-]*")


def format_value(value: str) -> str:
    """The text written after `KEY=` in .env for `value`."""
    return value if _SAFE_UNQUOTED.fullmatch(value) else f"'{value}'"


def _unrepresentable(value: str) -> str | None:
    """Why `value` cannot be written so that every reader agrees, or None."""
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        return "must be one line, without control characters"
    if "'" in value or "\\" in value:
        return (
            "cannot contain a single quote or a backslash: the shell and the application "
            "would read it differently (put such a value in .env by hand if you are sure)"
        )
    if "${" in value:
        return "cannot contain '${': the application would expand it as a variable reference"
    return None


def meaning_error(key: str, value: str) -> str | None:
    """Why `value` is not valid for `key`, apart from how it would be written."""
    kind = RULES[key]
    if value == "":
        return "must not be empty" if key in REQUIRED else None
    return _meaning(kind, value)


def validate(key: str, value: str) -> str | None:
    """None when `value` may be written for `key`, else the reason."""
    if key not in RULES:
        raise KeyError(key)
    return _unrepresentable(value) or meaning_error(key, value)


# --- reading and writing .env: the one implementation behind `./start.sh --config`,
# `./start.sh --config KEY=VALUE` and the Admin API routes GET and PATCH /config. ---

# Values never shown in clear. The names are kept as `start.sh` always had them.
SENSITIVE = frozenset(
    {
        "ENCRYPTION_KEY",
        "API_SERVER_KEY",
        "TELEGRAM_BOT_TOKEN",
        "EMAIL_PASSWORD",
        "MATRIX_ACCESS_TOKEN",
        "MATRIX_BOT_ACCESS_TOKEN",
        "HF_TOKEN",
        "HOST_HELPER_SECRET",
        "LLAMA_SERVER_API_KEY",
    }
)


class ConfigError(Exception):
    """A change that was refused. `kind` is one of `unknown`, `key_protected`, `invalid`,
    `backup_failed`."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def read_env(path: str) -> dict:
    """KEY -> value of the assignments in `path`, quotes removed. {} if missing."""
    values = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                values[key] = value
    except FileNotFoundError:
        pass
    return values


def example_keys(example_path: str) -> list:
    """The variables the application knows: the ones of .env.example, in its order."""
    return list(read_env(example_path))


def config_entries(env_path: str, example_path: str) -> list:
    """One entry per known variable: key, value (as in .env, "" if unset), is_set, secret."""
    current = read_env(env_path)
    return [
        {
            "key": key,
            "value": current.get(key, ""),
            "is_set": bool(current.get(key, "")),
            "secret": key in SENSITIVE,
        }
        for key in example_keys(example_path)
    ]


def format_config(env_path: str, example_path: str) -> str:
    """The text of `./start.sh --config`. A secret shows its first four characters."""
    lines = ["Current configuration (keys from .env.example, values from .env):\n"]
    for entry in config_entries(env_path, example_path):
        value = entry["value"]
        if entry["secret"]:
            display = f"{value[:4]}...(hidden)" if value else "(not set)"
        else:
            display = value if value else "(not set)"
        lines.append(f"  {entry['key']:<28} {display}")
    return "\n".join(lines)


def backup_env(env_path: str) -> str:
    """Copy .env to `<env>.bak-YYYYMMDD-HHMMSS` (mode 600, never an existing file: `-2`, `-3`
    on a clash) and read the copy back. Every generation is kept; deleting them is the
    owner's choice. Returns the backup's file name; raises ConfigError (`backup_failed`) when
    the copy cannot be made or differs, so the caller writes nothing.
    """
    try:
        with open(env_path, "rb") as f:
            previous = f.read()
        base = f"{env_path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        backup, n = base, 1
        while True:
            try:
                fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                break
            except FileExistsError:
                n += 1
                backup = f"{base}-{n}"
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, previous)
            os.fsync(fd)
        finally:
            os.close(fd)
        with open(backup, "rb") as f:
            copied = f.read()
    except OSError as exc:
        raise ConfigError(
            f".env was NOT changed: the previous .env could not be backed up ({exc.strerror}).",
            "backup_failed",
        ) from None
    if copied != previous:
        raise ConfigError(
            ".env was NOT changed: the backup of the previous .env does not match it.",
            "backup_failed",
        )
    return os.path.basename(backup)


def check_change(
    key: str, value: str, known: list, current_key: str, allow_initial_key: bool = False
) -> None:
    """The checks every write of .env passes, whoever asks: a known variable,
    ENCRYPTION_KEY never replaced (`current_key` is its value in .env), the value rules.
    Raises ConfigError; the message never holds the value.
    """
    if key not in known:
        close = difflib.get_close_matches(key, known, n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ConfigError(
            f"unknown variable '{key}'.{hint} Known variables: {', '.join(known)}", "unknown"
        )
    if key == "ENCRYPTION_KEY":
        if current_key.strip("'\"") != "":
            raise ConfigError(
                "ENCRYPTION_KEY is already set and was NOT changed: replacing it would make every "
                "encrypted value unreadable. Rotate it with ./start.sh --rekey, which re-encrypts "
                "the data (or python -m app.admin.rekey by hand, the same rotation).",
                "key_protected",
            )
        if not allow_initial_key:
            raise ConfigError(
                "ENCRYPTION_KEY cannot be set through this route: only ./start.sh --config on an "
                "empty key, or the rekey job, changes it.",
                "key_protected",
            )
    reason = validate(key, value)
    if reason:
        raise ConfigError(f"{key} was NOT changed: the value {reason}", "invalid")


def _key_in(lines: list) -> str:
    return next(
        (ln.split("=", 1)[1].strip() for ln in lines if ln.startswith("ENCRYPTION_KEY=")), ""
    )


def set_many(
    env_path: str,
    example_path: str,
    changes: dict,
    allow_initial_key: bool = False,
    create: bool = False,
) -> dict:
    """Write every `key: value` of `changes` to .env at once, after `check_change` on each:
    one refusal writes nothing. An existing .env is kept by `backup_env` first, once
    for the whole batch. `create`Writes a new .env, mode 600, from .env.example's
    lines and never over an existing file; nothing to back up then. Returns
    {"backup": name or None}; raises ConfigError, writing nothing.
    """
    known = example_keys(example_path)
    source = example_path if create else env_path
    with open(source) as f:
        lines = f.readlines()
    current_key = "" if create else _key_in(lines)
    for key, value in changes.items():
        check_change(key, value, known, current_key, allow_initial_key)

    if create:
        backup = None
    else:
        backup = backup_env(env_path)

    for key, value in changes.items():
        written = format_value(value)
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={written}\n"
                break
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"{key}={written}\n")
    if create:
        try:
            fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise ConfigError(
                ".env was NOT created: one appeared in the meantime; run the command again.",
                "invalid",
            ) from None
        with os.fdopen(fd, "w") as f:
            os.fchmod(fd, 0o600)
            f.writelines(lines)
    else:
        with open(env_path, "w") as f:
            f.writelines(lines)
    return {"backup": backup}


def set_config(
    env_path: str, example_path: str, key: str, value: str, allow_initial_key: bool = False
) -> dict:
    """Write `key=value` to .env after the same checks whoever asks (`check_change`), the
    previous file kept by `backup_env`. Returns {"backup": name}; raises ConfigError,
    writing nothing. `allow_initial_key` lets `start.sh` put a first key in an empty
    ENCRYPTION_KEY; the API never sets it (only the rekey job changes it).
    """
    return set_many(env_path, example_path, {key: value}, allow_initial_key)


def cli_show(env_path: str, example_path: str) -> int:
    print(format_config(env_path, example_path))
    return 0


def cli_set(env_path: str, example_path: str, key: str, value: str) -> int:
    try:
        result = set_config(env_path, example_path, key, value, allow_initial_key=True)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Previous .env saved as {result['backup']} (mode 600, every earlier backup kept).")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--engine-kind":
        # start.sh: prints "local" or "remote" for LLAMA_SERVER_URL on stdin, or
        # exits 1 with the reason a native run cannot use it.
        url = sys.stdin.read()
        reason = native_engine_problem(url)
        if reason:
            print(f"LLAMA_SERVER_URL {reason}", file=sys.stderr)
            return 1
        print("local" if engine_is_local(url) else "remote")
        return 0
    if len(argv) == 3 and argv[1] == "--exposure":
        # start.sh: exits 1 with the reason when the variable named here (from the
        # environment, as start.sh exports .env) exposes the API without API_REMOTE=tls-proxy.
        variable = argv[2]
        reason = api_exposure_problem(
            variable, os.environ.get(variable, "127.0.0.1"), os.environ.get("API_REMOTE", "")
        )
        if reason:
            print(reason, file=sys.stderr)
            return 1
        return 0
    if len(argv) == 4 and argv[1] == "--show":
        return cli_show(argv[2], argv[3])
    if len(argv) == 6 and argv[1] == "--set":
        return cli_set(argv[2], argv[3], argv[4], argv[5])
    if len(argv) != 2:
        print("usage: printf %s VALUE | python3 app/settings_rules.py KEY", file=sys.stderr)
        return 2
    key = argv[1]
    if key not in RULES:
        print(f"{key} has no value rule", file=sys.stderr)
        return 2
    reason = validate(key, sys.stdin.read())
    if reason:
        print(f"{key} was NOT changed: the value {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
