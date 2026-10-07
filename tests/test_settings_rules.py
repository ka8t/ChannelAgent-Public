"""Tests: `start.sh --set` refuses a value the application would
reject, with the rules in one module (app/settings_rules.py) that the
application also uses for the API key. Nothing is written when a value is
refused, and a message never contains the value.
"""

import ast
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from app import settings_rules as rules

REPO = Path(__file__).resolve().parent.parent
FERNET = Fernet.generate_key().decode()
HEX_KEY = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
TOKEN = "1234567890:ABCdefGhiJklMnoPqrStuVwxYz012345"


def _example_keys() -> list[str]:
    keys = []
    for line in (REPO / ".env.example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.append(line.partition("=")[0])
    return keys


# One accepted value per variable of .env.example.
VALID = {
    "ENCRYPTION_KEY": FERNET,
    "DATABASE_URL": "sqlite+aiosqlite:///./data/channelagent.db",
    "LLAMA_SERVER_URL": "http://host.docker.internal:8080",
    "LLAMA_CTX_SIZE": "65536",
    "CHECKPOINT_DB_PATH": "/var/lib/channelagent/checkpoints.db",
    "MIGRATION_BACKUPS_KEEP": "0",
    "SLOT_CACHE_MAX_MB": "4096",
    "MODEL_FILE": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "LLAMA_PORT": "8081",
    "LLAMA_SERVER_BIN": "/opt/llama/llama-server",
    "MODELS_DIR": "/opt/llama/models",
    "LLAMA_SERVER_BIN_DIR": "/opt/llama/bin",
    "LLAMA_THREADS": "4",
    "LLAMA_ROUTER_MODE": "true",
    "LLAMA_MODELS_MAX": "4",
    "TELEGRAM_BOT_TOKEN": TOKEN,
    "TELEGRAM_ALLOWED_USERS": "111,222",
    "EMAIL_IMAP_HOST": "ssl0.ovh.net",
    "EMAIL_IMAP_PORT": "993",
    "EMAIL_SMTP_HOST": "ssl0.ovh.net",
    "EMAIL_SMTP_PORT": "465",
    "EMAIL_USERNAME": "contact@example.org",
    "EMAIL_PASSWORD": "S3cret-Pass_word.ok",
    "EMAIL_TRIGGER_TAG": "[agent]",
    "EMAIL_AGENT_FOLDER": "INBOX.Agent",
    "MATRIX_HOMESERVER_URL": "https://matrix.org",
    "MATRIX_BOT_USER_ID": "@bot:matrix.org",
    "MATRIX_BOT_ACCESS_TOKEN": "syt_abc123_def456",
    "API_SERVER_PORT": "8700",
    "API_SERVER_HOST": "127.0.0.1",
    "API_BIND_ADDRESS": "0.0.0.0",
    "API_REMOTE": "tls-proxy",
    "ALLOWED_HOSTS": "localhost,127.0.0.1,::1",
    "API_MAX_BODY_BYTES": "1048576",
    "API_REQUEST_TIMEOUT_SECONDS": "60",
    "MODEL_HUB_URL": "https://huggingface.co",
    "MODEL_PULL_ALLOWED_HOSTS": "cdn.example.org,models.example.net",
    "MODEL_PULL_MAX_BYTES": "68719476736",
    "MODEL_PULL_TIMEOUT_SECONDS": "21600",
    "HF_TOKEN": "hf_abcdefghijklmnop",
    "WEB_FETCH_ALLOWED_HOSTS": "en.wikipedia.org,example.org",
    "LLAMA_SERVER_API_KEY": HEX_KEY,
    "LLM_THINKING": "off",
    "WEB_FETCH_MAX_BYTES": "2097152",
    "WEB_FETCH_TIMEOUT_SECONDS": "15",
    "WEB_FETCH_CACHE_SECONDS": "0",
    "WEB_FETCH_USER_AGENT": "MyReader/1.0",
    "WEB_FETCH_BROWSER": "true",
    "NOTES_DIR": "/Users/me/Documents/Vault",
    "NOTES_MAX_BYTES": "200000",
    "SEARXNG_URL": "http://127.0.0.1:8888",
    "SEARCH_MAX_RESULTS": "8",
    "API_SERVER_KEY": HEX_KEY,
    "API_URL": "https://agent.example.org",
    "HOST_HELPER_ENABLED": "true",
    "HOST_HELPER_PORT": "8701",
    "HOST_HELPER_SOCKET": "/run/channelagent-host/helper.sock",
    "HOST_HELPER_SECRET": HEX_KEY,
    "HOST_HELPER_URL": "http://host.docker.internal:8701",
    "HOST_HELPER_GID": "1000",
    "MCP_SIDECAR_IMAGES": "ghcr.io/acme/mcp-x@sha256:" + "a" * 64 + ",sha256:" + "b" * 64,
    "MCP_SIDECAR_FORWARDER_IMAGE": "python:3.14-slim",
    "RATE_LIMIT_MESSAGES_PER_MINUTE": "20",
    "RATE_LIMIT_CONCURRENT_TURNS": "0",
    "SELF_SERVICE_MAX_AGENTS": "10",
    "TASK_MIN_INTERVAL_MINUTES": "15",
    "LOG_FILE": "logs/channelagent.log",
    "MCP_GUARD_REFUSALS_PER_HOUR": "5",
    "MCP_GUARD_OUTBOUND_PER_HOUR": "30",
    "MCP_GUARD_SUSPEND_AFTER": "3",
    "LLM_TASK_TIMEOUT_SECONDS": "600",
    "SKILLS_DIR": "data/skills",
}

# The 7 values measured as accepted with exit 0.
MEASURED = [
    ("LLAMA_PORT", "abc"),
    ("API_SERVER_PORT", "99999999"),
    ("EMAIL_IMAP_PORT", "-5"),
    ("API_SERVER_HOST", "not_an_ip"),
    ("MIGRATION_BACKUPS_KEEP", "lots"),
    ("API_SERVER_KEY", "abc"),
    ("ENCRYPTION_KEY", "notafernetkey"),
]

INVALID = MEASURED + [
    ("LLAMA_PORT", "0"),
    ("LLAMA_PORT", "65536"),
    ("LLAMA_PORT", "+5"),
    ("LLAMA_PORT", "٣"),  # an Arabic-Indic digit
    ("LLAMA_CTX_SIZE", "0"),
    ("LLAMA_CTX_SIZE", "1e3"),
    ("LLAMA_THREADS", "-1"),
    ("LLAMA_MODELS_MAX", "0"),
    ("LLAMA_ROUTER_MODE", "yes"),
    ("LLAMA_ROUTER_MODE", "True"),
    ("MIGRATION_BACKUPS_KEEP", "1234567890"),
    ("MIGRATION_BACKUPS_KEEP", "-1"),
    ("SLOT_CACHE_MAX_MB", "-1"),
    ("API_SERVER_HOST", "-bad.example"),
    ("API_SERVER_HOST", "a..b"),
    ("API_SERVER_HOST", "http://x"),
    ("EMAIL_IMAP_HOST", "mail_server"),
    ("API_BIND_ADDRESS", "localhost"),
    ("API_BIND_ADDRESS", "256.1.1.1"),
    ("LLAMA_SERVER_URL", "host.docker.internal:8080"),
    ("LLAMA_SERVER_URL", "ftp://host"),
    ("LLAMA_SERVER_URL", "http://"),
    ("MATRIX_HOMESERVER_URL", "matrix.org"),
    ("DATABASE_URL", "postgresql://user@host/db"),
    ("DATABASE_URL", "sqlite:///x.db"),
    ("TELEGRAM_BOT_TOKEN", "abc:def"),
    ("TELEGRAM_ALLOWED_USERS", "abc"),
    ("TELEGRAM_ALLOWED_USERS", "111,,222"),
    ("MATRIX_BOT_USER_ID", "bot:matrix.org"),
    ("EMAIL_AGENT_FOLDER", "IN*BOX"),
    ("EMAIL_AGENT_FOLDER", "a%b"),
    ("EMAIL_AGENT_FOLDER", "bôîte"),
    ("EMAIL_TRIGGER_TAG", "é"),
    ("EMAIL_TRIGGER_TAG", ""),
    ("API_MAX_BODY_BYTES", "0"),
    ("API_MAX_BODY_BYTES", ""),
    ("API_REQUEST_TIMEOUT_SECONDS", "-5"),
    ("MODEL_PULL_MAX_BYTES", "0"),
    ("MODEL_PULL_TIMEOUT_SECONDS", "abc"),
    ("WEB_FETCH_MAX_BYTES", "0"),
    ("WEB_FETCH_TIMEOUT_SECONDS", "0"),
    ("WEB_FETCH_CACHE_SECONDS", "-1"),
    ("WEB_FETCH_BROWSER", "yes please"),
    ("LLM_THINKING", "maybe"),
    ("MODEL_HUB_URL", "not a url"),
    # API_URL: plain http only to this machine, no path, no credentials (the key).
    ("API_URL", "http://192.168.1.20:8700"),
    ("API_URL", "http://agent.example.org"),
    ("API_URL", "https://agent.example.org/admin"),
    ("API_URL", "https://user:pw@agent.example.org"),
    ("API_URL", "https://agent.example.org?x=1"),
    ("API_URL", "agent.example.org"),
    ("API_URL", "https://agent.example.org:0"),
    ("API_SERVER_KEY", "a" * 40),
    ("API_SERVER_KEY", "changeme-changeme-changeme"),
    ("ENCRYPTION_KEY", "x" * 44),
    ("EMAIL_USERNAME", "line1\nline2"),
    ("EMAIL_PASSWORD", "tab\there"),
    # The host helper: never reached over the network, never off-loopback.
    ("HOST_HELPER_URL", "http://192.168.1.20:8701"),
    ("HOST_HELPER_URL", "https://127.0.0.1:8701"),
    ("HOST_HELPER_URL", "unix://relative/helper.sock"),
    ("HOST_HELPER_SOCKET", "run/helper.sock"),
    ("HOST_HELPER_SOCKET", "/" + "a" * 120),
    ("HOST_HELPER_SECRET", "changeme-changeme-changeme"),
    ("HOST_HELPER_ENABLED", "yes"),
    ("HOST_HELPER_PORT", ""),
    ("MCP_SIDECAR_IMAGES", "ghcr.io/acme/mcp-x:latest"),
    ("MCP_SIDECAR_IMAGES", "sha256:" + "a" * 63),
    ("RATE_LIMIT_MESSAGES_PER_MINUTE", "-1"),
    ("RATE_LIMIT_MESSAGES_PER_MINUTE", "twenty"),
    ("RATE_LIMIT_CONCURRENT_TURNS", ""),
    ("SELF_SERVICE_MAX_AGENTS", "-1"),
    ("SELF_SERVICE_MAX_AGENTS", ""),
    ("TASK_MIN_INTERVAL_MINUTES", "-1"),
    ("TASK_MIN_INTERVAL_MINUTES", ""),
    ("LLM_TASK_TIMEOUT_SECONDS", "0"),
    ("LLM_TASK_TIMEOUT_SECONDS", ""),
]


# --- the rule table is complete ---


def test_every_variable_of_the_example_has_a_rule_and_no_rule_is_orphaned():
    assert sorted(rules.RULES) == sorted(_example_keys())
    assert set(VALID) == set(rules.RULES), "this test file gives an accepted value for each"


def test_every_rule_kind_is_handled():
    for key in rules.RULES:
        assert rules.meaning_error(key, VALID[key]) is None, key


def test_an_unknown_rule_kind_is_an_error_not_a_silent_accept():
    with pytest.raises(ValueError):
        rules._meaning("no-such-kind", "x")


def test_the_kinds_are_the_ones_the_module_knows():
    known = {
        "free", "fernet_key", "port", "uint", "posint", "bytes", "host", "ip", "http_url",
        "database_url", "telegram_token", "telegram_users", "matrix_user", "api_key",
        "email_tag", "email_folder", "bool", "api_url", "abs_path", "helper_url", "digest_list",
        "thinking", "api_remote",
    }  # fmt: skip
    assert set(rules.RULES.values()) <= known


@pytest.mark.parametrize("key", sorted(VALID))
def test_a_valid_value_is_accepted(key):
    assert rules.validate(key, VALID[key]) is None


@pytest.mark.parametrize(("key", "value"), INVALID)
def test_an_invalid_value_is_refused(key, value):
    assert rules.validate(key, value), f"{key}={value!r} was accepted"


def test_empty_is_accepted_except_where_the_variable_cannot_be_empty():
    for key in rules.RULES:
        reason = rules.validate(key, "")
        if key in rules.REQUIRED:
            assert reason, f"{key} must not be empty"
        else:
            assert reason is None, f"{key}: {reason}"


@pytest.mark.parametrize(
    ("key", "value", "ok"),
    [
        ("LLAMA_PORT", "1", True),
        ("LLAMA_PORT", "65535", True),
        ("LLAMA_PORT", "08080", True),
        ("MIGRATION_BACKUPS_KEEP", "0", True),
        ("LLAMA_CTX_SIZE", "1", True),
        ("API_SERVER_HOST", "::1", True),
        ("API_SERVER_HOST", "localhost", True),
        ("EMAIL_IMAP_HOST", "a-b.c-d.example", True),
        ("API_BIND_ADDRESS", "::", True),
        ("LLAMA_SERVER_URL", "https://x.example:9/", True),
        ("EMAIL_AGENT_FOLDER", "", True),
        ("TELEGRAM_ALLOWED_USERS", "100200300", True),
    ],
)
def test_edges_that_must_stay_valid(key, value, ok):
    assert (rules.validate(key, value) is None) is ok


# --- how values are written ---


@pytest.mark.parametrize("char", list(' &;|<>()$`"!#*?{}~'))
def test_a_special_character_is_accepted_now_that_values_are_quoted(char):
    assert rules.validate("EMAIL_PASSWORD", f"ab{char}cd") is None


@pytest.mark.parametrize("value", ["it's", "back\\slash", "a${x}b"])
def test_a_value_the_readers_would_disagree_on_is_refused(value):
    assert rules.validate("EMAIL_PASSWORD", value)


def test_control_characters_are_refused():
    for value in ("a\nb", "a\rb", "a\x00b", "a\x7fb", "a\tb"):
        assert "control" in rules.validate("EMAIL_PASSWORD", value)


def test_a_message_never_contains_the_value():
    secret = "S3cret'Value$never;shown"
    for key in ("EMAIL_PASSWORD", "API_SERVER_KEY", "ENCRYPTION_KEY"):
        reason = rules.validate(key, secret)
        assert reason and secret not in reason


# --- one source of truth with the application ---


def test_the_api_key_rule_is_the_one_the_api_uses():
    from app.api import deps

    assert deps.api_key_is_acceptable is rules.api_key_is_acceptable


def test_the_email_tag_and_folder_rules_agree_with_the_adapter():
    from app.channels.email import _folder_is_usable, _tag_is_usable

    samples = ["[agent]", "", " ", "a", "é", 'a"b', "a\\b", "INBOX.Agent", "IN*BOX", "a%b", "bô"]
    for value in samples:
        if value != "":
            assert (rules.meaning_error("EMAIL_TRIGGER_TAG", value) is None) == _tag_is_usable(
                value
            ), value
        assert (rules.meaning_error("EMAIL_AGENT_FOLDER", value) is None) == _folder_is_usable(
            value
        ), value
    assert rules.meaning_error("EMAIL_TRIGGER_TAG", "") and not _tag_is_usable("")


@pytest.mark.parametrize("key", sorted(VALID))
def test_a_value_the_rules_accept_is_loaded_by_the_application_settings(key):
    from app.config import Settings

    values = {"ENCRYPTION_KEY": FERNET, key: VALID[key]}
    settings = Settings(_env_file=None, **values)
    assert settings is not None


def test_the_rules_module_runs_on_an_old_system_python():
    """start.sh runs it with whatever `python3` the machine has, before any
    virtualenv exists: standard library only, and syntax an older Python parses.
    """
    source = (REPO / "app" / "settings_rules.py").read_text()
    tree = ast.parse(source, feature_version=(3, 9))
    assert "from __future__ import annotations" in source
    imported = {
        n.module if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [None])
    }
    third_party = {m.split(".")[0] for m in imported if m} - set(sys.stdlib_module_names)
    assert third_party == set(), third_party


# --- the command line entry point ---


def _cli(key: str, value: str):
    return subprocess.run(
        [sys.executable, str(REPO / "app" / "settings_rules.py"), key],
        input=value,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_the_command_line_exit_codes():
    assert _cli("LLAMA_PORT", "8081").returncode == 0
    refused = _cli("LLAMA_PORT", "abc")
    assert refused.returncode == 1 and "LLAMA_PORT was NOT changed" in refused.stderr
    assert _cli("NOT_A_SETTING", "1").returncode == 2
    no_args = subprocess.run(
        [sys.executable, str(REPO / "app" / "settings_rules.py")], capture_output=True, text=True
    )
    assert no_args.returncode == 2


def test_the_command_line_never_prints_the_value():
    for key, value in (("API_SERVER_KEY", "SecretAbc99"), ("LLAMA_PORT", "Zzz999wrong")):
        result = _cli(key, value)
        assert result.returncode == 1
        assert value not in result.stdout + result.stderr


def test_the_command_line_reads_the_value_from_stdin_exactly():
    assert _cli("EMAIL_PASSWORD", "abc\n").returncode == 1  # a newline is not silently stripped
    assert _cli("EMAIL_PASSWORD", "abc").returncode == 0


# --- start.sh --set ---


@pytest.fixture
def sandbox(tmp_path):
    shutil.copy(REPO / "start.sh", tmp_path / "start.sh")
    shutil.copy(REPO / ".env.example", tmp_path / ".env.example")
    (tmp_path / "app").mkdir()
    shutil.copy(REPO / "app" / "settings_rules.py", tmp_path / "app" / "settings_rules.py")
    (tmp_path / ".env").write_text((tmp_path / ".env.example").read_text())
    (tmp_path / ".env").chmod(0o600)
    return tmp_path


def _run(sandbox: Path, *args: str):
    return subprocess.run(
        ["bash", "start.sh", *args], cwd=sandbox, capture_output=True, text=True, timeout=60
    )


def _sha(sandbox: Path) -> str:
    return hashlib.sha256((sandbox / ".env").read_bytes()).hexdigest()


@pytest.mark.parametrize(("key", "value"), MEASURED)
def test_the_seven_measured_values_are_refused_and_nothing_is_written(sandbox, key, value):
    before = _sha(sandbox)
    result = _run(sandbox, "--config", f"{key}={value}")
    assert result.returncode != 0, f"{key}={value} exited 0"
    assert _sha(sandbox) == before
    assert not list(sandbox.glob(".env.bak*"))
    assert f"{key} was NOT changed" in result.stderr
    assert value not in result.stdout + result.stderr


def test_a_valid_value_for_every_variable_is_accepted(sandbox):
    for key in _example_keys():
        result = _run(sandbox, "--config", f"{key}={VALID[key]}")
        assert result.returncode == 0, f"{key}: {result.stderr}"
        assert f"{key}={VALID[key]}\n" in (sandbox / ".env").read_text()


def test_a_value_with_a_shell_metacharacter_is_written_quoted_and_never_executed(sandbox):
    marker = sandbox / "pwned"
    result = _run(sandbox, "--config", f"EMAIL_PASSWORD=ab&touch {marker}")
    assert result.returncode == 0, result.stderr
    assert f"EMAIL_PASSWORD='ab&touch {marker}'\n" in (sandbox / ".env").read_text()
    assert not marker.exists()  # nothing ran when the file was written


def test_if_the_rules_cannot_be_run_nothing_is_written(sandbox):
    (sandbox / "app" / "settings_rules.py").unlink()
    before = _sha(sandbox)
    result = _run(sandbox, "--config", "LLAMA_PORT=8123")
    assert result.returncode != 0
    assert _sha(sandbox) == before and not list(sandbox.glob(".env.bak*"))


def test_the_key_guard_still_comes_first_for_an_existing_key(sandbox):
    assert _run(sandbox, "--config", f"ENCRYPTION_KEY={FERNET}").returncode == 0
    other = Fernet.generate_key().decode()
    before = _sha(sandbox)
    result = _run(sandbox, "--config", f"ENCRYPTION_KEY={other}")
    assert result.returncode != 0 and "rotation" in result.stderr.lower()
    assert _sha(sandbox) == before


# --- the inference engine's machine ---


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("", True),
        ("http://host.docker.internal:8080", True),
        ("http://localhost:8090", True),
        ("http://127.0.0.1:9", True),
        ("http://[::1]:9", True),
        ("https://gpu.example.org", False),
        ("http://192.168.1.30:8080", False),
        ("https://0.0.0.0:8443", False),
    ],
)
def test_the_engine_is_local_only_on_this_machine(url, local):
    assert rules.engine_is_local(url) is local


@pytest.mark.parametrize(
    ("url", "refused"),
    [
        ("http://localhost:8090", False),
        ("https://gpu.example.org", False),
        ("http://192.168.1.30:8080", True),  # prompts in clear across the network
        ("http://llm:8080", True),
        ("not a url", True),
    ],
)
def test_a_native_run_needs_https_for_a_remote_engine(url, refused):
    assert (rules.native_engine_problem(url) is not None) is refused


def test_the_engine_kind_command_line(tmp_path):
    import subprocess

    def kind(url):
        return subprocess.run(
            [sys.executable, str(REPO / "app" / "settings_rules.py"), "--engine-kind"],
            input=url,
            capture_output=True,
            text=True,
        )

    assert (kind("").stdout.strip(), kind("").returncode) == ("local", 0)
    assert kind("https://gpu.example.org").stdout.strip() == "remote"
    refused = kind("http://192.168.1.30:8080")
    assert refused.returncode == 1 and "plain http" in refused.stderr
