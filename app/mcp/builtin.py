"""Registry of vetted built-in MCP servers: `stdio` transport never runs an
admin-supplied command — that would be arbitrary code execution as a feature
. It only ever launches one of these, by id,
as `sys.executable -m <module>` inside this same virtualenv.
"""

REGISTRY: dict[str, str] = {
    "time": "app.mcp.builtin_servers.time_server",
    "web": "app.mcp.builtin_servers.web_server",
    "feeds": "app.mcp.builtin_servers.feeds_server",
    "calc": "app.mcp.builtin_servers.calc_server",
    "notes": "app.mcp.builtin_servers.notes_server",
    "search": "app.mcp.builtin_servers.search_server",
}

# What a built-in server is given from the application's settings, and nothing else of `.env`:
# the SDK starts it with a cleared environment plus these values.
BUILTIN_SETTINGS: dict[str, tuple[str, ...]] = {
    "web": (
        "WEB_FETCH_ALLOWED_HOSTS",
        "WEB_FETCH_MAX_BYTES",
        "WEB_FETCH_TIMEOUT_SECONDS",
        "WEB_FETCH_CACHE_SECONDS",
        "WEB_FETCH_USER_AGENT",
        "WEB_FETCH_BROWSER",
    ),
    # The same guard and limits as "web" (no browser is used for a feed).
    "feeds": (
        "WEB_FETCH_ALLOWED_HOSTS",
        "WEB_FETCH_MAX_BYTES",
        "WEB_FETCH_TIMEOUT_SECONDS",
        "WEB_FETCH_USER_AGENT",
    ),
    # The one folder of notes and the size cap of a note, nothing else.
    "notes": ("NOTES_DIR", "NOTES_MAX_BYTES"),
    # The owner's SearXNG instance, how many results, and the time limit of a request.
    "search": ("SEARXNG_URL", "SEARCH_MAX_RESULTS", "WEB_FETCH_TIMEOUT_SECONDS"),
}


# The egress label `enable-builtin`Declares each built-in with: what its tools reach.
BUILTIN_EGRESS: dict[str, str] = {
    "time": "local",
    "calc": "local",
    "notes": "local",
    "web": "internet",
    "feeds": "internet",
    "search": "internet",
}

# The setting without which a built-in refuses every call, and what to put there.
BUILTIN_REQUIRED: dict[str, tuple[str, str]] = {
    "web": ("WEB_FETCH_ALLOWED_HOSTS", "the hosts pages may come from, comma separated, or *"),
    "feeds": ("WEB_FETCH_ALLOWED_HOSTS", "the hosts feeds may come from, comma separated, or *"),
    "notes": ("NOTES_DIR", "the absolute path of a folder of Markdown notes"),
    "search": (
        "SEARXNG_URL",
        "the address of a SearXNG instance; SEARXNG_MANAGED=true makes start.sh run one",
    ),
}


def missing_setting(builtin_id: str) -> dict[str, str] | None:
    """The required setting of this built-in when it is empty, with what it needs (never its
    value), or None."""
    from app.config import get_settings

    required = BUILTIN_REQUIRED.get(builtin_id)
    if required is None:
        return None
    name, needs = required
    settings = get_settings()
    field = next(f for f, info in type(settings).model_fields.items() if info.alias == name)
    if str(getattr(settings, field) or "").strip():
        return None
    return {"name": name, "needs": needs, "command": f"./start.sh --config {name}=..."}


def builtin_environment(builtin_id: str | None) -> dict[str, str]:
    """The settings a built-in server receives, read from the application's settings."""
    from app.config import get_settings

    names = BUILTIN_SETTINGS.get(builtin_id or "", ())
    if not names:
        return {}
    settings = get_settings()
    fields = {f.alias: name for name, f in type(settings).model_fields.items() if f.alias}
    env = {n: str(getattr(settings, fields[n])) for n in names}
    for name, value in env.items():
        if isinstance(getattr(settings, fields[name]), bool):
            env[name] = value.lower()
    if builtin_id == "web":
        # The headless browser lives in this repository, never in a user cache.
        from pathlib import Path

        env["PLAYWRIGHT_BROWSERS_PATH"] = str(
            Path(__file__).resolve().parents[2] / "vendor/playwright"
        )
    return env
