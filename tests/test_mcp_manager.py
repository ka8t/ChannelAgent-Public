"""Tests: the MCP manager — lazy connection, idle stop, restart with
backoff, per-server concurrency and timeout, the per-tool switch. Uses the real
built-in "time" server and two test-only fixtures (tests/mcp_fixtures/) over real
stdio subprocesses rather than a mocked session, since the manager's own job is
managing that real process lifecycle.
"""

import asyncio
import time

import pytest

from app.db.models import McpTransport
from app.mcp.manager import ManagedServer, Manager, McpServerError, ServerConfig


def _config(**overrides) -> ServerConfig:
    base = dict(
        name="time",
        protocol=McpTransport.STDIO,
        builtin_id="time",
        timeout_seconds=20,
        concurrency_limit=2,
        result_max_bytes=1_000_000,
    )
    base.update(overrides)
    return ServerConfig(**base)


@pytest.fixture
async def real_server():
    server = ManagedServer(_config())
    yield server
    await server.disconnect()


async def test_a_stdio_call_round_trips_through_the_real_built_in_server(real_server):
    tools = await real_server.list_tools()
    assert [t.name for t in tools] == ["get_time"]
    text, is_error = await real_server.call_tool("get_time", {"timezone": "UTC"})
    assert is_error is False
    assert "T" in text  # ISO 8601 datetime


async def test_a_result_is_capped_at_result_max_bytes():
    server = ManagedServer(_config(result_max_bytes=5))
    try:
        text, _is_error = await server.call_tool("get_time", {"timezone": "UTC"})
        assert len(text) == 5
    finally:
        await server.disconnect()


async def test_list_tools_still_shows_a_disabled_tool():
    """The admin listing (and the route behind it) needs to see a disabled tool
    to let it be turned back on; app.mcp.catalogue.build_tools is what excludes
    it from what a live turn may call.
    """
    server = ManagedServer(_config(disabled_tools=("get_time",)))
    try:
        assert [t.name for t in await server.list_tools()] == ["get_time"]
    finally:
        await server.disconnect()


async def test_a_disabled_tool_cannot_be_called():
    server = ManagedServer(_config(disabled_tools=("get_time",)))
    try:
        with pytest.raises(McpServerError, match="disabled"):
            await server.call_tool("get_time", {"timezone": "UTC"})
    finally:
        await server.disconnect()


async def test_an_idle_session_is_reconnected_not_reused(real_server, monkeypatch):
    import app.mcp.manager as manager_module

    monkeypatch.setattr(manager_module, "IDLE_TIMEOUT_SECONDS", 0.05)
    await real_server.list_tools()
    first_session = real_server._session
    await asyncio.sleep(0.1)
    await real_server.list_tools()
    assert real_server._session is not first_session


# --- failure isolation: one broken server never blocks another ---


def _slow_config(**overrides) -> ServerConfig:
    base = dict(
        name="slow",
        protocol=McpTransport.STDIO,
        builtin_id="slow",
        timeout_seconds=20,
        concurrency_limit=2,
        result_max_bytes=1_000_000,
    )
    base.update(overrides)
    return ServerConfig(**base)


@pytest.fixture
def slow_registry(monkeypatch):
    """Registers the test-only slow/broken fixtures as if they were vetted
    built-ins, for this test only (never a real production registry entry).
    """
    import app.mcp.manager as manager_module

    monkeypatch.setattr(
        manager_module,
        "BUILTIN_REGISTRY",
        {
            **manager_module.BUILTIN_REGISTRY,
            "slow": "tests.mcp_fixtures.slow_server",
            "broken": "tests.mcp_fixtures.broken_server",
        },
    )


async def test_a_slow_call_is_cut_off_by_the_per_server_timeout(slow_registry):
    server = ManagedServer(_slow_config(timeout_seconds=1))
    try:
        started = time.monotonic()
        with pytest.raises(McpServerError, match="did not answer within"):
            await server.call_tool("sleep", {"seconds": 30})
        elapsed = time.monotonic() - started
        assert elapsed < 5, f"timeout took {elapsed:.1f}s, expected close to 1s"
    finally:
        await server.disconnect()


async def test_the_concurrency_limit_serializes_calls_past_it(slow_registry):
    server = ManagedServer(_slow_config(concurrency_limit=1, timeout_seconds=20))
    try:
        # Connect once first: a cold connect (~0.3s) mixed into the timed section
        # would make a mutant that dropped the semaphore look serialized too.
        await server.call_tool("sleep", {"seconds": 0.01})
        started = time.monotonic()
        await asyncio.gather(
            server.call_tool("sleep", {"seconds": 0.3}),
            server.call_tool("sleep", {"seconds": 0.3}),
        )
        elapsed = time.monotonic() - started
        # Serialized: close to 0.6s. Parallel (the bug) would be close to 0.3s.
        assert elapsed >= 0.55, f"only {elapsed:.2f}s elapsed, calls ran in parallel"
    finally:
        await server.disconnect()


async def test_a_broken_server_fails_without_blocking_a_healthy_one(slow_registry):
    broken = ManagedServer(
        ServerConfig(name="broken", protocol=McpTransport.STDIO, builtin_id="broken")
    )
    healthy = ManagedServer(_config())
    try:
        with pytest.raises(McpServerError):
            await broken.list_tools()
        # The healthy server, independent of the broken one, still answers.
        assert [t.name for t in await healthy.list_tools()] == ["get_time"]
    finally:
        await broken.disconnect()
        await healthy.disconnect()


async def test_a_failed_connection_backs_off_before_retrying(slow_registry, monkeypatch):
    import app.mcp.manager as manager_module

    monkeypatch.setattr(manager_module, "BACKOFF_BASE_SECONDS", 10.0)
    broken = ManagedServer(
        ServerConfig(name="broken", protocol=McpTransport.STDIO, builtin_id="broken")
    )
    try:
        with pytest.raises(McpServerError):
            await broken.list_tools()
        # Immediately retrying is refused with a distinct message, not a second
        # slow connection attempt: the caller (or read of `_next_retry_at`) can
        # tell "backing off" from "just failed to connect" apart.
        with pytest.raises(McpServerError, match="backing off"):
            await broken.list_tools()
    finally:
        await broken.disconnect()


# --- exits / sleeps past the timeout / returns 10 MB, others keep working ---


async def test_a_10mb_result_is_capped_and_answered_within_the_timeout(slow_registry):
    server = ManagedServer(_slow_config(result_max_bytes=1_000_000, timeout_seconds=10))
    try:
        started = time.monotonic()
        text, is_error = await server.call_tool("big", {"bytes": 10_000_000})
        elapsed = time.monotonic() - started
        assert len(text) == 1_000_000
        assert is_error is False
        assert elapsed < 10
    finally:
        await server.disconnect()


async def test_exits_sleeps_past_timeout_and_10mb_none_block_a_healthy_server(slow_registry):
    """The three failure shapes, each against its own
    server, with a fourth healthy server (the real built-in) that must keep
    answering throughout — timings measured, not guessed.
    """
    manager = Manager()
    manager.configure(
        [
            ServerConfig(name="broken", protocol=McpTransport.STDIO, builtin_id="broken"),
            _slow_config(name="sleepy", timeout_seconds=1),
            _slow_config(name="big", timeout_seconds=10, result_max_bytes=1_000_000),
            _config(),  # the real built-in "time" server
        ]
    )
    try:
        started = time.monotonic()
        with pytest.raises(McpServerError):
            await manager.get("broken").list_tools()
        exits_elapsed = time.monotonic() - started

        started = time.monotonic()
        with pytest.raises(McpServerError, match="did not answer within"):
            await manager.get("sleepy").call_tool("sleep", {"seconds": 30})
        sleeps_elapsed = time.monotonic() - started

        started = time.monotonic()
        text, _is_error = await manager.get("big").call_tool("big", {"bytes": 10_000_000})
        big_elapsed = time.monotonic() - started

        # The healthy server, independent of the three failures above, still works.
        healthy_text, healthy_error = await manager.get("time").call_tool(
            "get_time", {"timezone": "UTC"}
        )
        print(
            f"exits: {exits_elapsed:.2f}s, sleeps past timeout: {sleeps_elapsed:.2f}s "
            f"(limit 1s), 10MB capped to {len(text)} bytes in {big_elapsed:.2f}s "
            f"(limit 10s)"
        )
        assert sleeps_elapsed < 5, "the 1s timeout should cut the 30s sleep off promptly"
        assert big_elapsed < 10
        assert len(text) == 1_000_000
        assert healthy_error is False and "T" in healthy_text
    finally:
        await manager.reset()


# --- the Manager: which servers exist, reconfiguration ---


async def test_manager_configure_creates_and_drops_servers():
    manager = Manager()
    manager.configure([_config()])
    assert manager.get("time") is not None
    manager.configure([])
    assert manager.get("time") is None
    await manager.reset()


async def test_manager_configure_keeps_an_unchanged_server_connected(real_server):
    manager = Manager()
    manager.configure([_config()])
    server = manager.get("time")
    await server.list_tools()
    session_before = server._session
    manager.configure([_config()])  # identical config, same object expected
    assert manager.get("time") is server
    assert server._session is session_before
    await manager.reset()
