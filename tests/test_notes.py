"""Tests for the built-in MCP server "notes": one folder of Markdown notes, no path
leading out of it (.., absolute, hidden, links), sizes capped, a write that asks first."""

import os
import re

import pytest

from app.notes import MAX_LISTED, MAX_MATCHES, Notes, NotesError


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "ideas").mkdir(parents=True)
    (root / "ideas" / "today.md").write_text("# Today\nBuy milk\nCall Bob\n")
    (root / "journal.md").write_text("Rainy day. The milk is gone.\n")
    (root / "ideas" / "draft.txt").write_text("not a note")
    (root / ".obsidian").mkdir()
    (root / ".obsidian" / "secret.md").write_text("plugin settings, milk")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.md").write_text("milk outside the vault")
    return root


def test_listing_shows_the_notes_only(vault):
    listing = Notes(vault).list()
    shown = [line.split(" (")[0] for line in listing.splitlines()]
    assert shown == ["journal.md", "ideas/today.md"]
    assert "draft.txt" not in listing and ".obsidian" not in listing
    assert re.search(r"ideas/today.md \(\d+ bytes, \d{4}-\d{2}-\d{2}\)", listing)
    assert Notes(vault).list("ideas").splitlines()[0].startswith("ideas/today.md")


def test_reading_and_searching(vault):
    notes = Notes(vault)
    assert notes.read("ideas/today.md") == "# Today\nBuy milk\nCall Bob\n"
    assert notes.read("ideas/today") == "# Today\nBuy milk\nCall Bob\n", ".md is added"
    found = notes.search("MILK").splitlines()
    assert sorted(found) == ["ideas/today.md: Buy milk", "journal.md: Rainy day. The milk is gone."]
    assert notes.search("journal") == "journal.md", "a match on the name"
    assert notes.search("nothing like this") == "no note contains 'nothing like this'"


@pytest.mark.parametrize(
    "path, message",
    [
        ("../outside/private.md", "'..'"),
        ("ideas/../../outside/private.md", "'..'"),
        ("/etc/passwd", "relative"),
        ("~/notes.md", "relative"),
        ("C:/notes.md", "relative"),
        (".obsidian/secret.md", "hidden"),
        ("ideas\\today.md", "backslash"),
        ("ideas/to\x00day.md", "control character"),
        ("a/" * 151 + "b.md", "longer than 300"),
        ("", "give the path of a note"),
    ],
)
def test_a_path_leading_out_or_hidden_is_refused_everywhere(vault, path, message):
    notes = Notes(vault)
    for action in (lambda: notes.read(path), lambda: notes.write(path, "x")):
        with pytest.raises(NotesError, match=re.escape(message)):
            action()
    assert (vault.parent / "outside" / "private.md").read_text() == "milk outside the vault"


def test_a_link_leading_out_of_the_folder_is_refused_and_never_listed(vault):
    outside = vault.parent / "outside"
    os.symlink(outside / "private.md", vault / "link.md")
    os.symlink(outside, vault / "elsewhere")
    notes = Notes(vault)
    for path in ("link.md", "elsewhere/private.md", "elsewhere/new.md"):
        with pytest.raises(NotesError, match="outside the notes folder"):
            notes.read(path)
        with pytest.raises(NotesError, match="outside the notes folder"):
            notes.write(path, "overwritten")
    with pytest.raises(NotesError, match="outside the notes folder"):
        notes.list("elsewhere")
    assert "link.md" not in notes.list() and "private" not in notes.list()
    assert "outside the vault" not in notes.search("milk")
    assert (outside / "private.md").read_text() == "milk outside the vault"
    assert not (outside / "new.md").exists()


def test_a_link_inside_the_folder_is_followed(vault):
    os.symlink(vault / "journal.md", vault / "alias.md")
    assert Notes(vault).read("alias.md").startswith("Rainy day")


def test_writing_replaces_or_appends_through_a_rename(vault):
    notes = Notes(vault)
    assert notes.write("new/plan.md", "# Plan\nStep 1") == "new/plan.md written (13 bytes)"
    assert notes.write("new/plan", "Step 2", append=True) == "new/plan.md appended to (20 bytes)"
    assert (vault / "new" / "plan.md").read_text() == "# Plan\nStep 1\nStep 2"
    assert notes.write("new/plan.md", "replaced") == "new/plan.md written (8 bytes)"
    assert (vault / "new" / "plan.md").read_text() == "replaced"
    assert [p.name for p in (vault / "new").iterdir()] == ["plan.md"], "no temporary file left"
    with pytest.raises(NotesError, match="is not a note"):
        (vault / "dir.md").mkdir()
        notes.write("dir.md", "x")


def test_a_write_over_a_link_inside_replaces_the_link_not_its_target(vault):
    os.symlink(vault / "journal.md", vault / "alias.md")
    Notes(vault).write("alias.md", "new text")
    assert not (vault / "alias.md").is_symlink()
    assert (vault / "journal.md").read_text() == "Rainy day. The milk is gone.\n"


def test_sizes_and_counts_are_capped(vault, monkeypatch):
    notes = Notes(vault, max_bytes=50)
    with pytest.raises(NotesError, match="larger than 50 bytes"):
        notes.write("big.md", "x" * 51)
    notes.write("big.md", "x" * 50)
    with pytest.raises(NotesError, match="larger than 50 bytes"):
        notes.write("big.md", "y", append=True)
    (vault / "huge.md").write_text("milk " * 20)
    with pytest.raises(NotesError, match="larger than 50 bytes"):
        notes.read("huge.md")
    assert "huge.md" not in notes.search("milk"), "a file over the cap is not searched"
    with pytest.raises(NotesError, match="longer than 200"):
        notes.search("x" * 201)
    for i in range(MAX_LISTED + 5):
        (vault / f"n{i:03}.md").write_text("milk")
    listing = Notes(vault).list().splitlines()
    assert len(listing) == MAX_LISTED + 1 and listing[-1].startswith("[only the first")
    matches = Notes(vault).search("milk").splitlines()
    assert len(matches) == MAX_MATCHES + 1 and matches[-1].startswith("[only the first")


def test_no_folder_configured_or_missing_is_refused(tmp_path):
    with pytest.raises(NotesError, match="NOTES_DIR is empty"):
        Notes("")
    with pytest.raises(NotesError, match="does not exist"):
        Notes(tmp_path / "missing")
    with pytest.raises(NotesError, match="no note"):
        Notes(tmp_path).read("absent.md")
    with pytest.raises(NotesError, match="no folder"):
        Notes(tmp_path).list("absent")


# --- the server ---


def test_the_notes_server_asks_before_writing_and_gets_only_its_settings(monkeypatch, vault):
    from app.config import get_settings
    from app.mcp import builtin
    from app.mcp.builtin_servers import notes_server

    assert builtin.REGISTRY["notes"] == "app.mcp.builtin_servers.notes_server"
    monkeypatch.setenv("NOTES_DIR", str(vault))
    get_settings.cache_clear()
    try:
        assert builtin.builtin_environment("notes") == {
            "NOTES_DIR": str(vault),
            "NOTES_MAX_BYTES": "200000",
        }
    finally:
        get_settings.cache_clear()
    for name in ("list_notes", "read_note", "search_notes"):
        tool = notes_server.mcp._tool_manager.get_tool(name)
        assert (tool.annotations.readOnlyHint, tool.annotations.openWorldHint) == (True, False)
    write = notes_server.mcp._tool_manager.get_tool("write_note").annotations
    assert (write.readOnlyHint, write.destructiveHint, write.openWorldHint) == (False, True, False)
    refused = notes_server.read_note("../x.md")
    assert refused == "error: a path with '..' or a hidden part is refused"


def test_write_note_gets_the_confirm_policy_and_reading_tools_allow():
    from app.mcp.builtin_servers import notes_server
    from app.mcp.policy import default_policy

    policies = {
        name: default_policy(
            {"annotations": notes_server.mcp._tool_manager.get_tool(name).annotations.model_dump()}
        )
        for name in ("list_notes", "read_note", "search_notes", "write_note")
    }
    assert {name: p.value for name, p in policies.items()} == {
        "list_notes": "allow", "read_note": "allow", "search_notes": "allow",
        "write_note": "confirm",
    }  # fmt: skip


async def test_a_call_round_trips_through_the_real_notes_server(vault):
    from app.db.models import McpTransport
    from app.mcp.manager import ManagedServer, ServerConfig

    server = ManagedServer(
        ServerConfig(
            name="notes", protocol=McpTransport.STDIO, builtin_id="notes", timeout_seconds=20,
            concurrency_limit=1, result_max_bytes=100_000,
        )
    )  # fmt: skip
    os.environ["NOTES_DIR"] = str(vault)
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        names = sorted(t.name for t in await server.list_tools())
        assert names == ["list_notes", "read_note", "search_notes", "write_note"]
        text, is_error = await server.call_tool("read_note", {"path": "ideas/today.md"})
        assert (text, is_error) == ("# Today\nBuy milk\nCall Bob\n", False)
        text, _ = await server.call_tool("read_note", {"path": "../outside/private.md"})
        assert text.startswith("error: a path with '..'")
    finally:
        await server.disconnect()
        del os.environ["NOTES_DIR"]
        get_settings.cache_clear()
