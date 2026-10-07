"""The built-in MCP server "notes": list, read, search and write Markdown notes in the
one folder NOTES_DIR (app/notes.py), never outside it.

Launched over stdio by app.mcp.manager with a cleared environment holding only NOTES_DIR and
NOTES_MAX_BYTES (app.mcp.builtin.BUILTIN_SETTINGS). The three reading tools are read-only and
closed-world (default policy "allow"); `write_note` is destructive (default policy "confirm",
The user says yes before a note is written). A note's text is the owner's own data, but
it can hold anything pasted into it, so the model treats it as data, never as instructions.
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from app.notes import Notes, NotesError

mcp = FastMCP("notes")

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def _run(action):
    try:
        return action(Notes.from_environment())
    except NotesError as exc:
        return f"error: {exc}"


@mcp.tool(annotations=READ)
def list_notes(folder: str = "") -> str:
    """List the Markdown notes of the notes folder, or of one of its subfolders: path, size and
    last change of each."""
    return _run(lambda notes: notes.list(folder))


@mcp.tool(annotations=READ)
def read_note(path: str) -> str:
    """Read one note, given by its path in the notes folder, e.g. "ideas/today.md"."""
    return _run(lambda notes: notes.read(path))


@mcp.tool(annotations=READ)
def search_notes(query: str) -> str:
    """Find the notes whose name or text contains some words: the path of each and the first
    line that matches."""
    return _run(lambda notes: notes.search(query))


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)
)
def write_note(path: str, content: str, append: bool = False) -> str:
    """Create or update a note: save the given text as a Markdown note in the notes folder,
    e.g. path "shopping.md". It replaces the note, or adds the text at its end with
    append=true. No need to list the notes first. The user is asked before it is saved."""
    return _run(lambda notes: notes.write(path, content, append=append))


if __name__ == "__main__":
    mcp.run(transport="stdio")
