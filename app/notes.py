"""A folder of Markdown notes for the built-in MCP server "notes": list, read, search
and write, inside the one folder NOTES_DIR (for example an Obsidian vault) and nowhere else.

- A note is a relative path ending in `.md` (added when missing). Refused: an absolute path, a
  `..` or hidden part (`.obsidian`, `.git`), a backslash or a control character, and any path
  whose real location (links followed) is outside the folder.
- Sizes are capped: a note read or written holds at most NOTES_MAX_BYTES bytes; a listing shows
  at most MAX_LISTED notes and a search at most MAX_MATCHES, files larger than the cap skipped.
- A write replaces the note or appends to it, through a temporary file and a rename, never
  following a link at the last step. It is the server's only destructive tool: its default
  policy is "confirm".

Every refusal raises NotesError with a message the model can act on.
"""

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_MAX_BYTES = 200_000
MAX_LISTED = 200
MAX_MATCHES = 20
MAX_PATH_CHARS = 300
SNIPPET_CHARS = 160


class NotesError(ValueError):
    """The request is refused; the message says why."""


class Notes:
    def __init__(self, root: str | os.PathLike, max_bytes: int = DEFAULT_MAX_BYTES):
        if not str(root).strip():
            raise NotesError("no notes folder is configured (NOTES_DIR is empty)")
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotesError("the notes folder does not exist")
        self.max_bytes = max_bytes

    @classmethod
    def from_environment(cls, environ=os.environ) -> "Notes":
        return cls(
            environ.get("NOTES_DIR", ""),
            int(environ.get("NOTES_MAX_BYTES") or DEFAULT_MAX_BYTES),
        )

    # --- paths ---

    def _relative_parts(self, relative: str, folder: bool = False) -> list[str]:
        text = (relative or "").strip()
        if len(text) > MAX_PATH_CHARS:
            raise NotesError(f"a path longer than {MAX_PATH_CHARS} characters")
        if any(ord(c) < 32 or c == "\\" for c in text):
            raise NotesError("a path with a backslash or a control character is refused")
        if text.startswith(("/", "~")) or (len(text) > 1 and text[1] == ":"):
            raise NotesError("give a path relative to the notes folder")
        parts = [p for p in text.split("/") if p not in ("", ".")]
        for part in parts:
            if part == ".." or part.startswith("."):
                raise NotesError("a path with '..' or a hidden part is refused")
        if not folder:
            if not parts:
                raise NotesError("give the path of a note, for example ideas/today.md")
            if not parts[-1].lower().endswith(".md"):
                parts[-1] += ".md"
        return parts

    def _inside(self, path: Path) -> Path:
        real = path.resolve()
        if real != self.root and not real.is_relative_to(self.root):
            raise NotesError("the path leads outside the notes folder")
        return real

    def resolve(self, relative: str, folder: bool = False) -> Path:
        """The real location of `relative`, checked to be inside the folder (links followed)."""
        return self._inside(self.root.joinpath(*self._relative_parts(relative, folder)))

    def _show(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _notes_under(self, start: Path):
        for directory, subdirs, files in os.walk(start):
            subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
            for name in sorted(files):
                if name.startswith(".") or not name.lower().endswith(".md"):
                    continue
                path = Path(directory) / name
                try:
                    real = self._inside(path)
                except NotesError:
                    continue  # a link out of the folder is never listed or searched
                if real.is_file():
                    yield path, real

    # --- tools ---

    def list(self, folder: str = "") -> str:
        start = self.resolve(folder, folder=True)
        if not start.is_dir():
            raise NotesError(f"no folder {folder!r} in the notes")
        lines, more = [], False
        for path, real in self._notes_under(start):
            if len(lines) == MAX_LISTED:
                more = True
                break
            stat = real.stat()
            day = datetime.fromtimestamp(stat.st_mtime, UTC).strftime("%Y-%m-%d")
            lines.append(f"{self._show(path)} ({stat.st_size} bytes, {day})")
        if not lines:
            return "no note"
        if more:
            lines.append(f"[only the first {MAX_LISTED} notes are listed]")
        return "\n".join(lines)

    def read(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.is_file():
            raise NotesError(f"no note {self._show(path)}")
        if path.stat().st_size > self.max_bytes:
            raise NotesError(f"the note is larger than {self.max_bytes} bytes")
        return path.read_text(encoding="utf-8", errors="replace")

    def search(self, query: str) -> str:
        needle = (query or "").strip().lower()
        if not needle:
            raise NotesError("the search is empty")
        if len(needle) > 200:
            raise NotesError("a search longer than 200 characters")
        found = []
        for path, real in self._notes_under(self.root):
            if real.stat().st_size > self.max_bytes:
                continue
            name = self._show(path)
            text = real.read_text(encoding="utf-8", errors="replace")
            line = next((ln.strip() for ln in text.splitlines() if needle in ln.lower()), None)
            if line is None and needle not in name.lower():
                continue
            found.append(f"{name}: {line[:SNIPPET_CHARS]}" if line else name)
            if len(found) == MAX_MATCHES:
                found.append(f"[only the first {MAX_MATCHES} matches are shown]")
                break
        return "\n".join(found) if found else f"no note contains {query.strip()!r}"

    def write(self, relative: str, content: str, append: bool = False) -> str:
        parts = self._relative_parts(relative)
        target = self._inside(self.root.joinpath(*parts))
        data = (content or "").encode("utf-8")
        existing = b""
        if target.exists():
            if not target.is_file():
                raise NotesError(f"{self._show(target)} is not a note")
            if append:
                existing = target.read_bytes()
        if len(existing) + len(data) > self.max_bytes:
            raise NotesError(f"the note would be larger than {self.max_bytes} bytes")
        if append and existing and not existing.endswith(b"\n"):
            existing += b"\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        self._inside(target.parent)  # a parent created through a link stays checked
        handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".note-", suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(existing + data)
            # The rename replaces a link at `target` itself, it never writes through it.
            os.replace(temporary, self.root.joinpath(*parts))
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        verb = "appended to" if append and existing else "written"
        return f"{self._show(target)} {verb} ({len(existing) + len(data)} bytes)"
