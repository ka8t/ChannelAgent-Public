"""Tests: the password-protected export and its import. Round trip with equal row
counts, no plain value in the file, a wrong password, every way of damaging the file (a byte,
two chunks swapped, cut, data appended, the header changed), an archive holding other files,
nothing left behind; the Admin API routes as jobs with the owner scope; the secret field never
taken from a command line, masked in the UI and the console.
"""

import io
import json
import secrets
import sqlite3
import struct
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.admin import export
from app.admin.export import ExportError

SENTINEL = "SENTINEL-" + secrets.token_hex(8)
PASSWORD = "correct horse battery 42"


@pytest.fixture
def fast_kdf(monkeypatch):
    """scrypt at n=2**10 in the tests that are not about the key derivation itself."""
    monkeypatch.setattr(export, "SCRYPT", {"n": 2**10, "r": 8, "p": 1})


def _databases(tmp_path: Path) -> tuple[Path, Path]:
    db, ck = tmp_path / "data" / "channelagent.db", tmp_path / "data" / "checkpoints.db"
    db.parent.mkdir()
    con = sqlite3.connect(db)
    con.execute("create table users (id integer primary key, display_name text)")
    con.executemany("insert into users (display_name) values (?)",
                    [(SENTINEL,), ("Alex",), ("Sam",)])  # fmt: skip
    con.execute("create table notes (id integer primary key, body text)")
    con.executemany("insert into notes (body) values (?)", [(f"note {i}",) for i in range(50)])
    con.commit()
    con.close()
    con = sqlite3.connect(ck)
    con.execute("create table checkpoints (thread_id text, blob blob)")
    con.execute("insert into checkpoints values ('t1', ?)", (b"x" * 5000,))
    con.commit()
    con.close()
    return db, ck


def _export(tmp_path, password=PASSWORD):
    db, ck = _databases(tmp_path)
    result = export.export(db, ck, password, tmp_path / "data" / "backups")
    return db, ck, tmp_path / "data" / "backups" / result["name"], result


# --- round trip ---


def test_export_then_import_keeps_every_row_and_no_plain_value(tmp_path):
    db, ck, file, result = _export(tmp_path)  # the real scrypt parameters
    raw = file.read_bytes()
    assert SENTINEL.encode() not in raw and b"SQLite format 3" not in raw
    assert raw.startswith(export.MAGIC)
    assert oct(file.stat().st_mode & 0o777) == "0o600"
    assert [p.name for p in file.parent.iterdir()] == [file.name], "no plain copy left"
    imported = export.import_export(file, PASSWORD, tmp_path / "data")
    target = Path(imported["directory"])
    assert target.parent == tmp_path / "data" / "imports"
    assert oct(target.stat().st_mode & 0o777) == "0o700"
    assert sorted(p.name for p in target.iterdir()) == ["channelagent.db", "checkpoints.db"]
    assert imported["tables"] == {
        "channelagent.db": export.row_counts(db),
        "checkpoints.db": export.row_counts(ck),
    }
    assert imported["tables"]["channelagent.db"] == {"notes": 50, "users": 3}
    assert result["tables"] == imported["tables"]
    con = sqlite3.connect(target / "channelagent.db")
    assert con.execute("select display_name from users where id = 1").fetchone() == (SENTINEL,)
    con.close()


def test_a_wrong_password_fails_and_leaves_nothing(tmp_path, fast_kdf):
    _db, _ck, file, _ = _export(tmp_path)
    with pytest.raises(ExportError, match="wrong password"):
        export.import_export(file, "not the password at all", tmp_path / "data")
    assert list((tmp_path / "data" / "imports").iterdir()) == []


@pytest.mark.parametrize(
    ("password", "reason"),
    [("short", "at least 12"), ("aaaaaaaaaaaaaaaa", "too simple"), (None, "at least 12")],
)
def test_a_weak_password_is_refused_before_anything_is_written(tmp_path, password, reason):
    db, ck = _databases(tmp_path)
    with pytest.raises(ExportError, match=reason):
        export.export(db, ck, password, tmp_path / "data" / "backups")
    assert not (tmp_path / "data" / "backups").exists() or not any(
        (tmp_path / "data" / "backups").iterdir()
    )


# --- a damaged file ---


def _chunks(file: Path):
    raw = file.read_bytes()
    header_end = raw.index(b"\n", len(export.MAGIC)) + 1
    body, chunks, at = raw[header_end:], [], 0
    while at < len(body):
        (length,) = struct.unpack(">I", body[at : at + 4])
        chunks.append(body[at : at + 4 + length])
        at += 4 + length
    return raw[:header_end], chunks


def _damaged(tmp_path, change) -> str:
    file = next((tmp_path / "data" / "backups").glob("*.caexport"))
    header, chunks = _chunks(file)
    header, chunks = change(header, chunks)
    file.write_bytes(header + b"".join(chunks))
    with pytest.raises(ExportError) as caught:
        export.import_export(file, PASSWORD, tmp_path / "data")
    assert list((tmp_path / "data" / "imports").iterdir()) == []
    return str(caught.value)


@pytest.fixture
def small_chunks(monkeypatch, fast_kdf):
    monkeypatch.setattr(export, "CHUNK", 4096)


def _change_salt(header: bytes) -> bytes:
    """The first two characters of the salt itself. A replace() of those characters changed
    their first occurrence anywhere in the header, sometimes another field (a full run,
    2026-10-04: "the export's header is damaged" instead of the wrong-password message)."""
    at = header.index(b'"salt": "') + 9
    new = b"00" if header[at : at + 2] != b"00" else b"11"
    return header[:at] + new + header[at + 2 :]


def test_every_kind_of_damage_is_refused(tmp_path, small_chunks):
    _export(tmp_path)
    file = next((tmp_path / "data" / "backups").glob("*.caexport"))
    _header, chunks = _chunks(file)
    assert len(chunks) > 3
    original = file.read_bytes()

    def flip(h, c):
        c[2] = c[2][:10] + bytes([c[2][10] ^ 1]) + c[2][11:]
        return h, c

    cases = {
        "a byte changed": flip,
        "two chunks swapped": lambda h, c: (h, [c[0], c[2], c[1], *c[3:]]),
        "the last chunk dropped": lambda h, c: (h, c[:-1]),
        "a chunk appended": lambda h, c: (h, [*c, c[-1]]),
        "a header field changed": lambda h, c: (h.replace(b"AES-256-GCM", b"AES-256-GCN"), c),
        "the salt changed": lambda h, c: (_change_salt(h), c),
    }
    reasons = {}
    for name, change in cases.items():
        file.write_bytes(original)
        reasons[name] = _damaged(tmp_path, change)
    assert reasons == {
        "a byte changed": "the export is damaged",
        "two chunks swapped": "the export is damaged",
        "the last chunk dropped": "the export is cut short",
        "a chunk appended": "the export has data after its end",
        "a header field changed": "wrong password, or the export is damaged",
        "the salt changed": "wrong password, or the export is damaged",
    }, reasons


def test_a_file_that_is_not_an_export_is_refused(tmp_path):
    (tmp_path / "data").mkdir()
    fake = tmp_path / "data" / "fake.caexport"
    fake.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    with pytest.raises(ExportError, match="not a ChannelAgent export"):
        export.import_export(fake, PASSWORD, tmp_path / "data")


def _seal(tar_bytes: bytes, path: Path, password=PASSWORD) -> None:
    """An export made by hand, the format of app/admin/export.py, around any tar."""
    salt, prefix = secrets.token_bytes(16), secrets.token_bytes(8)
    header = export.MAGIC + json.dumps(
        {"kdf": "scrypt", **export.SCRYPT, "salt": salt.hex(), "nonce": prefix.hex(),
         "chunk": export.CHUNK, "cipher": "AES-256-GCM"}, sort_keys=True,
    ).encode() + b"\n"  # fmt: skip
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    aes = AESGCM(export._key(password, salt, export.SCRYPT))
    sealed = aes.encrypt(prefix + struct.pack(">I", 0), tar_bytes, export._aad(header, 0, True))
    path.write_bytes(header + struct.pack(">I", len(sealed)) + sealed)


def _tar(members: dict[str, bytes], symlink: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if symlink:
            info = tarfile.TarInfo(symlink)
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(info)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("members", "symlink"),
    [
        ({"manifest.json": b"{}", "channelagent.db": b"", "../evil": b"x"}, None),
        ({"manifest.json": b"{}", "channelagent.db": b"", "notes.txt": b"x"}, None),
        ({"manifest.json": b"{}"}, "channelagent.db"),
        ({"channelagent.db": b""}, None),
    ],
)
def test_an_archive_with_other_files_is_refused_and_nothing_is_written(
    tmp_path, fast_kdf, members, symlink
):
    (tmp_path / "data").mkdir()
    file = tmp_path / "data" / "crafted.caexport"
    _seal(_tar(members, symlink), file)
    with pytest.raises(ExportError, match="unexpected files"):
        export.import_export(file, PASSWORD, tmp_path / "data")
    assert not (tmp_path / "evil").exists() and not (tmp_path / "data" / "evil").exists()
    assert list((tmp_path / "data" / "imports").iterdir()) == []


def test_a_manifest_that_does_not_match_the_rows_is_refused(tmp_path, fast_kdf):
    db, _ck = _databases(tmp_path)
    manifest = json.dumps({"tables": {"channelagent.db": {"users": 99}}}).encode()
    file = tmp_path / "data" / "lying.caexport"
    _seal(_tar({"manifest.json": manifest, "channelagent.db": db.read_bytes()}), file)
    with pytest.raises(ExportError, match="does not hold the rows its manifest lists"):
        export.import_export(file, PASSWORD, tmp_path / "data")


# --- the Admin API ---


@pytest.fixture
async def api(fresh_db, monkeypatch, fast_kdf):
    import httpx

    from app.admin.jobs import registry
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    key = secrets.token_hex(24)
    monkeypatch.setenv("API_SERVER_KEY", key)
    get_settings.cache_clear()
    deps.reset_failure_state()
    registry.clear()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        c.app = app
        yield c
    app.dependency_overrides.clear()
    deps.reset_failure_state()
    registry.clear()


async def _job(api, response):
    from app.admin.jobs import registry

    assert response.status_code == 202, response.text
    job = registry.get(response.json()["id"])
    await job.task
    return job


async def test_the_api_exports_and_imports_as_jobs_with_admin_events(api):
    from sqlalchemy import select

    from app.admin import service
    from app.db.models import AdminEvent
    from app.db.session import session_scope

    async with session_scope() as session:
        await service.create_user(session, SENTINEL)
        await session.commit()
    exported = await _job(api, await api.post("/backups/export", json={"password": PASSWORD}))
    assert exported.status == "done", exported.error
    name = exported.result["name"]
    assert exported.result["tables"]["channelagent.db"]["users"] == 1
    imported = await _job(
        api, await api.post("/backups/import", json={"name": name, "password": PASSWORD})
    )
    assert imported.status == "done", imported.error
    assert imported.result["tables"] == exported.result["tables"]
    wrong = await _job(
        api, await api.post("/backups/import", json={"name": name, "password": "wrong password 1"})
    )
    assert (wrong.status, wrong.error) == ("failed", "wrong password, or the export is damaged")
    async with session_scope() as session:
        events = (
            (await session.execute(select(AdminEvent).where(AdminEvent.action.like("backup.%"))))
            .scalars()
            .all()
        )
    assert [e.action for e in events] == ["backup.export", "backup.import"]
    assert all(PASSWORD not in json.dumps(e.details) for e in events)


async def test_the_api_refuses_bad_input_before_any_job(api):
    weak = await api.post("/backups/export", json={"password": "short"})
    assert weak.status_code == 422 and "at least 12" in weak.json()["detail"]
    for name, code in (("../../etc/passwd", 422), ("x.caexport", 422),
                       ("channelagent-export-20260101T000000000000Z.caexport", 404)):  # fmt: skip
        answer = await api.post("/backups/import", json={"name": name, "password": PASSWORD})
        assert answer.status_code == code, (name, answer.text)


async def test_export_and_import_need_the_owner_scope(api):
    from app.api.scopes import Principal, Scope, get_principal

    api.app.dependency_overrides[get_principal] = lambda: Principal("admin", Scope.ADMIN)
    assert (await api.post("/backups/export", json={"password": PASSWORD})).status_code == 403
    assert (
        await api.post("/backups/import", json={"name": "x", "password": PASSWORD})
    ).status_code == 403


# --- the secret field in the clients ---


def _export_operation():
    from app.admin.manifest import operations
    from app.api.app import app

    return next(o for o in operations(app) if o["command"] == "export-backup")


def test_the_password_is_a_secret_field_of_the_manifest():
    fields = {f["name"]: f for f in _export_operation()["fields"]}
    assert fields["password"]["secret"] is True


class _Stdin(io.StringIO):
    def __init__(self, text, tty=False):
        super().__init__(text)
        self.tty = tty

    def isatty(self):
        return self.tty


def test_the_script_never_takes_the_password_from_its_command_line():
    from app.admin.client import UsageError, _dest, resolve_secrets

    op = _export_operation()
    dest = _dest("password")
    args = SimpleNamespace(**{dest: PASSWORD})
    with pytest.raises(UsageError, match="is a secret"):
        resolve_secrets(op, args, stdin=_Stdin(""))
    args = SimpleNamespace(**{dest: "-"})
    resolve_secrets(op, args, stdin=_Stdin(PASSWORD + "\n"))
    assert getattr(args, dest) == PASSWORD
    args = SimpleNamespace(**{dest: None})
    resolve_secrets(op, args, stdin=_Stdin(PASSWORD + "\n"))
    assert getattr(args, dest) == PASSWORD, "piped when not a terminal"
    asked = []
    args = SimpleNamespace(**{dest: None})
    resolve_secrets(op, args, stdin=_Stdin("", tty=True), ask=lambda p: asked.append(p) or "typed!")
    assert getattr(args, dest) == "typed!" and asked == ["password: "]
    with pytest.raises(UsageError, match="is empty"):
        resolve_secrets(op, SimpleNamespace(**{dest: "-"}), stdin=_Stdin("\n"))


def test_the_ui_masks_the_password_and_never_fills_it_back():
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("app/ui/templates"), autoescape=True)
    macros = env.get_template("macros.html").module
    field = next(f for f in _export_operation()["fields"] if f["name"] == "password")
    html = str(macros.field(field, PASSWORD))
    assert 'type="password"' in html and PASSWORD not in html


async def test_the_console_asks_the_password_without_echo():
    from app.admin.cli import run_operation

    plain, hidden = [], []

    class _Http:
        async def request(self, *a, **k):
            raise AssertionError("not reached: the confirmation is refused")

    await run_operation(
        _Http(), _export_operation(),
        ask=lambda label: plain.append(label) or "no",
        out=lambda text: None,
        ask_secret=lambda label: hidden.append(label) or PASSWORD,
    )  # fmt: skip
    assert [h.split(" ")[0] for h in hidden] == ["password"]
    assert not any(label.startswith("password") for label in plain)


def test_an_export_copy_ignores_rows_written_while_it_is_made(tmp_path, monkeypatch):
    """The copy is checked against the state it copied (app.db.backup.copy_consistent): the
    API call trail of the job being followed grows during the copy (a full run, 2026-10-04)."""
    import sqlite3

    from app.db import backup

    source, target = tmp_path / "s.db", tmp_path / "t.db"
    con = sqlite3.connect(source)
    con.execute("pragma journal_mode=wal")
    con.execute("create table api_calls (id integer)")
    con.execute("insert into api_calls values (1)")
    con.commit()
    con.close()
    real_counts = backup._row_counts

    def counts_then_write(connection):
        counted = real_counts(connection)
        writer = sqlite3.connect(source)
        writer.execute("insert into api_calls values (2)")
        writer.commit()
        writer.close()
        return counted

    monkeypatch.setattr(backup, "_row_counts", counts_then_write)
    export._copy(source, target)
    assert export.row_counts(target) == {"api_calls": 1}
    assert export.row_counts(source) == {"api_calls": 2}


def test_an_export_copy_that_misses_rows_is_refused(tmp_path, monkeypatch):
    import sqlite3

    from app.db import backup

    source = tmp_path / "s.db"
    con = sqlite3.connect(source)
    con.execute("create table users (id integer)")
    con.commit()
    con.close()
    real_copy = backup.copy_consistent
    monkeypatch.setattr(
        backup, "copy_consistent", lambda s, t: {**real_copy(s, t), "users": 1}
    )
    with pytest.raises(ExportError, match="does not hold the same rows"):
        export._copy(source, tmp_path / "t.db")
