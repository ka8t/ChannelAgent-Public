"""Password-protected export and import of the data (gap C4: no relay).

Export: one file holding the database and the conversation checkpoints, for storage outside
the machine. Both are copied with SQLite's online backup API (consistent while the application
writes, verified), packed with a manifest of their row counts in a tar, then encrypted.

    <MAGIC><header JSON>\\n then chunks: <4-byte length><AES-256-GCM ciphertext and tag>

- The key is derived from the password with scrypt (Python's standard library, memory-hard:
  n=2**17, r=8, p=1, about 128 MiB per attempt) and a random salt kept in the header.
- The tar is encrypted in chunks of 1 MiB; each chunk's nonce is a random prefix and its
  number; its associated data is the header, its number and whether it is the last one, so a
  changed header, a reordered, dropped or appended chunk, or a cut file is refused.
- The plain tar exists only as a private file (0600) next to the export while it is built and
  is removed in every case.

Import: into a new, empty directory (`data/imports/import-<stamp>/`), never over the live data.
Every chunk is checked before anything is written out; the tar may hold only the expected
regular files; each database must pass the integrity check and hold the row counts of the
manifest. A wrong password fails on the first chunk. Putting the imported data in place is the
restore's job (`start.sh --restore`), with the application stopped.
"""

import hashlib
import io
import json
import os
import secrets
import shutil
import sqlite3
import struct
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"CHANNELAGENT-EXPORT-1\n"
SUFFIX = ".caexport"
CHUNK = 1024 * 1024
SCRYPT = {"n": 2**17, "r": 8, "p": 1}
SCRYPT_MAXMEM = 256 * 1024 * 1024
MIN_PASSWORD_LENGTH = 12
DATABASE, CHECKPOINTS, MANIFEST = "channelagent.db", "checkpoints.db", "manifest.json"
IMPORTS_DIR = "imports"
MAX_HEADER = 4096


class ExportError(Exception):
    """A refusal or a failure whose message is safe to show (never the password)."""


def check_password(password) -> str:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise ExportError(f"the password needs at least {MIN_PASSWORD_LENGTH} characters")
    if len(set(password)) < 6:
        raise ExportError("the password is too simple: use at least 6 different characters")
    return password


def _key(password: str, salt: bytes, params: dict) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=params["n"], r=params["r"], p=params["p"],
        maxmem=SCRYPT_MAXMEM, dklen=32,
    )  # fmt: skip


def _aad(header: bytes, number: int, last: bool) -> bytes:
    return header + struct.pack(">I?", number, last)


def row_counts(path: Path) -> dict[str, int]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "select name from sqlite_master where type = 'table' order by name"
            )
        ]
        return {t: con.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables}
    finally:
        con.close()


def _copy(source: Path, target: Path) -> None:
    os.close(os.open(target, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600))
    from app.db.backup import copy_consistent

    # Compared with the counts of the copied state, not with the live source afterwards: the
    # API call trail of the job being followed grew during the copy (a full run, 2026-10-04).
    if copy_consistent(source, target) != row_counts(target):
        raise ExportError(f"the copy of {source.name} does not hold the same rows")


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def export(database: Path, checkpoints: Path | None, password: str, directory: Path) -> dict:
    """Write `<directory>/channelagent-export-<stamp>.caexport`; returns its name, size and
    the row counts it holds."""
    check_password(password)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    _private_dir(directory)
    work = directory / f".export-{stamp}"
    work.mkdir(mode=0o700)
    target = directory / f"channelagent-export-{stamp}{SUFFIX}"
    try:
        files = {DATABASE: work / DATABASE}
        _copy(database, files[DATABASE])
        if checkpoints is not None and checkpoints.exists() and checkpoints.stat().st_size:
            files[CHECKPOINTS] = work / CHECKPOINTS
            _copy(checkpoints, files[CHECKPOINTS])
        manifest = {
            "created_at": stamp,
            "tables": {name: row_counts(path) for name, path in files.items()},
        }
        plain = work / "export.tar"
        os.close(os.open(plain, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600))
        with tarfile.open(plain, "w") as tar:
            data = json.dumps(manifest, indent=1).encode()
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            for name, path in files.items():
                tar.add(path, arcname=name)
        salt, prefix = secrets.token_bytes(16), secrets.token_bytes(8)
        header = (
            MAGIC
            + json.dumps(
                {
                    "kdf": "scrypt",
                    **SCRYPT,
                    "salt": salt.hex(),
                    "nonce": prefix.hex(),
                    "chunk": CHUNK,
                    "cipher": "AES-256-GCM",
                },  # fmt: skip
                sort_keys=True,
            ).encode()
            + b"\n"
        )
        aes = AESGCM(_key(password, salt, SCRYPT))
        os.close(os.open(target, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600))
        size = plain.stat().st_size
        with plain.open("rb") as source, target.open("wb") as out:
            out.write(header)
            number, done = 0, 0
            while True:
                block = source.read(CHUNK)
                done += len(block)
                last = done >= size
                sealed = aes.encrypt(prefix + struct.pack(">I", number), block,
                                     _aad(header, number, last))  # fmt: skip
                out.write(struct.pack(">I", len(sealed)) + sealed)
                number += 1
                if last:
                    break
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"name": target.name, "size_bytes": target.stat().st_size, "tables": manifest["tables"]}


def _decrypt(source: Path, password: str, out) -> None:
    with source.open("rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ExportError("not a ChannelAgent export")
        line = f.readline(MAX_HEADER)
        if not line.endswith(b"\n"):
            raise ExportError("the export's header is damaged")
        header = MAGIC + line
        try:
            params = json.loads(line)
            salt, prefix = bytes.fromhex(params["salt"]), bytes.fromhex(params["nonce"])
            kdf = {k: int(params[k]) for k in ("n", "r", "p")}
        except (ValueError, KeyError, TypeError):
            raise ExportError("the export's header is damaged") from None
        if kdf != SCRYPT or params.get("kdf") != "scrypt" or len(prefix) != 8:
            raise ExportError("the export uses parameters this version does not read")
        aes = AESGCM(_key(password, salt, kdf))
        number = 0
        while True:
            raw = f.read(4)
            if len(raw) < 4:
                raise ExportError("the export is cut short")
            (length,) = struct.unpack(">I", raw)
            if length > CHUNK + 16:
                raise ExportError("the export is damaged")
            sealed = f.read(length)
            nonce = prefix + struct.pack(">I", number)
            for last in (False, True):
                try:
                    block = aes.decrypt(nonce, sealed, _aad(header, number, last))
                    break
                except InvalidTag:
                    block = None
            if block is None:
                raise ExportError(
                    "wrong password, or the export is damaged"
                    if number == 0
                    else "the export is damaged"
                )
            out.write(block)
            number += 1
            if last:
                if f.read(1):
                    raise ExportError("the export has data after its end")
                return


def import_export(source: Path, password: str, data_dir: Path) -> dict:
    """Decrypt `source` into a new `data_dir/imports/import-<stamp>/`; returns the directory
    and the row counts, checked against the manifest. Nothing is left behind on a failure."""
    if not isinstance(password, str) or not password:
        raise ExportError("the password is required")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = _private_dir(data_dir / IMPORTS_DIR) / f"import-{stamp}"
    target.mkdir(mode=0o700)
    try:
        plain = target / ".import.tar"
        os.close(os.open(plain, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600))
        with plain.open("wb") as out:
            _decrypt(source, password, out)
        with tarfile.open(plain, "r") as tar:
            members = tar.getmembers()
            names = [m.name for m in members]
            allowed = {MANIFEST, DATABASE, CHECKPOINTS}
            if (
                len(set(names)) != len(names)
                or not set(names) <= allowed
                or {MANIFEST, DATABASE} - set(names)
                or not all(m.isfile() for m in members)
            ):
                raise ExportError("the export holds unexpected files")
            manifest = json.loads(tar.extractfile(MANIFEST).read())
            if set(manifest.get("tables", {})) != set(names) - {MANIFEST}:
                raise ExportError("the export's manifest does not match its files")
            for member in members:
                if member.name == MANIFEST:
                    continue
                path = target / member.name
                os.close(os.open(path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600))
                with tar.extractfile(member) as data, path.open("wb") as dest:
                    shutil.copyfileobj(data, dest)
        plain.unlink()
        counts = {}
        for name in (DATABASE, CHECKPOINTS):
            path = target / name
            if not path.exists():
                continue
            con = sqlite3.connect(path)
            try:
                if con.execute("pragma integrity_check").fetchall() != [("ok",)]:
                    raise ExportError(f"{name} fails the integrity check")
            finally:
                con.close()
            counts[name] = row_counts(path)
            if counts[name] != manifest["tables"].get(name):
                raise ExportError(f"{name} does not hold the rows its manifest lists")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return {"directory": str(target), "tables": counts}
