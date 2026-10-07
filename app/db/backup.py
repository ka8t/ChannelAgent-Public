"""Backup before migrating.

`init_db()` runs `alembic upgrade head` at every startup, so starting the
application or the console can change the schema of the real database. A
migration that fails half-way, or that turns out to be wrong, would leave the
only copy modified. Before applying any migration to an existing SQLite file
that is behind head, this module copies it with SQLite's online backup API
(consistent even if something is writing), checks the copy, and keeps the
last few. If the copy cannot be made or does not check out, the migration does
not run.

Restore: stop the application, then copy the wanted file from
`<data dir>/backups/` over the database file.
"""

import logging
import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("channelagent")

BACKUP_DIR_NAME = "backups"


class BackupError(RuntimeError):
    """The backup could not be made or verified: the migration must not run."""


def _current_revision(path: Path) -> str | None:
    """The Alembic revision stored in the file, or None when there is no
    version table (a fresh file, or a database from before Alembic).
    """
    con = sqlite3.connect(path)
    try:
        try:
            row = con.execute("select version_num from alembic_version").fetchone()
        except sqlite3.OperationalError:
            return None
        return row[0] if row else None
    finally:
        con.close()


def _has_tables(path: Path) -> bool:
    con = sqlite3.connect(path)
    try:
        count = con.execute("select count(*) from sqlite_master where type = 'table'")
        return count.fetchone()[0] > 0
    finally:
        con.close()


def _row_counts(con: sqlite3.Connection) -> dict[str, int]:
    tables = [r[0] for r in con.execute("select name from sqlite_master where type = 'table'")]
    return {t: con.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables}


def copy_consistent(source: Path, target: Path) -> dict[str, int]:
    """Copy `source` into `target` with SQLite's online backup API and return the row count of
    every table of the copied state. The rows are counted and the pages copied inside one read
    transaction, so both see the same state. Before, a copy was compared with the live source
    afterwards: any row written in between (a conversation, a scheduled task, the API call
    trail) failed it ("the backup has 0 rows in task_config, the source has 1",
    2026-10-04). Used by the backups and by the export (app/admin/export.py)."""
    src = sqlite3.connect(source, isolation_level=None)
    dst = sqlite3.connect(target)
    try:
        src.execute("begin")
        expected = _row_counts(src)
        src.backup(dst)
        src.execute("commit")
    finally:
        dst.close()
        src.close()
    return expected


def _verify(target: Path, expected: dict[str, int]) -> None:
    """The copy opens, passes the integrity check and holds, in every table, the number of
    rows the source had at the moment it was copied (`expected`, counted in the same read
    transaction as the copy, see `make_backup`).
    """
    dst = sqlite3.connect(target)
    try:
        if dst.execute("pragma integrity_check").fetchall() != [("ok",)]:
            raise BackupError(f"the backup {target.name} fails the integrity check")
        copied = _row_counts(dst)
        for table, a in expected.items():
            b = copied.get(table)
            if a != b:
                raise BackupError(f"the backup has {b} rows in {table}, the source has {a}")
    finally:
        dst.close()


_STAMP = re.compile(r"-(\d{8}T\d{12}Z)\.db$")
# Copies that are the way back from a deliberate operation: never rotated.
_NEVER_PRUNED = ("-prerekey-", "-before-restore-")
# Copies with their own rotation or none: the migration rotation leaves them alone.
# Before, it kept the 5 newest files of the stem whatever their label, so a migration
# could delete manual backups, and would have deleted scheduled ones.
_OWN_ROTATION = ("-manual-", "-scheduled-")
SCHEDULED_LABEL = "scheduled"


def _prune(directory: Path, stem: str, keep: int, label: str | None = None) -> None:
    """Keep the `keep` newest backups of one kind and delete the older ones.

    Without `label`: the migration backups (their label is the Alembic revision).
    With `label`: only the files `<stem>-<label>-<stamp>.db`.

    Newest is decided by the UTC stamp at the end of the file name, not by the
    whole name: the alembic revision sits in front of it and sorts like a random
    string. A file without a stamp is never touched, and neither is a
    "prerekey" copy (the way back from a key rotation) or a "before-restore"
    copy (the way back from a restore).
    """
    dated = []
    pattern = f"{stem}-{label}-*.db" if label else f"{stem}-*.db"
    for path in directory.glob(pattern):
        if any(tag in path.name for tag in _NEVER_PRUNED):
            continue
        if label is None and any(tag in path.name for tag in _OWN_ROTATION):
            continue
        match = _STAMP.search(path.name)
        if match:
            dated.append((match.group(1), path.name, path))
    dated.sort()
    for _stamp, _name, old in dated[:-keep] if keep > 0 else []:
        old.unlink()
        logger.info("Removed the old backup %s", old.name)


def make_backup(path: Path, label: str) -> Path:
    """Copy `path` into `backups/` next to it with SQLite's online backup API,
    verify the copy, and return it. Raises BackupError (and leaves no bad copy)
    if it cannot be made or does not check out.
    """
    directory = path.parent / BACKUP_DIR_NAME
    if not directory.exists():
        directory.mkdir(parents=True, mode=0o700)  # a copy of everything: owner only
        os.chmod(directory, 0o700)  # mkdir's mode is filtered by the umask; be exact
    # Only a directory this function just created is changed: an existing one
    # keeps whatever mode its owner gave it.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = directory / f"{path.stem}-{label}-{stamp}.db"
    # Created private before any data is copied into it, whatever the umask.
    os.close(os.open(target, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600))
    try:
        expected = copy_consistent(path, target)
    except sqlite3.Error as exc:
        target.unlink(missing_ok=True)
        raise BackupError(f"could not back up {path.name}: {exc}") from exc
    try:
        _verify(target, expected)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def backup_before_migration(database_url: str, head_revision: str, keep: int) -> Path | None:
    """Back up the database when a migration is about to change it.

    Returns the backup path, or None when nothing had to be done: not a SQLite
    file, no file yet, an empty file, already at head, or `keep` is 0.
    """
    from app.db.session import sqlite_file_path

    path = sqlite_file_path(database_url)
    if keep <= 0 or path is None or not path.exists() or path.stat().st_size == 0:
        return None
    current = _current_revision(path)
    if current == head_revision or (current is None and not _has_tables(path)):
        return None

    target = make_backup(path, current or "none")
    logger.info(
        "Backed up %s to %s before migrating from %s to %s",
        path.name,
        target,
        current or "no revision",
        head_revision,
    )
    _prune(target.parent, path.stem, keep)
    return target
