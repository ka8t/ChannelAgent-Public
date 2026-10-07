"""Restore a database backup: one guided, checked command.

    ./start.sh --restore                   list the backups, ask which one
    ./start.sh --restore FILE              restore this backup (a name in backups/ or a path)
    ./start.sh --restore --list            only list them
    ./start.sh --restore FILE --yes        do not ask for the RESTORE confirmation
    ./start.sh --restore FILE --allow-unreadable

Before it touches anything it refuses when the application is running, when the
backup is not a healthy ChannelAgent database, when its schema is newer than
this code, or when values in it cannot be decrypted with the current
ENCRYPTION_KEY (unless told to go on). The current database is first copied to
`backups/<name>-before-restore-<time>.db`, then the backup is copied next to it,
checked, and swapped in atomically. The conversation checkpoints are a separate
file and are not restored.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from app.admin.rekey import APP_COLUMNS
from app.db.backup import BACKUP_DIR_NAME, _current_revision, make_backup
from app.logging_setup import install_redaction
from app.security.permissions import harden_process

CONFIRM_WORD = "RESTORE"
_STAMP = re.compile(r"-(\d{8}T\d{12}Z)\.db$")


class RestoreError(RuntimeError):
    """The restore was refused, or failed before the database was replaced."""


@dataclass(frozen=True)
class BackupInfo:
    path: Path
    kind: str  # "migration", "prerekey", "before-restore" or "other"
    revision: str | None  # the alembic revision in the name (migration backups)
    stamp: datetime | None
    size: int


@dataclass
class RestoreReport:
    backup: Path
    database: Path
    before_restore_copy: Path | None = None
    revision: str | None = None
    counts_before: dict[str, int | None] = field(default_factory=dict)
    counts_after: dict[str, int | None] = field(default_factory=dict)
    unreadable: dict[str, int] = field(default_factory=dict)
    needs_migration: bool = False

    @property
    def unreadable_total(self) -> int:
        return sum(self.unreadable.values())


# --- listing ---


def _parse_name(path: Path, stem: str) -> BackupInfo:
    name = path.name
    match = _STAMP.search(name)
    stamp = (
        datetime.strptime(match.group(1), "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC) if match else None
    )
    label = name[len(stem) + 1 : match.start()] if match else ""
    if label == "prerekey":
        kind, revision = "prerekey", None
    elif label == "before-restore":
        kind, revision = "before-restore", None
    elif match and re.fullmatch(r"[0-9a-f]{12}|none", label):
        kind, revision = "migration", (None if label == "none" else label)
    else:
        kind, revision = "other", None
    return BackupInfo(path, kind, revision, stamp, path.stat().st_size)


def list_backups(db_path: Path) -> list[BackupInfo]:
    """The backups of this database, newest first. Backups of other files (the
    conversation checkpoints) are not listed.
    """
    directory = db_path.parent / BACKUP_DIR_NAME
    if not directory.is_dir():
        return []
    infos = [_parse_name(p, db_path.stem) for p in directory.glob(f"{db_path.stem}-*.db")]
    return sorted(
        infos,
        key=lambda i: (i.stamp or datetime.min.replace(tzinfo=UTC), i.path.name),
        reverse=True,
    )


# --- checks ---


def compose_project() -> str:
    """The Docker Compose project of this directory, named the way `docker compose` names it:
    COMPOSE_PROJECT_NAME, else the directory's name, lowercased, other characters dropped.
    Only this project's container blocks a restore, not a copy of it elsewhere."""
    name = os.environ.get("COMPOSE_PROJECT_NAME") or Path.cwd().name
    return re.sub(r"[^a-z0-9_-]", "", name.lower())


def running_reasons(db_path: Path, host: str, port: int) -> list[str]:
    """Why the application looks like it is running (empty when it does not)."""
    reasons = []
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    try:
        with socket.create_connection((probe_host, port), timeout=1):
            reasons.append(f"the Admin API answers on {probe_host}:{port}")
    except OSError:
        pass
    if db_path.exists() and db_path.stat().st_size > 0:
        con = None
        try:
            con = sqlite3.connect(db_path, timeout=0)
            con.execute("begin exclusive")
            con.execute("rollback")
        except sqlite3.OperationalError as exc:
            if "lock" in str(exc).lower() or "busy" in str(exc).lower():
                reasons.append("the database is locked by another process")
        finally:
            if con is not None:
                con.close()
    if shutil.which("docker"):
        try:
            out = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"label=com.docker.compose.project={compose_project()}",
                    "--filter",
                    "name=channelagent",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.split()
        except (OSError, subprocess.SubprocessError):
            out = []
        app_containers = [n for n in out if "llama" not in n]
        if app_containers:
            reasons.append(f"a container is running ({', '.join(app_containers)})")
    return reasons


def _tables(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("select name from sqlite_master where type = 'table'")}


def row_counts(path: Path) -> dict[str, int | None]:
    """Rows in each main table; None for a table the file does not have."""
    from app.admin.service import _COUNTED_MODELS

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        have = _tables(con)
        return {
            m.__tablename__: (
                con.execute(f'select count(*) from "{m.__tablename__}"').fetchone()[0]
                if m.__tablename__ in have
                else None
            )
            for m in _COUNTED_MODELS
        }
    finally:
        con.close()


def _known_revisions() -> tuple[str, set[str]]:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    from app.db.session import _REPO_ROOT

    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
    script = ScriptDirectory.from_config(cfg)
    return script.get_current_head(), {r.revision for r in script.walk_revisions()}


def check_backup(path: Path) -> tuple[str | None, bool]:
    """Refuse anything that is not a healthy ChannelAgent database this code
    can use. Returns (its alembic revision, whether a migration will run).
    """
    if not path.is_file():
        raise RestoreError(f"{path} is not a file")
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            integrity = con.execute("pragma integrity_check").fetchall()
            have = _tables(con)
        finally:
            con.close()
    except sqlite3.DatabaseError as exc:
        raise RestoreError(f"{path.name} is not a readable SQLite database ({exc})") from exc
    if integrity != [("ok",)]:
        raise RestoreError(f"{path.name} fails the integrity check: it is corrupt")
    if not {"users", "alembic_version"} <= have:
        raise RestoreError(
            f"{path.name} is not a ChannelAgent database (no users or alembic_version)"
        )
    revision = _current_revision(path)
    head, known = _known_revisions()
    if revision not in known:
        raise RestoreError(
            f"{path.name} is at schema revision {revision}, which this version of the code does "
            "not know: update the code before restoring it"
        )
    return revision, revision != head


def unreadable_values(path: Path, encryption_key: str) -> dict[str, int]:
    """Encrypted values in `path` that `encryption_key` cannot decrypt, per column."""
    fernet = Fernet(encryption_key.encode())
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    counts: dict[str, int] = {}
    try:
        have = _tables(con)
        for table, _key, column in APP_COLUMNS:
            if table not in have:
                continue
            bad = 0
            for (value,) in con.execute(
                f'select "{column}" from "{table}" where "{column}" is not null'
            ):
                try:
                    fernet.decrypt(value.encode() if isinstance(value, str) else bytes(value))
                except InvalidToken:
                    bad += 1
            if bad:
                counts[f"{table}.{column}"] = bad
    finally:
        con.close()
    return counts


# --- the restore ---


def _copy_private(source: Path, target: Path) -> None:
    """Copy a SQLite file with the online backup API into a new file created
    private (mode 600) before any data goes into it.
    """
    os.close(os.open(target, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600))
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def run_restore(
    db_path: Path,
    backup: Path,
    *,
    encryption_key: str,
    api_host: str = "127.0.0.1",
    api_port: int = 8700,
    yes: bool = False,
    allow_unreadable: bool = False,
    confirm: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> RestoreReport:
    """Restore `backup` over `db_path`. Nothing is changed unless every check
    passes; the only irreversible step, the swap, is preceded by a verified
    copy of the current database.
    """
    backup = backup.resolve()
    if backup == db_path.resolve():
        raise RestoreError("the backup is the live database itself")
    reasons = running_reasons(db_path, api_host, api_port)
    if reasons:
        raise RestoreError(
            "the application is running (" + "; ".join(reasons) + "). Stop it first: "
            "docker compose down, or stop ./start.sh --native"
        )

    revision, needs_migration = check_backup(backup)
    report = RestoreReport(backup=backup, database=db_path, revision=revision)
    report.needs_migration = needs_migration
    report.unreadable = unreadable_values(backup, encryption_key)
    have_current = db_path.exists() and db_path.stat().st_size > 0
    report.counts_before = row_counts(db_path) if have_current else {}
    backup_counts = row_counts(backup)

    out(f"Backup      : {backup.name} (schema revision {revision})")
    out(f"Database    : {db_path}" + ("" if have_current else " (none yet)"))

    def shown(n: int | None) -> str:
        return "-" if n is None else str(n)

    out(
        "Rows        : "
        + ", ".join(
            f"{t} {shown(report.counts_before.get(t))} -> {shown(n)}"
            for t, n in backup_counts.items()
        )
    )
    if needs_migration:
        out(
            "Note        : this backup is behind the current schema; the next start migrates it "
            "(with an automatic backup first)."
        )
    if report.unreadable:
        out(
            f"WARNING     : {report.unreadable_total} encrypted value(s) in this backup cannot be "
            "decrypted with the current ENCRYPTION_KEY: "
            + ", ".join(f"{k} {v}" for k, v in report.unreadable.items())
        )
        if not allow_unreadable:
            raise RestoreError(
                "refused: a backup made under another key stays unreadable (for a 'prerekey' "
                "backup, put the OLD key back in ENCRYPTION_KEY first). "
                "Use --allow-unreadable to restore it anyway."
            )
    if not yes:
        answer = confirm(f"Type {CONFIRM_WORD} to replace the database with this backup: ")
        if answer.strip() != CONFIRM_WORD:
            raise RestoreError("cancelled: nothing was changed")

    if have_current:
        report.before_restore_copy = make_backup(db_path, "before-restore")
        out(f"Kept        : the current database, as {report.before_restore_copy.name}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    staging = db_path.with_name(db_path.name + ".restoring")
    staging.unlink(missing_ok=True)
    try:
        _copy_private(backup, staging)
        con = sqlite3.connect(staging)
        try:
            if con.execute("pragma integrity_check").fetchall() != [("ok",)]:
                raise RestoreError("the staged copy fails the integrity check: nothing was changed")
        finally:
            con.close()
        if row_counts(staging) != backup_counts:
            raise RestoreError("the staged copy differs from the backup: nothing was changed")
        # Leftovers of the old database would be replayed onto the new file.
        for suffix in ("-wal", "-shm", "-journal"):
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
        os.replace(staging, db_path)
    finally:
        staging.unlink(missing_ok=True)
    os.chmod(db_path, 0o600)

    report.counts_after = row_counts(db_path)
    con = sqlite3.connect(db_path)
    try:
        ok = con.execute("pragma integrity_check").fetchall() == [("ok",)]
    finally:
        con.close()
    if not ok:  # cannot happen after the staged check, but never claim success blindly
        raise RestoreError("the restored database fails the integrity check")
    out(
        "Restored    : integrity_check ok; rows now "
        + ", ".join(f"{t} {n if n is not None else '-'}" for t, n in report.counts_after.items())
    )
    out(
        "Checkpoints : the conversation history (checkpoints.db) is a separate file and was NOT "
        "restored: conversations continue from their current state, which can be newer than "
        "this database."
    )
    return report


# --- command line ---


_KIND_TEXT = {
    "migration": "before a migration",
    "prerekey": "before a key rotation",
    "before-restore": "before a restore",
    "other": "other",
}


def describe(number: int, b: BackupInfo) -> str:
    when = b.stamp.strftime("%Y-%m-%d %H:%M:%S UTC") if b.stamp else "-"
    revision = f", revision {b.revision}" if b.revision else ""
    return f"  {number}. {b.path.name}  {when}  {b.size} bytes  ({_KIND_TEXT[b.kind]}{revision})"


def _choose(
    backups: list[BackupInfo], out: Callable[[str], None], ask: Callable[[str], str]
) -> Path:
    for i, b in enumerate(backups, 1):
        out(describe(i, b))
    raw = ask("Number of the backup to restore (blank to cancel): ").strip()
    if not raw:
        raise RestoreError("cancelled: nothing was changed")
    if not raw.isdigit() or not 1 <= int(raw) <= len(backups):
        raise RestoreError(f"{raw!r} is not a number from 1 to {len(backups)}")
    return backups[int(raw) - 1].path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore a database backup.")
    parser.add_argument("backup", nargs="?", help="a file name in backups/ or a path")
    parser.add_argument("--list", action="store_true", help="only list the backups")
    parser.add_argument("--yes", action="store_true", help="skip the RESTORE confirmation")
    parser.add_argument(
        "--allow-unreadable",
        action="store_true",
        help="restore even if some values cannot be decrypted with the current key",
    )
    args = parser.parse_args(argv)
    install_redaction()
    harden_process()

    from app.config import get_settings
    from app.db.session import sqlite_file_path

    settings = get_settings()
    db_path = sqlite_file_path(settings.database_url)
    if db_path is None:
        print("Only SQLite databases are supported.", file=sys.stderr)
        return 1
    backups = list_backups(db_path)
    try:
        if args.list:
            if not backups:
                print(f"No backup in {db_path.parent / BACKUP_DIR_NAME}.")
            for i, b in enumerate(backups, 1):
                print(describe(i, b))
            return 0
        if args.backup:
            candidate = Path(args.backup)
            if not candidate.exists():
                candidate = db_path.parent / BACKUP_DIR_NAME / args.backup
            if not candidate.exists():
                raise RestoreError(f"no backup named {args.backup}")
            chosen = candidate
        else:
            if not backups:
                raise RestoreError(f"no backup in {db_path.parent / BACKUP_DIR_NAME}")
            chosen = _choose(backups, print, input)
        run_restore(
            db_path,
            chosen,
            encryption_key=settings.encryption_key,
            api_host=settings.api_server_host,
            api_port=settings.api_server_port,
            yes=args.yes,
            allow_unreadable=args.allow_unreadable,
        )
    except RestoreError as exc:
        print(f"Restore refused: {exc}", file=sys.stderr)
        return 1
    except EOFError:
        print("Restore refused: cancelled: nothing was changed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
