"""Rotate ENCRYPTION_KEY: re-encrypt every stored value with a new key.

    OLD_ENCRYPTION_KEY=<old key> python -m app.admin.rekey --dry-run
    OLD_ENCRYPTION_KEY=<old key> python -m app.admin.rekey

The NEW key is the one the application is configured with (ENCRYPTION_KEY in
`.env`), the OLD one comes from the environment. Stop the application first.

What it covers: `action_logs.text`, `access_requests.first_message_text`,
`channel_identities.raw_address` in the application database, and the payloads
of the conversation checkpoints (`checkpoints.checkpoint`, `writes.value`).

Safety, in this order: every value is classified without writing anything
(readable with the old key, already on the new key, or unreadable by both); if
any value is unreadable nothing is written unless --allow-unreadable is given;
both databases are copied into `backups/` (a "prerekey" copy, never rotated
away); all changes are made in one transaction per database; then every value
is read again with the new key and none may still need the old one. It can be
run twice: values already on the new key are left alone.
"""

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from app.db.backup import make_backup
from app.logging_setup import install_redaction
from app.security.permissions import harden_process

# (table, key column, value column)
APP_COLUMNS = (
    ("action_logs", "id", "text"),
    ("access_requests", "id", "first_message_text"),
    ("request_messages", "id", "text"),
    ("channel_identities", "id", "raw_address"),
    ("admin_events", "id", "details"),
    ("agents", "id", "system_prompt"),
    ("agents", "id", "purpose"),
    ("mcp_servers", "id", "env_vars"),
    ("mcp_calls", "id", "detail"),
    ("mcp_calls", "id", "arguments"),
    ("memory_entries", "id", "title"),
    ("memory_entries", "id", "content"),
    ("scheduled_tasks", "id", "prompt"),
    ("admin_accounts", "id", "totp_secret"),
)
# LangGraph SQLite checkpointer tables: (table, key columns, type column, blob column)
CHECKPOINT_COLUMNS = (
    ("checkpoints", ("thread_id", "checkpoint_ns", "checkpoint_id"), "type", "checkpoint"),
    ("writes", ("thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"), "type", "value"),
)


class RekeyError(RuntimeError):
    pass


@dataclass
class TargetReport:
    old: int = 0  # readable with the old key, will be re-encrypted
    new: int = 0  # already readable with the new key
    unreadable: int = 0  # readable with neither


@dataclass
class RekeyReport:
    targets: dict[str, TargetReport] = field(default_factory=dict)
    applied: bool = False
    backups: list[Path] = field(default_factory=list)
    verified_new: int = 0
    still_old: int = 0

    @property
    def unreadable(self) -> int:
        return sum(t.unreadable for t in self.targets.values())

    @property
    def to_rewrite(self) -> int:
        return sum(t.old for t in self.targets.values())


def _classify(value, old: Fernet, new: Fernet) -> tuple[str, bytes | None]:
    """("old", new token) | ("new", None) | ("unreadable", None)."""
    raw = value.encode() if isinstance(value, str) else bytes(value)
    try:
        plain = old.decrypt(raw)
    except InvalidToken:
        try:
            new.decrypt(raw)
        except InvalidToken:
            return "unreadable", None
        return "new", None
    return "old", new.encrypt(plain)


def _plan_app(con: sqlite3.Connection, old: Fernet, new: Fernet, report: RekeyReport):
    updates = []
    for table, key, column in APP_COLUMNS:
        target = report.targets.setdefault(f"{table}.{column}", TargetReport())
        for row_key, value in con.execute(
            f'select "{key}", "{column}" from "{table}" where "{column}" is not null'
        ).fetchall():
            kind, token = _classify(value, old, new)
            setattr(target, kind, getattr(target, kind) + 1)
            if kind == "old":
                updates.append(
                    (
                        f'update "{table}" set "{column}" = ? where "{key}" = ?',
                        (token.decode(), row_key),
                    )
                )
    return updates


def _plan_checkpoints(con: sqlite3.Connection, old: Fernet, new: Fernet, report: RekeyReport):
    updates = []
    for table, keys, type_col, blob_col in CHECKPOINT_COLUMNS:
        target = report.targets.setdefault(f"{table}.{blob_col}", TargetReport())
        key_list = ", ".join(f'"{k}"' for k in keys)
        where = " and ".join(f'"{k}" = ?' for k in keys)
        for row in con.execute(
            f'select {key_list}, "{type_col}", "{blob_col}" from "{table}" '
            f'where "{type_col}" like \'%+fernet\' and "{blob_col}" is not null'
        ).fetchall():
            row_keys, blob = row[: len(keys)], row[-1]
            kind, token = _classify(blob, old, new)
            setattr(target, kind, getattr(target, kind) + 1)
            if kind == "old":
                updates.append(
                    (f'update "{table}" set "{blob_col}" = ? where {where}', (token, *row_keys))
                )
    return updates


def _verify(app_con, cp_con, new: Fernet, old: Fernet, report: RekeyReport) -> None:
    r = RekeyReport()
    _plan_app(app_con, old, new, r)
    if cp_con is not None:
        _plan_checkpoints(cp_con, old, new, r)
    report.verified_new = sum(t.new for t in r.targets.values())
    report.still_old = sum(t.old for t in r.targets.values())
    if r.unreadable and not report.unreadable:
        raise RekeyError("verification found values that neither key can read")


def run_rekey(
    db_path: Path,
    checkpoint_path: Path | None,
    old_key: str,
    new_key: str,
    *,
    dry_run: bool = False,
    allow_unreadable: bool = False,
    backup: bool = True,
) -> RekeyReport:
    if not old_key or not new_key:
        raise RekeyError("both the old and the new key are required")
    if old_key == new_key:
        raise RekeyError("the old and the new key are the same: nothing to rotate")
    try:
        old, new = Fernet(old_key.encode()), Fernet(new_key.encode())
    except ValueError as exc:
        raise RekeyError(f"not a valid Fernet key: {exc}") from exc
    if not db_path.exists():
        raise RekeyError(f"the database {db_path} does not exist")

    report = RekeyReport()
    app_con = sqlite3.connect(db_path)
    has_cp = checkpoint_path is not None and checkpoint_path.exists()
    cp_con = sqlite3.connect(checkpoint_path) if has_cp else None
    try:
        app_updates = _plan_app(app_con, old, new, report)
        cp_updates = _plan_checkpoints(cp_con, old, new, report) if cp_con else []
        if dry_run:
            return report
        if report.unreadable and not allow_unreadable:
            raise RekeyError(
                f"{report.unreadable} value(s) can be read with neither key; nothing was "
                "changed. Restore the right old key, or use --allow-unreadable to leave "
                "those values as they are."
            )
        if not app_updates and not cp_updates:
            _verify(app_con, cp_con, new, old, report)  # nothing to do, and no useless backup
            return report
        if backup:
            report.backups.append(make_backup(db_path, "prerekey"))
            if has_cp:
                report.backups.append(make_backup(checkpoint_path, "prerekey"))
        for con, updates in ((app_con, app_updates), (cp_con, cp_updates)):
            if con is None:
                continue
            con.execute("begin immediate")
            for sql, params in updates:
                con.execute(sql, params)
        for con in (app_con, cp_con):
            if con is not None:
                con.commit()
        report.applied = True
        _verify(app_con, cp_con, new, old, report)
        if report.still_old:
            raise RekeyError(f"{report.still_old} value(s) can still be read with the old key")
        return report
    except Exception:
        for con in (app_con, cp_con):
            if con is not None and con.in_transaction:
                con.rollback()
        raise
    finally:
        app_con.close()
        if cp_con is not None:
            cp_con.close()


def _describe(report: RekeyReport) -> str:
    lines = []
    for name, t in report.targets.items():
        lines.append(
            f"  {name:<40} old key: {t.old:>5}   already new: {t.new:>5}"
            f"   unreadable: {t.unreadable:>5}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-encrypt stored data with a new ENCRYPTION_KEY."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="classify every value, write nothing"
    )
    parser.add_argument(
        "--allow-unreadable",
        action="store_true",
        help="go on even if some values are readable with neither key",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="skip the prerekey copies (not advised)"
    )
    parser.add_argument(
        "--old-key-env",
        default="OLD_ENCRYPTION_KEY",
        help="environment variable holding the OLD key (default %(default)s)",
    )
    args = parser.parse_args(argv)
    install_redaction()
    harden_process()

    from app.checkpoints import checkpoint_db_path
    from app.config import get_settings
    from app.db.session import sqlite_file_path

    settings = get_settings()
    db_path = sqlite_file_path(settings.database_url)
    if db_path is None:
        print("Only SQLite databases are supported.", file=sys.stderr)
        return 1
    try:
        report = run_rekey(
            db_path,
            checkpoint_db_path(),
            os.environ.get(args.old_key_env, ""),
            settings.encryption_key,
            dry_run=args.dry_run,
            allow_unreadable=args.allow_unreadable,
            backup=not args.no_backup,
        )
    except RekeyError as exc:
        print(f"Rekey refused: {exc}", file=sys.stderr)
        return 2
    print(_describe(report))
    if args.dry_run:
        print(
            f"Dry run: {report.to_rewrite} value(s) would be re-encrypted, "
            f"{report.unreadable} unreadable. Nothing written."
        )
        return 0
    for backup in report.backups:
        print(f"Backup: {backup}")
    print(
        f"Done: {report.to_rewrite} value(s) re-encrypted; verified {report.verified_new} readable "
        f"with the new key, {report.still_old} still readable with the old one."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
