"""Guided rotation of ENCRYPTION_KEY: `./start.sh --rekey`.

One sequence around the rotation tool (app/admin/rekey.py), so the
steps cannot be done in the wrong order:

  1. refuse while the application runs;
  2. read the current key from `.env` (it becomes the OLD key, held in memory,
     never printed) and generate the new one;
  3. dry run: count what would be re-encrypted and what is unreadable; stop
     when something is unreadable unless --allow-unreadable is given;
  4. ask for confirmation (type ROTATE), unless --yes;
  5. keep the current `.env` as `.env.pre-rekey` (mode 600), put the new key in
     `.env`, re-encrypt (the tool copies both databases into backups/ first);
  6. read every value back in a fresh process that loads the key from `.env`,
     the way the application does;
  7. say what to delete, and when.

If the rotation fails before anything was rewritten, `.env` is restored. The
new key is never printed: it is in `.env`, from where you copy it to a password
manager. Nothing is ever deleted here: `.env.pre-rekey` and the prerekey copies
are the safety nets, and only the owner removes them.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from cryptography.fernet import Fernet

from app.admin.rekey import RekeyError, _describe, run_rekey
from app.logging_setup import install_redaction
from app.security.permissions import harden_process

ENV_FILE = Path(".env")
SAFETY_NET = Path(".env.pre-rekey")
CONFIRMATION_WORD = "ROTATE"
_KEY_LINE = re.compile(r"^ENCRYPTION_KEY=(.*)$")

# Run in a fresh process, without ENCRYPTION_KEY in its environment, so the key
# comes from .env exactly as it does when the application starts.
_READ_BACK = """
import asyncio
from app.admin.service import count_undecryptable
from app.db.session import session_scope

async def main():
    async with session_scope() as session:
        print(sum((await count_undecryptable(session)).values()))

asyncio.run(main())
"""


def _current_key(env_path: Path) -> str:
    for line in env_path.read_text().splitlines():
        match = _KEY_LINE.match(line)
        if match:
            return match.group(1).strip().strip("'\"")
    return ""


def _replace_key(env_path: Path, new_key: str) -> None:
    lines = env_path.read_text().splitlines(keepends=True)
    out = [f"ENCRYPTION_KEY={new_key}\n" if _KEY_LINE.match(ln) else ln for ln in lines]
    _write_private(env_path, "".join(out))


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, text.encode())
    finally:
        os.close(fd)


def _read_back_unreadable() -> int:
    env = {k: v for k, v in os.environ.items() if k not in ("ENCRYPTION_KEY", "OLD_ENCRYPTION_KEY")}
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", _READ_BACK],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RekeyError("the read-back process failed: " + result.stderr.strip()[-300:])
    return int(result.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Guided rotation of ENCRYPTION_KEY.")
    parser.add_argument("--dry-run", action="store_true", help="show the counts, change nothing")
    parser.add_argument("--yes", action="store_true", help=f"do not ask for {CONFIRMATION_WORD}")
    parser.add_argument(
        "--allow-unreadable",
        action="store_true",
        help="go on although some values are readable with neither key (they stay as they are)",
    )
    args = parser.parse_args(argv)
    install_redaction()
    harden_process()

    from app.admin.restore import running_reasons
    from app.checkpoints import checkpoint_db_path
    from app.config import get_settings
    from app.db.session import sqlite_file_path

    settings = get_settings()
    db_path = sqlite_file_path(settings.database_url)
    if db_path is None:
        print("Only SQLite databases are supported.", file=sys.stderr)
        return 1
    if not ENV_FILE.exists():
        print("No .env here: run this from the project directory.", file=sys.stderr)
        return 1

    print("==> 1/6 Checking that the application is stopped")
    reasons = running_reasons(db_path, settings.api_server_host, settings.api_server_port)
    if reasons:
        print(
            "Refused: the application looks like it is running ("
            + "; ".join(reasons)
            + "). Stop it first: ./start.sh --stop",
            file=sys.stderr,
        )
        return 2

    old_key = _current_key(ENV_FILE)
    if not old_key:
        print(
            "Refused: ENCRYPTION_KEY is empty in .env, there is nothing to rotate.",
            file=sys.stderr,
        )
        return 2
    if SAFETY_NET.exists() and not args.dry_run:
        print(
            f"Refused: {SAFETY_NET} already exists. It holds the key of an earlier rotation "
            "and may still be the only way to read a prerekey backup. Move or delete it "
            "yourself first.",
            file=sys.stderr,
        )
        return 2
    new_key = Fernet.generate_key().decode()
    checkpoints = checkpoint_db_path()

    print("==> 2/6 Dry run (nothing is written)")
    try:
        report = run_rekey(db_path, checkpoints, old_key, new_key, dry_run=True)
    except RekeyError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(_describe(report))
    print(f"{report.to_rewrite} value(s) to re-encrypt, {report.unreadable} unreadable.")
    if report.unreadable and not args.allow_unreadable:
        print(
            "Refused: some values can be read with neither the current key nor a new one, so "
            "they were probably written with another key. Find that key first "
            "(python -m app.admin.rekey uses OLD_ENCRYPTION_KEY), or rerun with "
            "--allow-unreadable to leave those values as they are.",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        print("Dry run only: nothing was changed.")
        return 0

    print("==> 3/6 Confirmation")
    if not args.yes:
        answer = input(f"Type {CONFIRMATION_WORD} to rotate the key now: ").strip()
        if answer != CONFIRMATION_WORD:
            print("Cancelled, nothing was changed.")
            return 1

    print("==> 4/6 Rotating (the databases are copied into backups/ first)")
    previous_env = ENV_FILE.read_text()
    _write_private(SAFETY_NET, previous_env)
    _replace_key(ENV_FILE, new_key)
    try:
        done = run_rekey(
            db_path, checkpoints, old_key, new_key, allow_unreadable=args.allow_unreadable
        )
    except Exception as exc:
        return _after_failure(exc, previous_env, db_path, checkpoints, old_key, new_key)
    for backup in done.backups:
        print(f"Backup: {backup}")
    print(
        f"{done.to_rewrite} value(s) re-encrypted; {done.verified_new} read with the new key, "
        f"{done.still_old} still readable with the old one."
    )

    print("==> 5/6 Reading everything back the way the application does")
    try:
        unreadable = _read_back_unreadable()
    except (RekeyError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"!! The read-back could not run: {exc}", file=sys.stderr)
        return 3
    print(f"Values the key in .env cannot decrypt: {unreadable}")
    if unreadable and not args.allow_unreadable:
        print("!! Some values are unreadable with the new key: do not start the application.",
              file=sys.stderr)
        return 3

    print("==> 6/6 Done. What to do now")
    print("  - The new key is in .env (ENCRYPTION_KEY). It is not shown here: open .env and copy")
    print("    it to your password manager, apart from the data.")
    print("  - Start the application again: ./start.sh")
    print("  - Safety nets, deleted by you only, once the new key is stored and the application")
    print(f"    has run fine: {SAFETY_NET} (the old key) and the prerekey copies in")
    print("    backups/ (data encrypted with the old key). Until then keep them: the copies")
    print("    are unreadable without that old key.")
    return 0


def _after_failure(exc, previous_env, db_path, checkpoints, old_key, new_key) -> int:
    """Decide what state .env should be in after a failed rotation."""
    print(f"!! The rotation failed: {exc}", file=sys.stderr)
    try:
        state = run_rekey(db_path, checkpoints, old_key, new_key, dry_run=True)
        rewritten = sum(t.new for t in state.targets.values())
    except Exception:
        rewritten = -1  # cannot tell
    if rewritten == 0:
        _write_private(ENV_FILE, previous_env)
        SAFETY_NET.unlink(missing_ok=True)
        print("Nothing had been re-encrypted: .env was restored, the old key is in place.",
              file=sys.stderr)
        return 3
    print(
        f"!! Some data may already be on the new key. .env keeps the new key and {SAFETY_NET} "
        "keeps the old one. Do not start the application. To finish: put the old key from "
        f"{SAFETY_NET} in OLD_ENCRYPTION_KEY and run python -m app.admin.rekey "
        "(it is safe to run twice).",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    sys.exit(main())
