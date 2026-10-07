"""Interactive configuration of .env: one question per variable of .env.example.

Run by `./start.sh --config --edit`, and by `./start.sh --native` when there is no `.env` next to
the script. Standard library only, like `settings_rules` (it runs before any virtualenv):

    python3 app/env_wizard.py .env .env.example

Every answer passes the same checks as `./start.sh --config KEY=VALUE` and `PATCH /config`
(`settings_rules.check_change`), and the file is written once, at the end, after a
confirmation, by `settings_rules.set_many` (the previous .env kept first). Ctrl+C or
end of input stops without writing anything. A secret is typed without echo and never
printed, not even its first characters.

Exit codes: 0 written or nothing to change, 1 stopped or declined (nothing written),
2 not run (no terminal, or a .env that is not a file).
"""

from __future__ import annotations

import base64
import os
import sys

try:
    from app import settings_rules as rules
except ImportError:  # run as a script: app/ is on sys.path, not the project
    import settings_rules as rules

KEEP_HINT = "Enter keeps the value shown, '-' empties it, Ctrl+C stops without writing anything."


def read_secret(prompt: str) -> str:
    """One line from the terminal with the echo off; EOFError at the end of input."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        line = sys.stdin.readline()
    else:
        import termios

        old = termios.tcgetattr(fd)
        quiet = termios.tcgetattr(fd)
        quiet[3] &= ~termios.ECHO
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, quiet)
            line = sys.stdin.readline()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write("\n")
    if not line:
        raise EOFError
    return line.rstrip("\n")


def read_line(prompt: str) -> str:
    return input(prompt)


def example_layout(example_path: str) -> list:
    """(section, help lines, key) per variable of .env.example, in its order. The help is the
    comment block written right above the variable; a `# --- Title ---` line opens a section.
    """
    layout, section, comments = [], "", []
    with open(example_path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                comments = []
            elif line.startswith("# ---"):
                section = line.strip("# -").strip()
                comments = []
            elif line.startswith("#"):
                comments.append(line[1:].strip())
            elif "=" in line:
                layout.append((section, comments, line.partition("=")[0]))
                comments = []
    return layout


def new_encryption_key() -> str:
    """A Fernet key: 32 random bytes, url-safe base64 (the format `cryptography` expects)."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


def run(env_path: str, example_path: str, ask=read_line, ask_secret=read_secret, out=print) -> int:
    exists = os.path.isfile(env_path)
    if os.path.lexists(env_path) and not exists:
        out(f"{env_path} exists but is not a file: nothing was asked, nothing was written.")
        return 2
    current = rules.read_env(env_path if exists else example_path)
    known = rules.example_keys(example_path)
    current_key = current.get("ENCRYPTION_KEY", "") if exists else ""
    layout = example_layout(example_path)
    if exists:
        out(f"Configuring {env_path}: {len(layout)} variables, the current values are shown.")
    else:
        out(f"No {env_path} yet: {len(layout)} variables, the defaults of .env.example are shown.")
    out(KEEP_HINT)

    changes: dict = {}
    section_shown = None
    try:
        for section, help_lines, key in layout:
            if section != section_shown:
                out("")
                out(f"--- {section} ---")
                section_shown = section
            for text in help_lines:
                out(f"  {text}")
            value = current.get(key, "")
            secret = key in rules.SENSITIVE
            if key == "ENCRYPTION_KEY" and current_key:
                out("ENCRYPTION_KEY is set and is not changed here "
                    "(./start.sh --rekey rotates it).")
                continue
            while True:
                if key == "ENCRYPTION_KEY":
                    answer = ask_secret("ENCRYPTION_KEY [Enter generates a new key]: ")
                elif secret:
                    shown = "set" if value else "not set"
                    answer = ask_secret(f"{key} [{shown}, typed without echo]: ")
                else:
                    answer = ask(f"{key} [{value}]: ")
                if answer == "":
                    if key == "ENCRYPTION_KEY":
                        new = new_encryption_key()
                        out("  A new key will be written. Copy it from .env to your password "
                            "manager: without it the encrypted data cannot be read.")
                    else:
                        new = value
                elif answer == "-":
                    new = ""
                else:
                    new = answer
                if new == value:
                    changes.pop(key, None)
                    break
                try:
                    rules.check_change(key, new, known, current_key, allow_initial_key=True)
                except rules.ConfigError as exc:
                    out(f"  {exc}")
                    continue
                changes[key] = new
                break
        out("")
        if exists and not changes:
            out(f"Nothing changed: {env_path} was not written.")
            return 0
        if changes:
            out(f"{len(changes)} change(s):")
            for key, new in changes.items():
                if key in rules.SENSITIVE:
                    out(f"  {key} (secret, not shown)")
                else:
                    out(f"  {key}={new}")
        target = f"write them to {env_path}" if exists else f"create {env_path}"
        if ask(f"{target[0].upper()}{target[1:]}? [y/N]: ").strip().lower() not in ("y", "yes"):
            out("Nothing was written.")
            return 1
    except (KeyboardInterrupt, EOFError):
        out("")
        out("Stopped: nothing was written.")
        return 1

    try:
        result = rules.set_many(
            env_path, example_path, changes, allow_initial_key=True, create=not exists
        )
    except rules.ConfigError as exc:
        out(str(exc))
        return 1
    if result["backup"]:
        out(f"Written. The previous {env_path} is kept as {result['backup']} (mode 600).")
    else:
        out(f"Created {env_path} (mode 600).")
    out("Changes apply at the next start of the application.")
    return 0


def main(argv: list) -> int:
    if len(argv) != 3:
        print("usage: python3 app/env_wizard.py ENV_FILE EXAMPLE_FILE", file=sys.stderr)
        return 2
    if not sys.stdin.isatty():
        print("The configuration asks questions: it needs a terminal. Nothing was written.",
              file=sys.stderr)
        return 2
    return run(argv[1], argv[2])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
