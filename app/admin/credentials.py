"""The credentials a client of the Admin API sends: the token saved by `sign-in` for
that API address, else API_SERVER_KEY. One implementation for the script client
(app/admin/client.py) and for the lines of `start.sh` that call the API with curl
(`--status`), so the two can never choose differently. Standard library only: `start.sh` runs
it with the system python3.

The tokens live in `$XDG_CONFIG_HOME/channelagent/api-tokens.json` (by default under
`~/.config`), a file of mode 600 in a directory of mode 700, one entry per API address.

    python3 app/admin/credentials.py --header API_BASE_URL
    prints `Authorization: Bearer ...` for curl's `-H @-` (never on a command line);
    exit 0, or 1 when there is neither a saved token nor API_SERVER_KEY.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def token_file() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "channelagent" / "api-tokens.json"


def _read_tokens() -> dict:
    try:
        data = json.loads(token_file().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_tokens(data: dict) -> None:
    path = token_file()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    old_umask = os.umask(0o077)
    try:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2))
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        os.umask(old_umask)


def saved_token(base_url: str) -> dict | None:
    entry = _read_tokens().get(base_url)
    return entry if isinstance(entry, dict) and entry.get("token") else None


def save_token(base_url: str, entry: dict) -> None:
    data = _read_tokens()
    data[base_url] = entry
    _write_tokens(data)


def forget_token(base_url: str) -> None:
    data = _read_tokens()
    if data.pop(base_url, None) is not None:
        _write_tokens(data)


def bearer(base_url: str) -> str | None:
    """The bearer credential for this API address: its saved token, else API_SERVER_KEY."""
    saved = saved_token(base_url)
    if saved:
        return saved["token"]
    return os.environ.get("API_SERVER_KEY") or None


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "--header":
        print("usage: credentials.py --header API_BASE_URL", file=sys.stderr)
        return 2
    credential = bearer(argv[2].rstrip("/"))
    if credential is None:
        return 1
    print(f"Authorization: Bearer {credential}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
