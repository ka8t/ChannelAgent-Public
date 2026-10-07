"""The machine's memory for the status: total, available, and a warning below
LOW_FREE_BYTES (10 GiB: saturated memory swaps and can stop a long session, owner's
docs/synthese-mac-ia-locale.pdf). Standard library only: `start.sh --status` runs it with the
system python3, `GET /status` imports it.

Available: on Linux `MemAvailable` of /proc/meminfo; on macOS the free, inactive and
speculative pages of `vm_stat` (memory the system gives back without swapping).

    python3 app/host_memory.py   -> one line for start.sh --status
"""

import re
import subprocess
import sys
from pathlib import Path

LOW_FREE_BYTES = 10 * 1024**3
GIB = 1024**3


def _linux() -> tuple[int, int] | None:
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    values = {k: int(v) * 1024 for k, v in re.findall(r"^(\w+):\s+(\d+) kB", text, re.M)}
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None
    return values["MemTotal"], values["MemAvailable"]


def _macos() -> tuple[int, int] | None:
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                   text=True, timeout=5).stdout.strip())  # fmt: skip
        stat = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    page = re.search(r"page size of (\d+) bytes", stat)
    pages = {k: int(v) for k, v in re.findall(r"^Pages (\w+):\s+(\d+)\.", stat, re.M)}
    if not page or "free" not in pages:
        return None
    free = sum(pages.get(k, 0) for k in ("free", "inactive", "speculative"))
    return total, free * int(page.group(1))


def snapshot() -> dict | None:
    """{total_bytes, available_bytes, low}, or None when the system does not say."""
    found = _macos() if sys.platform == "darwin" else _linux()
    if found is None:
        return None
    total, available = found
    return {"total_bytes": total, "available_bytes": available,
            "low": available < LOW_FREE_BYTES}  # fmt: skip


def line() -> str:
    found = snapshot()
    if found is None:
        return "unknown"
    available, total = found["available_bytes"] / GIB, found["total_bytes"] / GIB
    text = f"{available:.1f} GiB available of {total:.0f} GiB"
    return text + (" (WARNING: below 10 GiB, a long session may swap)" if found["low"] else "")


if __name__ == "__main__":
    print(line())
