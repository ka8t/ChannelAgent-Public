"""Tests: the machine's memory in the status, with a warning below 10 GiB."""

import subprocess

from app import host_memory

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              100000.
Pages active:                            900000.
Pages inactive:                          200000.
Pages speculative:                        50000.
"""


def test_macos_counts_free_inactive_and_speculative_pages(monkeypatch):
    def run(cmd, **kwargs):
        out = str(32 * 1024**3) if cmd[0] == "sysctl" else VM_STAT
        return subprocess.CompletedProcess(cmd, 0, stdout=out)

    monkeypatch.setattr(host_memory.sys, "platform", "darwin")
    monkeypatch.setattr(host_memory.subprocess, "run", run)
    found = host_memory.snapshot()
    assert found == {"total_bytes": 32 * 1024**3, "available_bytes": 350000 * 16384, "low": True}
    assert host_memory.line() == (
        "5.3 GiB available of 32 GiB (WARNING: below 10 GiB, a long session may swap)"
    )


def test_linux_reads_mem_available(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       134217728 kB\nMemAvailable:   20971520 kB\n")
    monkeypatch.setattr(host_memory.sys, "platform", "linux")
    monkeypatch.setattr(host_memory, "Path", lambda _p: meminfo)
    assert host_memory.snapshot() == {"total_bytes": 128 * 1024**3,
                                      "available_bytes": 20 * 1024**3, "low": False}  # fmt: skip
    assert host_memory.line() == "20.0 GiB available of 128 GiB"


def test_the_boundary_is_10_gib_and_an_unknown_system_says_so(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    monkeypatch.setattr(host_memory.sys, "platform", "linux")
    monkeypatch.setattr(host_memory, "Path", lambda _p: meminfo)
    meminfo.write_text(f"MemTotal: {64 * 1024**2} kB\nMemAvailable: {10 * 1024**2} kB\n")
    assert host_memory.snapshot()["low"] is False
    meminfo.write_text(f"MemTotal: {64 * 1024**2} kB\nMemAvailable: {10 * 1024**2 - 1} kB\n")
    assert host_memory.snapshot()["low"] is True
    meminfo.write_text("nothing useful\n")
    assert host_memory.snapshot() is None and host_memory.line() == "unknown"


def test_start_sh_status_shows_the_memory():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "start.sh").read_text()
    assert 'echo "  memory           : $(python3 app/host_memory.py' in script
