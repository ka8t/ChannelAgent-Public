"""The secret scanner lives in the virtualenv: scripts/install_gitleaks.sh installs a pinned,
checksum-checked gitleaks into .venv/bin, start.sh --native and the CI use it, and
scripts/secret_scan.sh runs it on the history or on a folder.
"""

import re
import secrets
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
INSTALL = REPO / "scripts" / "install_gitleaks.sh"
SCAN = REPO / "scripts" / "secret_scan.sh"
GITLEAKS = REPO / ".venv" / "bin" / "gitleaks"


def test_the_installer_pins_one_version_and_a_checksum_per_platform():
    script = INSTALL.read_text()
    assert re.search(r'^VERSION="\d+\.\d+\.\d+"$', script, re.M)
    platforms = re.findall(r'PLATFORM="(\w+)"\s+SHA256="([0-9a-f]{64})"', script)
    assert {p for p, _ in platforms} == {"darwin_arm64", "darwin_x64", "linux_arm64", "linux_x64"}
    # the archive is checked before anything is unpacked or installed
    assert script.index('if [ "$actual" != "$SHA256" ]') < script.index("tar -xzf")
    assert script.index("tar -xzf") < script.index('install -m 755 "$WORK/gitleaks" "$TARGET"')
    assert 'TARGET="$VENV_DIR/bin/gitleaks"' in script


def test_start_sh_installs_it_with_the_dependencies():
    script = (REPO / "start.sh").read_text()
    venv = script[script.index("setup_venv() {") :]
    venv = venv[: venv.index("\n}\n")]
    assert venv.index("pip install --quiet -r requirements.txt") < venv.index(
        "bash scripts/install_gitleaks.sh"
    )


def test_the_ci_scans_with_the_same_gitleaks():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "scripts/install_gitleaks.sh" in ci and "scripts/secret_scan.sh" in ci
    assert "zricethezav/gitleaks" not in ci


needs_gitleaks = pytest.mark.skipif(
    not GITLEAKS.exists(), reason="gitleaks not installed: run scripts/install_gitleaks.sh"
)


@needs_gitleaks
def test_the_history_has_no_secret():
    scan = subprocess.run(["bash", str(SCAN)], capture_output=True, text=True, timeout=120)
    assert scan.returncode == 0, scan.stdout[-2000:] + scan.stderr[-2000:]
    assert "no leaks found" in scan.stdout + scan.stderr


@needs_gitleaks
def test_a_planted_token_is_found(tmp_path):
    (tmp_path / "config.py").write_text(f'token = "ghp_{secrets.token_hex(18)}"\n')
    scan = subprocess.run(
        ["bash", str(SCAN), "--dir", str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    assert scan.returncode == 1
    assert "leaks found: 1" in scan.stdout + scan.stderr
