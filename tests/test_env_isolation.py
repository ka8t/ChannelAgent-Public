"""The tests never see the developer's own settings: tests/conftest.py removes every variable
of .env.example from the environment before it sets the test values. Without it, a shell that
exported EMAIL_PASSWORD failed two quoting tests, and a real ENCRYPTION_KEY would have been used.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_variables_exported_by_the_shell_do_not_reach_the_tests():
    shell = {
        **os.environ,
        "EMAIL_PASSWORD": "from-the-shell",
        "LLAMA_PORT": "9999",
        "API_URL": "https://elsewhere.example",
        "ENCRYPTION_KEY": "not-the-test-key",
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    code = (
        "import json, os, runpy, sys\n"
        f"runpy.run_path({str(REPO / 'tests' / 'conftest.py')!r})\n"
        "names = ['EMAIL_PASSWORD', 'LLAMA_PORT', 'API_URL', 'ENCRYPTION_KEY', 'HOME']\n"
        "print(json.dumps({n: os.environ.get(n) for n in names}))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=shell, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr[-2000:]
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen["EMAIL_PASSWORD"] is None and seen["LLAMA_PORT"] is None
    assert seen["API_URL"] is None
    assert seen["ENCRYPTION_KEY"] not in (None, "not-the-test-key")
    assert seen["HOME"] == shell["HOME"], "a variable that is not a setting stays"
