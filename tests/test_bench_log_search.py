"""The benchmark script stays runnable, so the figures in
Can be reproduced. Small sizes only, in a real process.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_the_benchmark_script_runs_and_prints_one_line_per_size(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_", "CHECKPOINT_"))}
    env["TMPDIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/bench_log_search.py", "300", "600"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-500:]
    lines = result.stdout.strip().splitlines()
    assert lines[1].split()[:3] == ["rows", "best", "case"]
    assert [line.split()[0] for line in lines[2:]] == ["300", "600"]
    assert all(" ms " in line for line in lines[2:])
