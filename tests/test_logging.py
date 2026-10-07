"""Regression tests: running migrations at startup (init_db)
must not disable the application's loggers or reset the root level.

alembic/env.py used to call fileConfig() with disable_existing_loggers
left at its default (True), so after init_db() every INFO log of the
app was silent.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.asyncio
async def test_init_db_leaves_app_loggers_enabled(fresh_db):
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)  # what app/main.py does at import
    app_logger = logging.getLogger("channelagent")
    app_logger.disabled = False
    try:
        from app.db.session import init_db

        await init_db()

        assert app_logger.disabled is False
        assert app_logger.isEnabledFor(logging.INFO)
        assert root.level == logging.INFO
    finally:
        root.setLevel(previous_level)


def test_alembic_command_line_still_logs_its_progress(fresh_db):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Running upgrade" in result.stderr
