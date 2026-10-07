"""File permissions: the secrets and the data are readable by their
owner only.

Everything the application creates is private: `harden_process()` sets
umask 077 at every entry point, so new files are 600 and new directories
700. Files the owner already placed are never changed, only reported
(`warn_about_loose_files`): changing the mode of something the owner put
there is the owner's call.
"""

import logging
import os
import stat
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger("channelagent")

_GROUP_AND_OTHER = 0o077


def harden_process() -> None:
    """New files 600, new directories 700, for the rest of this process."""
    os.umask(0o077)


def loose_files(paths: Iterable[Path]) -> list[tuple[Path, int]]:
    """The existing files among `paths` that other users can access, with
    their mode. Missing paths are skipped (a container has no .env).
    """
    found = []
    for path in paths:
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            continue
        if mode & _GROUP_AND_OTHER:
            found.append((path, mode))
    return found


def warn_about_loose_files(paths: Iterable[Path]) -> int:
    """One WARNING listing every file that other users can access. Returns
    how many there were. Never changes a mode.
    """
    loose = loose_files(paths)
    if loose:
        listing = ", ".join(f"{path} ({mode:o})" for path, mode in loose)
        fix = " ".join(str(path) for path, _ in loose)
        logger.warning(
            "Readable by other users: %s. They hold secrets or the encrypted data. "
            "Restrict them with: chmod 600 %s",
            listing,
            fix,
        )
    return len(loose)


def warn_about_loose_application_files() -> int:
    """Check the files a deployment holds: .env, the database and the
    conversation checkpoints.
    """
    from app.checkpoints import checkpoint_db_path
    from app.config import get_settings
    from app.db.session import sqlite_file_path

    paths = [Path(".env")]
    database = sqlite_file_path(get_settings().database_url)
    if database is not None:
        paths.append(database)
    paths.append(checkpoint_db_path())
    return warn_about_loose_files(paths)
