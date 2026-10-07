"""Tests: the migration backups that are kept are the newest ones,
whatever the alembic revision in their file name sorts like.
"""

from pathlib import Path

from app.db.backup import _prune

# Revisions chosen so that name order is the reverse of date order for some pairs.
OLDEST = "channelagent-bcf3aa387f5e-20260101T000000000000Z.db"
MIDDLE = "channelagent-5527b11034f7-20260201T000000000000Z.db"
NEWEST = "channelagent-85b89421b227-20260301T000000000000Z.db"


def _make(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).write_text("x")


def _left(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def test_keep_one_keeps_the_newest_backup_even_if_its_revision_sorts_first(tmp_path):
    _make(tmp_path, OLDEST, MIDDLE, NEWEST)
    _prune(tmp_path, "channelagent", 1)
    assert _left(tmp_path) == [NEWEST]


def test_keep_two_keeps_the_two_newest(tmp_path):
    _make(tmp_path, OLDEST, MIDDLE, NEWEST)
    _prune(tmp_path, "channelagent", 2)
    assert _left(tmp_path) == sorted([MIDDLE, NEWEST])


def test_the_order_does_not_depend_on_when_the_files_were_created(tmp_path):
    for name in (NEWEST, MIDDLE, OLDEST):  # created newest first, so oldest has the newest mtime
        _make(tmp_path, name)
    _prune(tmp_path, "channelagent", 1)
    assert _left(tmp_path) == [NEWEST]


def test_nothing_is_deleted_when_there_are_not_more_than_keep(tmp_path):
    _make(tmp_path, OLDEST, MIDDLE)
    _prune(tmp_path, "channelagent", 2)
    assert _left(tmp_path) == sorted([OLDEST, MIDDLE])


def test_a_file_that_does_not_follow_the_naming_is_never_deleted(tmp_path):
    _make(tmp_path, OLDEST, MIDDLE, NEWEST, "channelagent-notes.db", "channelagent-old.db")
    _prune(tmp_path, "channelagent", 1)
    assert _left(tmp_path) == sorted([NEWEST, "channelagent-notes.db", "channelagent-old.db"])


def test_prerekey_and_before_restore_copies_are_never_deleted(tmp_path):
    prerekey = "channelagent-prerekey-20250101T000000000000Z.db"
    restore = "channelagent-before-restore-20250102T000000000000Z.db"
    _make(tmp_path, OLDEST, MIDDLE, NEWEST, prerekey, restore)
    _prune(tmp_path, "channelagent", 1)
    assert _left(tmp_path) == sorted([NEWEST, prerekey, restore])


def test_other_databases_backups_are_left_alone(tmp_path):
    other = "checkpoints-prerekey-20250101T000000000000Z.db"
    other2 = "checkpoints-85b89421b227-20250101T000000000000Z.db"
    _make(tmp_path, OLDEST, NEWEST, other, other2)
    _prune(tmp_path, "channelagent", 1)
    assert _left(tmp_path) == sorted([NEWEST, other, other2])


def test_same_revision_backups_are_ordered_by_their_stamp(tmp_path):
    a = "channelagent-85b89421b227-20260101T000000000000Z.db"
    b = "channelagent-85b89421b227-20260101T000000000001Z.db"
    c = "channelagent-85b89421b227-20260102T000000000000Z.db"
    _make(tmp_path, c, a, b)
    _prune(tmp_path, "channelagent", 2)
    assert _left(tmp_path) == sorted([b, c])
