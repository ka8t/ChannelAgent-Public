"""Tests: `settings_rules.backup_env`, the copy kept before any write of .env."""

import os
import stat

import pytest

from app import settings_rules


@pytest.fixture
def env(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(b"ENCRYPTION_KEY=abc\nLLAMA_PORT=8080\n")
    path.chmod(0o600)
    return path


def test_the_backup_is_identical_private_and_never_replaces_an_earlier_one(env):
    names = [settings_rules.backup_env(str(env)) for _ in range(3)]
    assert len(set(names)) == 3, names
    for name in names:
        copy = env.parent / name
        assert copy.read_bytes() == env.read_bytes()
        assert stat.S_IMODE(copy.stat().st_mode) == 0o600


def test_a_copy_that_differs_is_refused(env, monkeypatch):
    real_write = os.write
    monkeypatch.setattr(settings_rules.os, "write", lambda fd, data: real_write(fd, data[:-1]))
    with pytest.raises(settings_rules.ConfigError) as exc:
        settings_rules.backup_env(str(env))
    assert exc.value.kind == "backup_failed"


def test_set_config_writes_nothing_when_the_backup_fails(env, tmp_path, monkeypatch):
    example = tmp_path / ".env.example"
    example.write_text("ENCRYPTION_KEY=\nLLAMA_PORT=8080\n")
    before = env.read_bytes()

    def fail(path):
        raise settings_rules.ConfigError("no backup", "backup_failed")

    monkeypatch.setattr(settings_rules, "backup_env", fail)
    with pytest.raises(settings_rules.ConfigError):
        settings_rules.set_config(str(env), str(example), "LLAMA_PORT", "9191")
    assert env.read_bytes() == before
