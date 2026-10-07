"""Unit tests for app/security/encryption.py round-trip."""

import pytest
from cryptography.fernet import Fernet


def test_round_trip():
    from app.security.encryption import decrypt_value, encrypt_value

    plaintext = "someone@example.com"
    token = encrypt_value(plaintext)
    assert token != plaintext
    assert decrypt_value(token) == plaintext


def test_decrypt_corrupted_token_raises_value_error():
    from app.security.encryption import decrypt_value, encrypt_value

    token = encrypt_value("test")
    corrupted = token[:-4] + "abcd"
    with pytest.raises(ValueError):
        decrypt_value(corrupted)


def test_decrypt_with_wrong_key_raises_value_error(monkeypatch):
    from app import config
    from app.security import encryption

    token = encryption.encrypt_value("test")

    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    config.get_settings.cache_clear()
    encryption._fernet.cache_clear()
    try:
        with pytest.raises(ValueError):
            encryption.decrypt_value(token)
    finally:
        # Restore, so later tests in the same process see the original key again.
        config.get_settings.cache_clear()
        encryption._fernet.cache_clear()
