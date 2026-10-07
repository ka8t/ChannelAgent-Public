"""SQLAlchemy column type that encrypts on write and decrypts on read.

Models use EncryptedString for any sensitive field
(raw email address, Matrix access token, other personal metadata)
instead of remembering to call encrypt_value/decrypt_value by hand at
every call site.
"""

import logging
import time

from sqlalchemy.types import String, TypeDecorator

from app.security.encryption import decrypt_value, encrypt_value

logger = logging.getLogger("channelagent")

UNDECRYPTABLE_MARKER = "<undecryptable>"

# A wrong key makes every value of every row unreadable, so the warning is
# limited to one per interval, saying how many were left out.
WARNING_INTERVAL_SECONDS = 60.0
_last_warning: float | None = None
_suppressed = 0


def reset_warning_state() -> None:
    global _last_warning, _suppressed
    _last_warning, _suppressed = None, 0


def _warn_undecryptable() -> None:
    global _last_warning, _suppressed
    now = time.monotonic()
    if _last_warning is not None and now - _last_warning < WARNING_INTERVAL_SECONDS:
        _suppressed += 1
        return
    extra = f" ({_suppressed} similar messages suppressed)" if _suppressed else ""
    logger.warning(
        "Cannot decrypt a stored value (wrong ENCRYPTION_KEY or a corrupted row); "
        "showing %s. Count them with the storage overview.%s",
        UNDECRYPTABLE_MARKER,
        extra,
    )
    _last_warning, _suppressed = now, 0


class UndecryptableText(str):
    """What an encrypted column yields when its value cannot be decrypted:
    a corrupted row, a restored backup made with another key, a rotated key.
    A str, so every caller keeps working, but a distinct type so code can tell
    "the stored value was unreadable" from a real message that happens to say
    `<undecryptable>`: a keyword search never matches it.
    """

    __slots__ = ()


class EncryptedString(TypeDecorator):
    """A TEXT column that transparently encrypts/decrypts its Python value.

    A Fernet token is longer than its plaintext and varies in length
    with the key/nonce, so this is stored as unbounded TEXT regardless
    of the logical field's plaintext length.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect) -> str | None:
        if value is None:
            return None
        return encrypt_value(value)

    def process_result_value(self, value: str | None, dialect) -> str | None:
        if value is None:
            return None
        try:
            return decrypt_value(value)
        except ValueError:
            # One bad row must not make every query that touches the table
            # fail. Never logs the ciphertext. Nothing writes the marker back:
            # SQLAlchemy only writes attributes that changed after loading.
            _warn_undecryptable()
            return UndecryptableText(UNDECRYPTABLE_MARKER)
