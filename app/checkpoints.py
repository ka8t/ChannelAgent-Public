"""Persistent, encrypted LangGraph checkpoints.

Conversations used to live in a process-wide MemorySaver, so every restart
erased every user's history. They now live in a SQLite file, and the
payloads are encrypted with the project's own ENCRYPTION_KEY (Fernet), the
same protection ActionLog.text gets, because a checkpoint holds the whole
conversation. Only the thread id (channel, identity key, agent id) and the
checkpoint ids stay readable, and an email identity is already a hash.

The file is separate from the application database on purpose: Alembic
keeps owning only our tables, the checkpointer creates and upgrades its own.
"""

import logging
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import get_settings
from app.db.session import sqlite_file_path
from app.security.encryption import decrypt_bytes, encrypt_bytes

logger = logging.getLogger("channelagent")

CIPHER_NAME = "fernet"


class FernetCipher:
    """LangGraph's CipherProtocol on top of app.security.encryption."""

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return CIPHER_NAME, encrypt_bytes(plaintext)

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if ciphername != CIPHER_NAME:
            raise ValueError(f"Unsupported checkpoint cipher: {ciphername}")
        return decrypt_bytes(ciphertext)


def checkpoint_db_path() -> Path:
    settings = get_settings()
    if settings.checkpoint_db_path and settings.checkpoint_db_path.strip():
        return Path(settings.checkpoint_db_path.strip())
    main = sqlite_file_path(settings.database_url)
    return (main.parent if main is not None else Path("data")) / "checkpoints.db"


async def open_saver(path: Path) -> AsyncSqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(path)
    saver = AsyncSqliteSaver(conn, serde=EncryptedSerializer(FernetCipher()))
    await saver.setup()
    return saver
