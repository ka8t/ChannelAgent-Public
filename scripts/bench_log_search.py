"""Time `search_action_logs` at 1k, 10k and 100k rows.

Repeatable: builds a throwaway database in a temp directory (never touches
data/), fills it with realistic encrypted rows (about 300 characters each,
Fernet-encrypted like real ones), then times two cases on the service
function the API and the console both call:

  best  : a keyword present in every row (the 20 wanted matches are the
          newest 20 rows, so the search stops after the first batch)
  worst : a keyword that matches nothing (decrypts every row)

A keyword found in only a few rows costs as much as the worst case: the
search keeps reading until it has `limit` matches or runs out of rows.

Usage:  .venv/bin/python scripts/bench_log_search.py [rows ...]
        (default rows: 1000 10000 100000)
"""

import asyncio
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from cryptography.fernet import Fernet

REPEATS = 3
CHUNK = 5000
_SENTENCE = (
    "Bonjour, pouvez-vous me rappeler les points de la reunion de mardi et "
    "preparer un resume pour l'equipe, avec les decisions et les prochaines etapes ? "
)


def _text(i: int) -> str:
    return f"message {i}: " + (_SENTENCE * 2)[:290]


async def _bench(rows: int, tmp: Path) -> dict:
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/bench-{rows}.db"
    os.environ["CHECKPOINT_DB_PATH"] = str(tmp / f"checkpoints-{rows}.db")
    os.environ["MIGRATION_BACKUPS_KEEP"] = "0"
    os.environ.setdefault("ENCRYPTION_KEY", Fernet.generate_key().decode())

    from sqlalchemy import insert

    from app.admin import service
    from app.config import get_settings
    from app.db.models import ActionLog, Channel, Direction
    from app.db.session import get_engine, get_sessionmaker, init_db, session_scope

    for cached in (get_settings, get_engine, get_sessionmaker):
        cached.cache_clear()
    await init_db()

    async with session_scope() as session:
        user = await service.create_user(session, "bench")
        agent = await service.get_or_create_default_agent(session, user.id)
        await session.commit()
        user_id, agent_id = user.id, agent.id

    started = time.perf_counter()
    async with session_scope() as session:
        for start in range(0, rows, CHUNK):
            await session.execute(
                insert(ActionLog),
                [
                    {
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "channel": Channel.TELEGRAM,
                        "direction": Direction.INBOUND if i % 2 == 0 else Direction.OUTBOUND,
                        "text": _text(i),
                    }
                    for i in range(start, min(start + CHUNK, rows))
                ],
            )
        await session.commit()
    fill = time.perf_counter() - started

    async def timed(keyword: str) -> tuple[float, int]:
        samples, found = [], 0
        for _ in range(REPEATS):
            async with session_scope() as session:
                t0 = time.perf_counter()
                result = await service.search_action_logs(session, keyword=keyword, limit=20)
                samples.append(time.perf_counter() - t0)
                found = len(result)
                await session.rollback()
        return statistics.median(samples), found

    best, best_found = await timed("message")
    worst, worst_found = await timed("zzz-matches-nothing")
    await get_engine().dispose()
    return {
        "rows": rows,
        "fill_s": fill,
        "best_s": best,
        "best_found": best_found,
        "worst_s": worst,
        "worst_found": worst_found,
    }


async def main(sizes: list[int]) -> None:
    from app import graph

    with tempfile.TemporaryDirectory(prefix="bench-log-search-") as tmp:
        print(f"Python {sys.version.split()[0]}, median of {REPEATS} runs, limit=20")
        print(f"{'rows':>8} {'best case':>12} {'worst case':>12}   (fill time)")
        for rows in sizes:
            r = await _bench(rows, Path(tmp))
            assert r["best_found"] == 20 and r["worst_found"] == 0, r
            print(
                f"{r['rows']:>8} {r['best_s'] * 1000:>9.1f} ms {r['worst_s'] * 1000:>9.1f} ms"
                f"   ({r['fill_s']:.1f} s)"
            )
    await graph.close_graph()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    asyncio.run(main([int(a) for a in sys.argv[1:]] or [1000, 10000, 100000]))
