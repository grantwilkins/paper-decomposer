from __future__ import annotations

from pathlib import Path

from psycopg_pool import AsyncConnectionPool

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class PaperDecomposerDB:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str, *, min_size: int = 1, max_size: int = 8) -> "PaperDecomposerDB":
        pool = AsyncConnectionPool(conninfo=dsn, min_size=min_size, max_size=max_size, open=False)
        await pool.open()
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def apply_schema(self) -> None:
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql)


__all__ = ["PaperDecomposerDB"]
