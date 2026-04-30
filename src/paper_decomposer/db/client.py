from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg
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

    # ── Write paths (skeletons; filled in by the next PR) ──────────────

    async def upsert_paper(
        self,
        *,
        title: str,
        arxiv_id: str | None = None,
        doi: str | None = None,
        authors: list[str] | None = None,
        year: int | None = None,
        venue: str | None = None,
        source_pdf_sha256: str | None = None,
    ) -> UUID:
        raise NotImplementedError

    async def upsert_method(
        self,
        *,
        canonical_name: str,
        description: str | None = None,
        canonical_parent_id: UUID | None = None,
        embedding: list[float] | None = None,
    ) -> UUID:
        raise NotImplementedError

    async def add_method_edge(
        self,
        *,
        parent_id: UUID,
        child_id: UUID,
        relation: str,
        confidence: float = 1.0,
    ) -> None:
        raise NotImplementedError

    async def upsert_setting(
        self,
        *,
        kind: str,
        canonical_name: str,
        description: str | None = None,
        canonical_parent_id: UUID | None = None,
        embedding: list[float] | None = None,
    ) -> UUID:
        raise NotImplementedError

    async def upsert_outcome(
        self,
        *,
        paper_id: UUID,
        method_id: UUID | None,
        setting_id: UUID | None,
        metric_name: str,
        value: float | None = None,
        delta_value: float | None = None,
        baseline_method_id: UUID | None = None,
        units: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        raise NotImplementedError

    async def insert_claim(
        self,
        *,
        paper_id: UUID,
        claim_type: str,
        statement: str,
        strength: float | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        raise NotImplementedError


__all__ = ["PaperDecomposerDB"]


# Re-export psycopg so callers can catch its errors without an extra import.
DatabaseError = psycopg.Error
