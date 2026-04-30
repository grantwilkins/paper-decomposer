"""
Smoke test for the Postgres schema. We don't spin up a live DB here — that
needs Docker and belongs in a separate integration suite — but we do verify
the file exists, declares the required extensions, and contains every table
the client refers to. Catches typos and accidental file moves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_decomposer.db.client import _SCHEMA_PATH


REQUIRED_EXTENSIONS = ("pgcrypto", "pg_trgm", "vector")
REQUIRED_TABLES = (
    "papers",
    "methods",
    "method_aliases",
    "method_edges",
    "settings",
    "setting_edges",
    "outcomes",
    "claims",
    "claim_links",
    "claim_evidence",
)


@pytest.fixture(scope="module")
def schema_sql() -> str:
    assert _SCHEMA_PATH.exists(), f"schema.sql missing at {_SCHEMA_PATH}"
    return _SCHEMA_PATH.read_text(encoding="utf-8")


def test_schema_declares_required_extensions(schema_sql: str) -> None:
    for ext in REQUIRED_EXTENSIONS:
        assert f"CREATE EXTENSION IF NOT EXISTS {ext}" in schema_sql, (
            f"schema.sql is missing required extension: {ext}"
        )


def test_schema_creates_every_required_table(schema_sql: str) -> None:
    for table in REQUIRED_TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in schema_sql, (
            f"schema.sql is missing required table: {table}"
        )


def test_schema_path_is_inside_db_package() -> None:
    assert _SCHEMA_PATH.parent.name == "db"
    assert _SCHEMA_PATH.name == "schema.sql"
    assert isinstance(_SCHEMA_PATH, Path)
