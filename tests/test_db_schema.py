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
    "extraction_runs",
    "evidence_spans",
    "methods",
    "method_aliases",
    "method_edges",
    "settings",
    "setting_edges",
    "method_setting_links",
    "outcomes",
    "claims",
    "claim_links",
    "claim_evidence",
    "evidence_links",
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


def test_schema_supports_paper_local_extraction_provenance(schema_sql: str) -> None:
    assert "extraction_run_id" in schema_sql
    assert "local_node_id" in schema_sql
    assert "local_setting_id" in schema_sql
    assert "local_span_id" in schema_sql
    assert "model_artifact" in schema_sql
    assert "method_setting_links" in schema_sql
    assert "evidence_links" in schema_sql
