"""
Claim:
Config loading must reject malformed/invalid inputs, preserve per-tier model
settings and pipeline PDF settings exactly, and cache settings via `get_config`
for the same path until cache clear.

Plausible wrong implementations:
- Parse YAML but accept non-mapping top-level structures.
- Map model tier values from the wrong source tier.
- Read pipeline limits/flags from the wrong config level.
- Leak raw Pydantic exceptions instead of wrapping in ConfigError.
- Remove or bypass `get_config` caching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from paper_decomposer.config import ConfigError, get_config, load_config


@pytest.fixture(autouse=True)
def _clear_config_cache() -> None:
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def _valid_config_dict() -> dict[str, Any]:
    return {
        "api": {
            "provider": "together",
            "base_url": "https://api.together.xyz/v1",
            "env_key_var": "TOGETHER_API_KEY",
            "max_retries": 3,
            "retry_backoff_base": 2.0,
        },
        "models": {
            "small": {"model": "small-model", "temperature": 0.11, "max_tokens": 1111},
            "medium": {"model": "medium-model", "temperature": 0.22, "max_tokens": 2222},
            "heavy": {"model": "heavy-model", "temperature": 0.33, "max_tokens": 3333},
        },
        "pipeline": {
            "pdf": {
                "parser": "pymupdf",
                "extract_captions": True,
                "extract_equations": False,
                "min_section_chars": 250,
                "max_section_chars": 7500,
            },
            "extraction": {},
        },
    }


def _write_yaml(tmp_path: Path, data: Any) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def test_load_config_preserves_distinct_model_tier_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    config_path = _write_yaml(tmp_path, _valid_config_dict())

    settings = load_config(config_path)

    assert settings.tier("small").model == "small-model"
    assert settings.tier("medium").model == "medium-model"
    assert settings.tier("heavy").model == "heavy-model"
    assert settings.tier("small").temperature == pytest.approx(0.11)
    assert settings.tier("medium").temperature == pytest.approx(0.22)
    assert settings.tier("heavy").temperature == pytest.approx(0.33)
    assert settings.tier("small").max_tokens == 1111
    assert settings.tier("medium").max_tokens == 2222
    assert settings.tier("heavy").max_tokens == 3333


def test_load_config_preserves_pdf_pipeline_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    raw = _valid_config_dict()
    raw["pipeline"]["pdf"] = {
        "parser": "custom-parser",
        "extract_captions": False,
        "extract_equations": True,
        "min_section_chars": 321,
        "max_section_chars": 6543,
    }
    config_path = _write_yaml(tmp_path, raw)

    settings = load_config(config_path)

    assert settings.pipeline.parser == "custom-parser"
    assert settings.pipeline.extract_captions is False
    assert settings.pipeline.extract_equations is True
    assert settings.pipeline.min_section_chars == 321
    assert settings.pipeline.max_section_chars == 6543


def test_load_config_rejects_non_mapping_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    config_path = _write_yaml(tmp_path, ["not", "a", "mapping"])

    with pytest.raises(ConfigError, match="top-level mapping"):
        load_config(config_path)


def test_load_config_wraps_model_validation_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    raw = _valid_config_dict()
    del raw["models"]["heavy"]
    config_path = _write_yaml(tmp_path, raw)

    with pytest.raises(ConfigError, match="Invalid config data"):
        load_config(config_path)


def test_get_config_caches_by_path_until_cache_clear(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_yaml(tmp_path, _valid_config_dict())
    monkeypatch.setenv("TOGETHER_API_KEY", "first-key")

    first = get_config(config_path)

    monkeypatch.setenv("TOGETHER_API_KEY", "second-key")
    second = get_config(config_path)

    assert second is first
    assert second.api_key == "first-key"

    get_config.cache_clear()
    third = get_config(config_path)

    assert third is not first
    assert third.api_key == "second-key"
