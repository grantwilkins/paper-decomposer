from pathlib import Path

import pytest

from paper_decomposer.config import ConfigError, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_config_loads_without_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")

    settings = load_config(CONFIG_PATH)

    assert settings.pipeline.min_section_chars > 0
    assert settings.pipeline.max_section_chars > settings.pipeline.min_section_chars
    assert settings.raw.database.dsn_env


def test_all_three_model_tiers_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")

    settings = load_config(CONFIG_PATH)

    assert set(settings.model_tiers.keys()) == {"small", "medium", "heavy"}
    for tier in ("small", "medium", "heavy"):
        tier_config = settings.model_tiers[tier]
        assert tier_config.model
        assert tier_config.max_tokens > 0


def test_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="TOGETHER_API_KEY"):
        load_config(CONFIG_PATH)
