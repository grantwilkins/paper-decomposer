from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import AppSettings, PaperDecomposerConfig, RuntimeModelConfig, RuntimePipelineConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class ConfigError(RuntimeError):
    pass


def load_config(config_path: str | Path | None = None) -> AppSettings:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must contain a top-level mapping: {path}")

    try:
        parsed = PaperDecomposerConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config data in {path}: {exc}") from exc

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ConfigError("Missing Together API key. Set TOGETHER_API_KEY in your environment.")

    model_tiers = {
        "small": RuntimeModelConfig(
            model=parsed.models.small.model,
            temperature=parsed.models.small.temperature,
            max_tokens=parsed.models.small.max_tokens,
        ),
        "medium": RuntimeModelConfig(
            model=parsed.models.medium.model,
            temperature=parsed.models.medium.temperature,
            max_tokens=parsed.models.medium.max_tokens,
        ),
        "heavy": RuntimeModelConfig(
            model=parsed.models.heavy.model,
            temperature=parsed.models.heavy.temperature,
            max_tokens=parsed.models.heavy.max_tokens,
        ),
    }

    pipeline = RuntimePipelineConfig(
        parser=parsed.pipeline.pdf.parser,
        extract_captions=parsed.pipeline.pdf.extract_captions,
        extract_equations=parsed.pipeline.pdf.extract_equations,
        min_section_chars=parsed.pipeline.pdf.min_section_chars,
        max_section_chars=parsed.pipeline.pdf.max_section_chars,
        seed=parsed.pipeline.seed,
        section_extraction=parsed.pipeline.section_extraction,
        dedup=parsed.pipeline.dedup,
        tree=parsed.pipeline.tree,
        output=parsed.pipeline.output,
    )

    return AppSettings(
        config_path=str(path),
        api_key=api_key,
        model_tiers=model_tiers,
        pipeline=pipeline,
        raw=parsed,
    )


@lru_cache(maxsize=1)
def get_config(config_path: str | Path | None = None) -> AppSettings:
    return load_config(config_path)
