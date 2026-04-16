"""
Claim:
Schema models should keep top-level config extensibility while rejecting nested
config drift, preserving strict enum contracts and recursive tree parsing.

Plausible wrong implementations:
- Allow unknown nested config keys to pass silently.
- Break recursive `ClaimNode` parsing.
- Loosen enum validation on `RawClaim.claim_type`.
- Reject future top-level config keys even though the top-level raw config is extensible.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from paper_decomposer.schema import AppSettings, ClaimNode, PaperDecomposerConfig, PdfPipelineConfig, PipelineConfig, RawClaim, RuntimeModelConfig, RuntimePipelineConfig


def _valid_config_dict() -> dict[str, Any]:
    return {
        "api": {"provider": "together", "base_url": "https://api.together.xyz/v1"},
        "models": {
            "small": {"model": "small", "temperature": 0.1, "max_tokens": 32},
            "medium": {"model": "medium", "temperature": 0.1, "max_tokens": 32},
            "heavy": {"model": "heavy", "temperature": 0.1, "max_tokens": 32},
        },
        "pipeline": {
            "pdf": {"min_section_chars": 1, "max_section_chars": 10},
            "seed": {},
            "section_extraction": {},
            "dedup": {},
            "tree": {},
            "output": {},
        },
    }


def test_raw_claim_rejects_invalid_enum_value() -> None:
    with pytest.raises(ValidationError):
        RawClaim(claim_id="m1", claim_type="METHOD", statement="x", source_section="1")


def test_claim_node_parses_recursive_children() -> None:
    node = ClaimNode.model_validate(
        {
            "claim_id": "m1",
            "claim_type": "method",
            "abstraction_level": "system_realization",
            "semantic_role": "method_core",
            "canonical_label": "root_claim",
            "normalized_statement": "root claim",
            "statement": "Root claim.",
            "children": [
                {
                    "claim_id": "r1",
                    "claim_type": "result",
                    "abstraction_level": "not_applicable",
                    "semantic_role": "scoped_result",
                    "canonical_label": "child_claim",
                    "normalized_statement": "child claim",
                    "result_subtype": "mechanism_validation",
                    "statement": "Child claim.",
                    "depends_on": ["m1"],
                    "children": [],
                }
            ],
        }
    )
    assert len(node.children) == 1
    assert isinstance(node.children[0], ClaimNode)
    assert node.children[0].claim_id == "r1"


def test_paper_decomposer_config_allows_top_level_future_fields() -> None:
    raw = _valid_config_dict()
    raw["future_feature"] = {"enabled": True}
    parsed = PaperDecomposerConfig.model_validate(raw)
    assert parsed.model_extra is not None
    assert parsed.model_extra["future_feature"] == {"enabled": True}


def test_nested_config_unknown_keys_are_rejected() -> None:
    raw = _valid_config_dict()
    raw["api"]["unknown_api_key"] = True
    with pytest.raises(ValidationError):
        PaperDecomposerConfig.model_validate(raw)


def test_app_settings_rejects_nonstandard_model_tier_keys() -> None:
    with pytest.raises(ValidationError):
        AppSettings(
            config_path="config.yaml",
            api_key="test-key",
            model_tiers={"tiny": RuntimeModelConfig(model="x", temperature=0.1, max_tokens=1)},
            pipeline=RuntimePipelineConfig(
                parser="pymupdf",
                extract_captions=False,
                extract_equations=False,
                min_section_chars=10,
                max_section_chars=100,
            ),
            raw=PaperDecomposerConfig.model_validate(_valid_config_dict()),
        )


def test_mutable_defaults_are_not_shared_between_instances() -> None:
    first_pipeline = PipelineConfig(pdf=PdfPipelineConfig(min_section_chars=1, max_section_chars=2))
    second_pipeline = PipelineConfig(pdf=PdfPipelineConfig(min_section_chars=1, max_section_chars=2))
    first_pipeline.seed["model_tier"] = "small"
    assert second_pipeline.seed == {}
