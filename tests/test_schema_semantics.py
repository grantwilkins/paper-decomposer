"""
Claim:
Schema models must enforce declared categorical contracts (enums and literal
tier keys), preserve recursive claim tree structure, allow forward-compatible
top-level config fields, and avoid shared mutable defaults across instances.

Plausible wrong implementations:
- Change enum/literal fields to plain strings and accept invalid categories.
- Break recursive `ClaimNode` typing so nested children are not parsed.
- Switch `PaperDecomposerConfig` from `extra="allow"` to rejecting extras.
- Use mutable defaults without `default_factory`, sharing state across objects.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from paper_decomposer.schema import (
    AppSettings,
    ClaimGroup,
    ClaimNode,
    CrossTypeOutput,
    PaperDecomposerConfig,
    PdfPipelineConfig,
    PipelineConfig,
    RawClaim,
    RuntimeModelConfig,
    RuntimePipelineConfig,
    TreeAssemblyOutput,
)


def _valid_config_dict() -> dict[str, Any]:
    return {
        "api": {"provider": "together", "base_url": "https://api.together.xyz/v1"},
        "models": {
            "small": {"model": "small", "temperature": 0.1, "max_tokens": 1024},
            "medium": {"model": "medium", "temperature": 0.2, "max_tokens": 2048},
            "heavy": {"model": "heavy", "temperature": 0.3, "max_tokens": 4096},
        },
        "pipeline": {
            "pdf": {"min_section_chars": 200, "max_section_chars": 12000},
            "seed": {},
            "section_extraction": {},
            "dedup": {},
            "tree": {},
            "output": {},
        },
    }


def test_raw_claim_rejects_invalid_enum_value() -> None:
    with pytest.raises(ValidationError):
        RawClaim(
            claim_id="m1",
            claim_type="METHOD",  # wrong case; should not coerce
            statement="Statement",
            source_section="1",
        )


def test_claim_node_parses_recursive_children() -> None:
    node = ClaimNode.model_validate(
        {
            "claim_id": "m1",
            "claim_type": "method",
            "abstraction_level": "system_realization",
            "semantic_role": "method_core",
            "canonical_label": "root_claim",
            "normalized_statement": "Root claim.",
            "result_subtype": None,
            "statement": "Root claim.",
            "depends_on": [],
            "children": [
                {
                    "claim_id": "r1",
                    "claim_type": "result",
                    "abstraction_level": "not_applicable",
                    "semantic_role": "scoped_result",
                    "canonical_label": "child_claim",
                    "normalized_statement": "Child claim.",
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
    assert node.children[0].depends_on == ["m1"]


def test_paper_decomposer_config_allows_top_level_future_fields() -> None:
    raw = _valid_config_dict()
    raw["future_feature"] = {"enabled": True}

    parsed = PaperDecomposerConfig.model_validate(raw)

    assert parsed.model_extra is not None
    assert parsed.model_extra["future_feature"] == {"enabled": True}


def test_mutable_defaults_are_not_shared_between_instances() -> None:
    first_claim = RawClaim(
        claim_id="m1",
        claim_type="method",
        statement="A",
        source_section="1",
    )
    second_claim = RawClaim(
        claim_id="m2",
        claim_type="method",
        statement="B",
        source_section="1",
    )
    first_pipeline = PipelineConfig(pdf=PdfPipelineConfig(min_section_chars=1, max_section_chars=2))
    second_pipeline = PipelineConfig(pdf=PdfPipelineConfig(min_section_chars=1, max_section_chars=2))

    first_claim.entity_names.append("PagedAttention")
    first_claim.evidence.append({"artifact_id": "fig_1", "role": "supports"})
    first_pipeline.seed["model_tier"] = "small"

    assert second_claim.entity_names == []
    assert second_claim.evidence == []
    assert second_pipeline.seed == {}


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


def test_tree_output_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TreeAssemblyOutput.model_validate(
            {
                "one_liner": {
                    "achieved": "higher throughput",
                    "via": "paged blocks",
                    "because": "less fragmentation",
                    "extra_detail": "should fail",
                },
                "nodes": [],
            }
        )

    with pytest.raises(ValidationError):
        TreeAssemblyOutput.model_validate(
            {
                "one_liner": {
                    "achieved": "higher throughput",
                    "via": "paged blocks",
                    "because": "less fragmentation",
                },
                "nodes": [
                    {
                        "claim_id": "m1",
                        "parent_id": None,
                        "depends_on": [],
                        "unexpected": "should fail",
                    }
                ],
            }
        )


def test_dedup_output_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ClaimGroup.model_validate(
            {
                "canonical_id": "m1",
                "member_ids": ["m2"],
                "parent_id": None,
                "extra": "should fail",
            }
        )

    with pytest.raises(ValidationError):
        CrossTypeOutput.model_validate(
            {
                "parent_child_links": [
                    {
                        "parent_id": "m1",
                        "child_id": "r1",
                        "relationship": "supports",
                        "note": "should fail",
                    }
                ]
            }
        )
