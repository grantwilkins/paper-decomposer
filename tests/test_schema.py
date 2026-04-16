"""
Claim:
`PaperDecomposition` should reject structurally invalid trees: duplicate claim
IDs, duplicate canonical labels, and dependency edges that point to descendant
nodes.

Plausible wrong implementations:
- Only validate canonical labels and ignore duplicate claim IDs.
- Permit a node to depend on its own descendant.
- Accept two nodes with the same canonical label in different branches.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_decomposer.schema import AbstractionLevel, ClaimNode, ClaimType, OneLiner, PaperDecomposition, PaperMetadata, ResultSubtype, SemanticRole


def _result_node(claim_id: str, canonical_label: str, *, depends_on: list[str]) -> ClaimNode:
    return ClaimNode(
        claim_id=claim_id,
        claim_type=ClaimType.result,
        abstraction_level=AbstractionLevel.not_applicable,
        semantic_role=SemanticRole.headline_result,
        canonical_label=canonical_label,
        normalized_statement=canonical_label.replace("_", " "),
        result_subtype=ResultSubtype.headline_result,
        statement=canonical_label,
        depends_on=depends_on,
        children=[],
    )


def test_paper_decomposition_rejects_duplicate_claim_ids() -> None:
    duplicate = ClaimNode(
        claim_id="M1",
        claim_type=ClaimType.method,
        abstraction_level=AbstractionLevel.primitive,
        semantic_role=SemanticRole.method_core,
        canonical_label="pagedattention_method",
        normalized_statement="pagedattention method",
        statement="PagedAttention method.",
        children=[],
    )
    root = ClaimNode(
        claim_id="C1",
        claim_type=ClaimType.context,
        abstraction_level=AbstractionLevel.problem,
        semantic_role=SemanticRole.problem,
        canonical_label="fragmentation_problem",
        normalized_statement="fragmentation problem",
        statement="Fragmentation problem.",
        children=[duplicate, duplicate.model_copy(update={"canonical_label": "pagedattention_method_2"})],
        depends_on=[],
    )
    with pytest.raises(ValidationError, match="claim_id must be unique"):
        PaperDecomposition(metadata=PaperMetadata(title="t"), one_liner=OneLiner(achieved="a", via="b", because="c"), claim_tree=[root])


def test_paper_decomposition_rejects_descendant_dependencies() -> None:
    child = _result_node("R1", "headline_result", depends_on=["C1"])
    root = ClaimNode(
        claim_id="C1",
        claim_type=ClaimType.context,
        abstraction_level=AbstractionLevel.problem,
        semantic_role=SemanticRole.problem,
        canonical_label="fragmentation_problem",
        normalized_statement="fragmentation problem",
        statement="Fragmentation problem.",
        children=[child],
        depends_on=["R1"],
    )
    with pytest.raises(ValidationError, match="descendant"):
        PaperDecomposition(metadata=PaperMetadata(title="t"), one_liner=OneLiner(achieved="a", via="b", because="c"), claim_tree=[root])


def test_paper_decomposition_rejects_duplicate_canonical_labels() -> None:
    root = ClaimNode(
        claim_id="C1",
        claim_type=ClaimType.context,
        abstraction_level=AbstractionLevel.problem,
        semantic_role=SemanticRole.problem,
        canonical_label="shared_label",
        normalized_statement="shared label",
        statement="Problem.",
        children=[
            ClaimNode(
                claim_id="M1",
                claim_type=ClaimType.method,
                abstraction_level=AbstractionLevel.primitive,
                semantic_role=SemanticRole.method_core,
                canonical_label="shared_label",
                normalized_statement="shared label method",
                statement="Method.",
                children=[],
            )
        ],
        depends_on=[],
    )
    with pytest.raises(ValidationError, match="canonical_label must be unique"):
        PaperDecomposition(metadata=PaperMetadata(title="t"), one_liner=OneLiner(achieved="a", via="b", because="c"), claim_tree=[root])
