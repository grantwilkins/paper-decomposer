from __future__ import annotations

import re
from typing import Any

from ..schema import (
    AbstractionLevel,
    ClaimNode,
    ClaimType,
    EvidenceArtifact,
    FacetedClaim,
    OneLiner,
    PaperDecomposition,
    PaperMetadata,
    RawClaim,
    ResultSubtype,
    SemanticRole,
    SupportDetail,
)
from .dedup import classify_method_abstraction, classify_result_family, select_result_for_one_liner

TREE_SYSTEM_PROMPT = """Assemble a deterministic claim tree from a stable promoted argument set."""
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}
_LABEL_BLOCKLIST = {"paper", "system", "method", "approach", "baseline", "dataset", "results"}


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _tokens(text: str) -> list[str]:
    lowered = _clean_text(text).lower()
    return [
        token
        for token in _TOKEN_SPLIT_RE.split(lowered)
        if token and len(token) > 2 and token not in _STOPWORDS
    ]


def _normalize_statement(claim: RawClaim) -> str:
    tokens = _tokens(claim.statement)
    if claim.claim_type == ClaimType.method and classify_method_abstraction(claim) == "primitive":
        tokens = [token for token in tokens if token != "vllm"]
    return _clean_text(" ".join(tokens)) or _clean_text(claim.statement).lower()


def _canonical_label(claim: RawClaim, normalized_statement: str, used: set[str]) -> str:
    tokens = [token for token in _tokens(normalized_statement) if token not in _LABEL_BLOCKLIST]
    if claim.claim_type == ClaimType.method and "pagedattention" in tokens:
        tokens = ["pagedattention", *[token for token in tokens if token != "pagedattention"]]
    base = "_".join(tokens[:5]) or f"{claim.claim_type.value}_claim"
    label = base
    suffix = 2
    while label in used:
        label = f"{base}_{suffix}"
        suffix += 1
    used.add(label)
    return label


def _abstraction_level(claim: RawClaim) -> AbstractionLevel:
    if claim.claim_type == ClaimType.context:
        return AbstractionLevel.problem
    if claim.claim_type != ClaimType.method:
        return AbstractionLevel.not_applicable
    level = classify_method_abstraction(claim)
    if level == "primitive":
        return AbstractionLevel.primitive
    if level == "system_realization":
        return AbstractionLevel.system_realization
    return AbstractionLevel.submechanism


def _result_subtype(claim: RawClaim) -> ResultSubtype | None:
    if claim.claim_type != ClaimType.result:
        return None
    family = classify_result_family(claim)
    if family == "headline_comparative_performance":
        return ResultSubtype.headline_result
    if family == "constraint_observation":
        return ResultSubtype.constraint_observation
    if family == "decoding_mode_improvement":
        return ResultSubtype.mechanism_validation
    if family == "memory_mechanism_validation":
        return ResultSubtype.mechanism_validation
    return ResultSubtype.mechanism_validation


def _semantic_role(claim: RawClaim, abstraction: AbstractionLevel, subtype: ResultSubtype | None) -> SemanticRole:
    if claim.claim_type == ClaimType.context:
        return SemanticRole.problem
    if claim.claim_type == ClaimType.method:
        return SemanticRole.method_core if abstraction in {AbstractionLevel.primitive, AbstractionLevel.system_realization} else SemanticRole.method_support
    if claim.claim_type == ClaimType.result:
        return SemanticRole.headline_result if subtype == ResultSubtype.headline_result else SemanticRole.scoped_result
    if claim.claim_type == ClaimType.assumption:
        return SemanticRole.assumption
    return SemanticRole.limitation


def _one_liner(claims: list[RawClaim]) -> OneLiner:
    context = next((claim for claim in claims if claim.claim_type == ClaimType.context), None)
    methods = [claim for claim in claims if claim.claim_type == ClaimType.method]
    primitive = next((claim for claim in methods if classify_method_abstraction(claim) == "primitive"), None)
    system = next((claim for claim in methods if classify_method_abstraction(claim) == "system_realization"), None)
    result = select_result_for_one_liner(claims)
    via_claim = system or primitive or (methods[0] if methods else None)
    fallback = claims[0].statement if claims else "UNSPECIFIED"
    return OneLiner(
        achieved=_clean_text(result.statement if result is not None else fallback),
        via=_clean_text(via_claim.statement if via_claim is not None else fallback),
        because=_clean_text(context.statement if context is not None else fallback),
    )


def build_tree_prompt(
    metadata: PaperMetadata,
    claims: list[RawClaim],
    faceted: list[FacetedClaim],
    negatives: list[RawClaim],
    artifacts: list[EvidenceArtifact],
    claim_groups: list[Any] | None = None,
) -> list[dict[str, str]]:
    _ = (metadata, faceted, negatives, artifacts, claim_groups)
    lines = [f"[{claim.claim_id}] {claim.claim_type.value}: {claim.statement}" for claim in claims]
    body = "\n".join(lines) if lines else "(no claims)"
    return [
        {"role": "system", "content": TREE_SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]


def _build_parent_map(claims: list[RawClaim]) -> dict[str, str | None]:
    contexts = [claim for claim in claims if claim.claim_type == ClaimType.context]
    methods = [claim for claim in claims if claim.claim_type == ClaimType.method]
    primitive = next((claim for claim in methods if classify_method_abstraction(claim) == "primitive"), None)
    system = next((claim for claim in methods if classify_method_abstraction(claim) == "system_realization"), None)
    parent_by_id: dict[str, str | None] = {}

    root_context = contexts[0] if contexts else None
    for claim in contexts:
        parent_by_id[claim.claim_id] = None if claim is root_context else (root_context.claim_id if root_context else None)

    for claim in methods:
        level = classify_method_abstraction(claim)
        if level == "primitive":
            parent_by_id[claim.claim_id] = root_context.claim_id if root_context else None
        elif level == "system_realization":
            parent_by_id[claim.claim_id] = primitive.claim_id if primitive else (root_context.claim_id if root_context else None)
        else:
            parent_by_id[claim.claim_id] = (system or primitive or root_context).claim_id if (system or primitive or root_context) else None

    default_method = system or primitive or (methods[0] if methods else None)
    for claim in claims:
        if claim.claim_type == ClaimType.context or claim.claim_type == ClaimType.method:
            continue
        if default_method is not None:
            parent_by_id[claim.claim_id] = default_method.claim_id
        else:
            parent_by_id[claim.claim_id] = root_context.claim_id if root_context else None
    return parent_by_id


async def assemble_tree_deterministic(
    metadata: PaperMetadata,
    claims: list[RawClaim],
    faceted: list[FacetedClaim],
    negatives: list[RawClaim],
    artifacts: list[EvidenceArtifact],
    config: Any,
    claim_groups: list[Any] | None = None,
    support_details: list[SupportDetail] | None = None,
) -> PaperDecomposition:
    _ = (config, claim_groups)
    combined = list(claims)
    seen = {claim.claim_id for claim in combined}
    for negative in negatives:
        if negative.claim_id not in seen:
            seen.add(negative.claim_id)
            combined.append(negative)

    if not any(claim.claim_type == ClaimType.context for claim in combined):
        raise ValueError("deterministic tree assembly requires at least one context claim")

    faceted_by_id = {item.claim.claim_id: item for item in faceted}
    parent_by_id = _build_parent_map(combined)
    children_by_parent: dict[str, list[RawClaim]] = {}
    for claim in combined:
        parent_id = parent_by_id.get(claim.claim_id)
        if parent_id is None:
            continue
        children_by_parent.setdefault(parent_id, []).append(claim)

    used_labels: set[str] = set()

    def make_node(claim: RawClaim) -> ClaimNode:
        abstraction = _abstraction_level(claim)
        subtype = _result_subtype(claim)
        normalized_statement = _normalize_statement(claim)
        node = ClaimNode(
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            abstraction_level=abstraction,
            semantic_role=_semantic_role(claim, abstraction, subtype),
            canonical_label=_canonical_label(claim, normalized_statement, used_labels),
            normalized_statement=normalized_statement,
            result_subtype=subtype,
            statement=claim.statement,
            evidence=list(claim.evidence),
            facets=faceted_by_id.get(claim.claim_id),
            children=[make_node(child) for child in children_by_parent.get(claim.claim_id, [])],
            depends_on=[parent_by_id[claim.claim_id]] if parent_by_id.get(claim.claim_id) is not None else [],
            rejected_what=claim.rejected_what,
            rejected_why=claim.rejected_why,
        )
        return node

    roots = [make_node(claim) for claim in combined if parent_by_id.get(claim.claim_id) is None]
    return PaperDecomposition(
        metadata=metadata,
        one_liner=_one_liner(combined),
        claim_tree=roots,
        support_details=list(support_details or []),
        all_artifacts=list(artifacts),
    )


async def assemble_tree(
    metadata: PaperMetadata,
    claims: list[RawClaim],
    faceted: list[FacetedClaim],
    negatives: list[RawClaim],
    artifacts: list[EvidenceArtifact],
    config: Any,
    claim_groups: list[Any] | None = None,
    support_details: list[SupportDetail] | None = None,
) -> PaperDecomposition:
    return await assemble_tree_deterministic(
        metadata=metadata,
        claims=claims,
        faceted=faceted,
        negatives=negatives,
        artifacts=artifacts,
        config=config,
        claim_groups=claim_groups,
        support_details=support_details,
    )


__all__ = [
    "TREE_SYSTEM_PROMPT",
    "assemble_tree",
    "assemble_tree_deterministic",
    "build_tree_prompt",
]
