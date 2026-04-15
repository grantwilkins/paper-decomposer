from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

from ..models import call_model
from ..schema import (
    ClaimLocalRole,
    ClaimGroup,
    ClaimNode,
    ClaimType,
    EvidenceArtifact,
    FacetedClaim,
    ModelTier,
    OneLiner,
    PaperDecomposition,
    PaperMetadata,
    ParentPreference,
    RawClaim,
    TreeAssemblyOutput,
    TreeNodeAssignment,
)

TREE_SYSTEM_PROMPT = """You are an expert research analyst. You
receive a structured extraction from a research paper and your job
is to assemble it into a coherent claim tree.

A claim tree has this structure:
- CONTEXT claims are roots (the problems being addressed)
- METHOD claims are children of the context claims they address
- METHOD sub-claims are children of their parent method claims
- RESULT claims are children of the method claims they validate
- ASSUMPTION claims attach to the claims that require them
- NEGATIVE claims attach to the method claims they're about

Hard parent grammar (must follow):
- CONTEXT parent: null or CONTEXT
- METHOD parent: CONTEXT or METHOD
- RESULT parent: METHOD
- ASSUMPTION parent: METHOD or RESULT (context only if no better anchor exists)
- NEGATIVE parent: METHOD or CONTEXT

Dependency rules:
1. Every non-CONTEXT claim should have a non-null parent_id whenever any
   compatible parent exists in the provided claims.
2. Every RESULT claim must depend on at least one METHOD claim.
   Ask: "if this method didn't exist, would this result hold?"
   Choose the MOST SPECIFIC method that explains the result.
   Do not default everything to the top-level method.
3. Every METHOD claim must address at least one CONTEXT claim.
   Ask: "what problem does this mechanism solve?"
4. Avoid star graphs. If many claims currently attach to one node,
   introduce hierarchy with method sub-claims and method-specific results.
5. Low-level implementation mechanisms (allocator/kernels/schedulers/tables)
   should usually attach under a METHOD parent, not directly under broad context.
6. If a RESULT's evidence artifact is in the same section as a
   METHOD discussion, there is likely a dependency.
7. If two claims mention the same entity_name, there is likely
   a dependency.
8. NEGATIVE claims of type "rejected_alternative" depend on the
   CONTEXT claim that motivated the rejected approach.
9. NEGATIVE claims of type "limitation" depend on the METHOD
   claim they limit.
10. Related-work positioning claims should usually be grouped under a
   small number of higher-level CONTEXT parents, not all as roots.
11. If CANONICALIZATION LINKS are provided, treat them as strong
    defaults unless claim content clearly contradicts them.
12. If a claim line includes [hints: ...], treat these as soft defaults
    for local role and preferred parent type unless contradicted.

Output JSON with:
- one_liner: {achieved, via, because}
- nodes: [{claim_id, parent_id, depends_on}]
Return JSON only.
"""

_VALID_TIERS: set[ModelTier] = {"small", "medium", "heavy"}


def _resolve_model_tier(config: Any) -> ModelTier:
    if config is None:
        return "heavy"

    pipeline = getattr(config, "pipeline", None)
    if pipeline is not None:
        tree_cfg = getattr(pipeline, "tree", None)
        if isinstance(tree_cfg, Mapping):
            tier = tree_cfg.get("model_tier")
            if isinstance(tier, str) and tier in _VALID_TIERS:
                return cast(ModelTier, tier)

    if isinstance(config, Mapping):
        pipeline_cfg = config.get("pipeline")
        if isinstance(pipeline_cfg, Mapping):
            tree_cfg = pipeline_cfg.get("tree")
            if isinstance(tree_cfg, Mapping):
                tier = tree_cfg.get("model_tier")
                if isinstance(tier, str) and tier in _VALID_TIERS:
                    return cast(ModelTier, tier)

    return "heavy"


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


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

_ALLOWED_PARENT_TYPES: dict[ClaimType, set[ClaimType]] = {
    ClaimType.context: {ClaimType.context},
    ClaimType.method: {ClaimType.context, ClaimType.method},
    ClaimType.result: {ClaimType.method},
    ClaimType.assumption: {ClaimType.method, ClaimType.result, ClaimType.context},
    ClaimType.negative: {ClaimType.method, ClaimType.context},
}

_LOW_LEVEL_METHOD_TOKENS = {
    "allocator",
    "allocation",
    "batching",
    "block",
    "cache",
    "cuda",
    "decode",
    "evict",
    "gather",
    "kernel",
    "mapping",
    "policy",
    "prefill",
    "preempt",
    "reclaim",
    "runtime",
    "scheduler",
    "swap",
    "table",
}

_HIGH_LEVEL_METHOD_HINTS = (
    "we introduce",
    "we propose",
    "we present",
    "our system",
    "our approach",
    "this paper",
)


def _claim_tokens(text: str) -> set[str]:
    lowered = _clean_text(text).lower()
    return {
        token
        for token in _TOKEN_SPLIT_RE.split(lowered)
        if token and len(token) > 2 and token not in _STOPWORDS
    }


def _evidence_ids(claim: RawClaim) -> set[str]:
    return {pointer.artifact_id.strip().lower() for pointer in claim.evidence if pointer.artifact_id.strip()}


def _related_work_source(source_section: str) -> bool:
    lowered = source_section.strip().lower()
    return any(
        key in lowered
        for key in ("related work", "background", "prior work", "preliminary", "preliminaries")
    )


def _method_affinity_score(claim: RawClaim, method: RawClaim) -> int:
    score = 0
    claim_evidence = _evidence_ids(claim)
    method_evidence = _evidence_ids(method)
    overlap_evidence = claim_evidence & method_evidence
    if overlap_evidence:
        score += 8 + len(overlap_evidence)

    claim_source = claim.source_section.strip().lower()
    method_source = method.source_section.strip().lower()
    if claim_source and method_source:
        if claim_source == method_source:
            score += 4
        elif claim_source in method_source or method_source in claim_source:
            score += 2

    claim_entities = {entity.strip().lower() for entity in claim.entity_names if entity.strip()}
    method_entities = {entity.strip().lower() for entity in method.entity_names if entity.strip()}
    entity_overlap = claim_entities & method_entities
    if entity_overlap:
        score += 3 * len(entity_overlap)

    token_overlap = _claim_tokens(claim.statement) & _claim_tokens(method.statement)
    if token_overlap:
        score += min(5, len(token_overlap))

    rejected_tokens = _claim_tokens((claim.rejected_what or "") + " " + (claim.rejected_why or ""))
    if rejected_tokens & _claim_tokens(method.statement):
        score += 2

    return score


def _best_method_parent(
    claim: RawClaim,
    method_claims: list[RawClaim],
) -> tuple[str | None, int]:
    best_id: str | None = None
    best_score = -1
    for method in method_claims:
        if method.claim_id == claim.claim_id:
            continue
        score = _method_affinity_score(claim, method)
        if score > best_score:
            best_score = score
            best_id = method.claim_id
    return best_id, best_score


def _context_affinity_score(claim: RawClaim, context_claim: RawClaim) -> int:
    score = 0
    overlap_evidence = _evidence_ids(claim) & _evidence_ids(context_claim)
    if overlap_evidence:
        score += 6 + len(overlap_evidence)

    claim_source = claim.source_section.strip().lower()
    context_source = context_claim.source_section.strip().lower()
    if claim_source and context_source:
        if claim_source == context_source:
            score += 3
        elif claim_source in context_source or context_source in claim_source:
            score += 2

    overlap_tokens = _claim_tokens(claim.statement) & _claim_tokens(context_claim.statement)
    if overlap_tokens:
        score += min(5, len(overlap_tokens))

    claim_entities = {entity.strip().lower() for entity in claim.entity_names if entity.strip()}
    context_entities = {
        entity.strip().lower()
        for entity in context_claim.entity_names
        if entity.strip()
    }
    overlap_entities = claim_entities & context_entities
    if overlap_entities:
        score += 2 * len(overlap_entities)

    return score


def _best_context_parent(
    claim: RawClaim,
    context_claims: list[RawClaim],
) -> tuple[str | None, int]:
    best_id: str | None = None
    best_score = -1
    for context_claim in context_claims:
        score = _context_affinity_score(claim, context_claim)
        if score > best_score:
            best_score = score
            best_id = context_claim.claim_id
    return best_id, best_score


def _hinted_parent_candidate(
    claim: RawClaim,
    *,
    claim_by_id: dict[str, RawClaim],
    method_claims: list[RawClaim],
    context_claims: list[RawClaim],
) -> str | None:
    hints = claim.structural_hints
    if hints is None:
        return None

    hinted_seed_parent = (hints.elaborates_seed_id or "").strip()
    if hinted_seed_parent and hinted_seed_parent in claim_by_id and hinted_seed_parent != claim.claim_id:
        parent_type = claim_by_id[hinted_seed_parent].claim_type
        if parent_type in _ALLOWED_PARENT_TYPES.get(claim.claim_type, {ClaimType.context, ClaimType.method}):
            return hinted_seed_parent

    preferred_parent = hints.preferred_parent_type
    if preferred_parent == ParentPreference.context and context_claims:
        best_context_id, score = _best_context_parent(claim, context_claims)
        if best_context_id is not None and score > 0:
            return best_context_id
        return context_claims[0].claim_id

    if preferred_parent == ParentPreference.method and method_claims:
        candidate_methods = [method for method in method_claims if method.claim_id != claim.claim_id]
        if claim.claim_type == ClaimType.method and _is_low_level_method(claim):
            top_level_methods = [method for method in candidate_methods if not _is_low_level_method(method)]
            if top_level_methods:
                candidate_methods = top_level_methods
        if candidate_methods:
            best_method_id, score = _best_method_parent(claim, candidate_methods)
            if best_method_id is not None and score >= 0:
                return best_method_id
            return candidate_methods[0].claim_id

    return None


def _method_specificity_score(claim: RawClaim) -> int:
    tokens = _claim_tokens(claim.statement)
    low_level_hits = len(tokens & _LOW_LEVEL_METHOD_TOKENS)
    evidence_bonus = min(2, len(_evidence_ids(claim)))
    return low_level_hits + evidence_bonus


def _is_low_level_method(claim: RawClaim) -> bool:
    hint = claim.structural_hints
    if hint is not None and hint.local_role is not None:
        if hint.local_role == ClaimLocalRole.implementation_detail:
            return True
        if hint.local_role == ClaimLocalRole.top_level:
            return False
    statement = _clean_text(claim.statement).lower()
    if any(hint in statement for hint in _HIGH_LEVEL_METHOD_HINTS):
        return False
    return _method_specificity_score(claim) >= 2


def _paper_label(metadata: PaperMetadata) -> str:
    title = _clean_text(metadata.title)
    venue_year: list[str] = []
    if metadata.venue:
        venue_year.append(_clean_text(metadata.venue))
    if metadata.year is not None:
        venue_year.append(str(metadata.year))
    if venue_year:
        return f"{title} ({' '.join(venue_year)})"
    return title


def _artifact_label(artifact: EvidenceArtifact) -> str:
    artifact_type = artifact.artifact_type.strip().lower()
    match = re.search(r"\d+[A-Za-z]?", artifact.artifact_id)
    number = match.group(0) if match else ""

    if artifact_type == "figure":
        return f"Fig. {number}" if number else artifact.artifact_id
    if artifact_type == "table":
        return f"Table {number}" if number else artifact.artifact_id
    if artifact_type == "equation":
        return f"Eq. {number}" if number else artifact.artifact_id
    return artifact.artifact_id


def _artifact_label_map(artifacts: list[EvidenceArtifact]) -> dict[str, str]:
    return {artifact.artifact_id: _artifact_label(artifact) for artifact in artifacts}


def _is_specified(value: str) -> bool:
    return bool(_clean_text(value)) and _clean_text(value).upper() != "UNSPECIFIED"


def _format_evidence(claim: RawClaim, artifact_labels: dict[str, str]) -> str:
    if not claim.evidence:
        return "[evidence: none]"

    labels: list[str] = []
    seen: set[str] = set()
    for pointer in claim.evidence:
        label = artifact_labels.get(pointer.artifact_id, pointer.artifact_id)
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)

    if not labels:
        return "[evidence: none]"
    return f"[evidence: {', '.join(labels)}]"


def _format_structural_hints(claim: RawClaim) -> str:
    hints = claim.structural_hints
    if hints is None:
        return ""

    fields: list[str] = []
    if hints.elaborates_seed_id:
        fields.append(f"seed={hints.elaborates_seed_id}")
    if hints.local_role is not None:
        fields.append(f"role={hints.local_role.value}")
    if hints.preferred_parent_type is not None:
        fields.append(f"pref_parent={hints.preferred_parent_type.value}")
    if not fields:
        return ""
    return f"[hints: {', '.join(fields)}]"


def _format_claim_block(prefix: str, claims: list[RawClaim], artifact_labels: dict[str, str]) -> str:
    if not claims:
        return "(none)"

    lines: list[str] = []
    for index, claim in enumerate(claims, start=1):
        hint_block = _format_structural_hints(claim)
        lines.append(
            f"{prefix}{index} [id={claim.claim_id}]: "
            f"{_clean_text(claim.statement)} {_format_evidence(claim, artifact_labels)}"
            f"{(' ' + hint_block) if hint_block else ''}"
        )
    return "\n".join(lines)


def _domain_facet_summary(faceted: FacetedClaim) -> list[str]:
    if faceted.systems_facets is not None:
        parts: list[str] = []
        if _is_specified(faceted.systems_facets.s1_resource):
            parts.append(f"resource: {_clean_text(faceted.systems_facets.s1_resource)}")
        if _is_specified(faceted.systems_facets.s4_mapping):
            parts.append(f"mapping: {_clean_text(faceted.systems_facets.s4_mapping)}")
        return parts

    for domain_facets in (
        faceted.architecture_facets,
        faceted.objective_facets,
        faceted.algorithm_facets,
        faceted.theory_facets,
        faceted.representation_facets,
        faceted.evaluation_facets,
        faceted.pipeline_facets,
    ):
        if domain_facets is None:
            continue
        parts = []
        for key, value in domain_facets.model_dump().items():
            if _is_specified(value):
                parts.append(f"{key}: {_clean_text(value)}")
            if len(parts) == 2:
                break
        return parts

    return []


def _facet_summary(faceted: FacetedClaim | None) -> str | None:
    if faceted is None:
        return None

    parts: list[str] = []
    if faceted.universal_facets.intervention_types:
        parts.append(faceted.universal_facets.intervention_types[0].value)

    parts.extend(_domain_facet_summary(faceted))
    parts.append(f"scope: {faceted.universal_facets.scope.value}")

    tradeoff = faceted.universal_facets.core_tradeoff
    if _is_specified(tradeoff):
        parts.append(f"tradeoff: {_clean_text(tradeoff)}")

    return " | ".join(parts) if parts else None


def _format_method_block(
    method_claims: list[RawClaim],
    faceted_by_id: dict[str, FacetedClaim],
    artifact_labels: dict[str, str],
) -> str:
    if not method_claims:
        return "(none)"

    lines: list[str] = []
    for index, claim in enumerate(method_claims, start=1):
        hint_block = _format_structural_hints(claim)
        lines.append(
            f"M{index} [id={claim.claim_id}]: "
            f"{_clean_text(claim.statement)} {_format_evidence(claim, artifact_labels)}"
            f"{(' ' + hint_block) if hint_block else ''}"
        )
        summary = _facet_summary(faceted_by_id.get(claim.claim_id))
        if summary:
            lines.append(f"    [{summary}]")
    return "\n".join(lines)


def _format_negative_claim(claim: RawClaim) -> str:
    rejected_what = _clean_text(claim.rejected_what or "")
    rejected_why = _clean_text(claim.rejected_why or "")
    if rejected_what or rejected_why:
        if rejected_what and rejected_why:
            return f"REJECTED {rejected_what} - {rejected_why}"
        if rejected_what:
            return f"REJECTED {rejected_what}"
        return f"REJECTED approach - {rejected_why}"
    return _clean_text(claim.statement)


def _format_negative_block(negative_claims: list[RawClaim], artifact_labels: dict[str, str]) -> str:
    if not negative_claims:
        return "(none)"

    lines: list[str] = []
    for index, claim in enumerate(negative_claims, start=1):
        source = _clean_text(claim.source_section)
        source_block = f"[source: {source}]" if source else "[source: unknown]"
        hint_block = _format_structural_hints(claim)
        lines.append(
            f"N{index} [id={claim.claim_id}]: "
            f"{_format_negative_claim(claim)} {source_block} {_format_evidence(claim, artifact_labels)}"
            f"{(' ' + hint_block) if hint_block else ''}"
        )
    return "\n".join(lines)


def _format_artifact_block(artifacts: list[EvidenceArtifact]) -> str:
    if not artifacts:
        return "(none)"
    return "\n".join(f"{_artifact_label(artifact)}: {_clean_text(artifact.caption)}" for artifact in artifacts)


def _canonical_parent_hints(
    claims: list[RawClaim],
    claim_groups: list[ClaimGroup] | None,
) -> dict[str, str]:
    if not claim_groups:
        return {}

    valid_ids = {claim.claim_id for claim in claims}
    hints: dict[str, str] = {}
    for group in claim_groups:
        child_id = group.canonical_id.strip()
        parent_id = (group.parent_id or "").strip()
        if child_id not in valid_ids or parent_id not in valid_ids or child_id == parent_id:
            continue
        if child_id not in hints:
            hints[child_id] = parent_id
    return hints


def _format_canonical_hint_block(parent_hints: dict[str, str]) -> str:
    if not parent_hints:
        return "(none)"
    return "\n".join(f"{child_id} <- {parent_id}" for child_id, parent_id in sorted(parent_hints.items()))


def build_tree_prompt(
    metadata: PaperMetadata,
    claims: list[RawClaim],
    faceted_claims: list[FacetedClaim],
    negatives: list[RawClaim],
    artifacts: list[EvidenceArtifact],
    claim_groups: list[ClaimGroup] | None = None,
) -> list[dict[str, str]]:
    artifact_labels = _artifact_label_map(artifacts)
    faceted_by_id = {faceted.claim.claim_id: faceted for faceted in faceted_claims}
    parent_hints = _canonical_parent_hints(claims, claim_groups)

    context_claims = [claim for claim in claims if claim.claim_type == ClaimType.context]
    method_claims = [claim for claim in claims if claim.claim_type == ClaimType.method]
    result_claims = [claim for claim in claims if claim.claim_type == ClaimType.result]
    assumption_claims = [claim for claim in claims if claim.claim_type == ClaimType.assumption]
    negative_claims = list(negatives) or [claim for claim in claims if claim.claim_type == ClaimType.negative]

    user_content = "\n".join(
        [
            f"PAPER: {_paper_label(metadata)}",
            "",
            "=== CONTEXT CLAIMS ===",
            _format_claim_block("C", context_claims, artifact_labels),
            "",
            "=== METHOD CLAIMS (with facets) ===",
            _format_method_block(method_claims, faceted_by_id, artifact_labels),
            "",
            "=== RESULT CLAIMS ===",
            _format_claim_block("R", result_claims, artifact_labels),
            "",
            "=== ASSUMPTION CLAIMS ===",
            _format_claim_block("A", assumption_claims, artifact_labels),
            "",
            "=== NEGATIVE CLAIMS ===",
            _format_negative_block(negative_claims, artifact_labels),
            "",
            "=== CANONICALIZATION LINKS ===",
            _format_canonical_hint_block(parent_hints),
            "",
            "=== EVIDENCE ARTIFACTS ===",
            _format_artifact_block(artifacts),
            "",
            'Return JSON with "one_liner" and "nodes" only.',
        ]
    )
    return [
        {"role": "system", "content": TREE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _would_create_parent_cycle(
    child_id: str,
    candidate_parent_id: str,
    parent_by_child: dict[str, str],
) -> bool:
    current = candidate_parent_id
    seen: set[str] = set()
    while current and current not in seen:
        if current == child_id:
            return True
        seen.add(current)
        current = parent_by_child.get(current, "")
    return False


def _clean_assignment(
    assignment: TreeNodeAssignment,
    valid_ids: set[str],
) -> tuple[str | None, list[str]]:
    parent_id = (assignment.parent_id or "").strip() or None
    if parent_id not in valid_ids:
        parent_id = None
    if parent_id == assignment.claim_id:
        parent_id = None

    depends_on: list[str] = []
    for dep_id in assignment.depends_on:
        cleaned = dep_id.strip()
        if cleaned in valid_ids and cleaned != assignment.claim_id and cleaned not in depends_on:
            depends_on.append(cleaned)
    return parent_id, depends_on


def _one_liner_or_unspecified(one_liner: OneLiner) -> OneLiner:
    return OneLiner(
        achieved=_clean_text(one_liner.achieved) or "UNSPECIFIED",
        via=_clean_text(one_liner.via) or "UNSPECIFIED",
        because=_clean_text(one_liner.because) or "UNSPECIFIED",
    )


def _build_tree_nodes(
    claims: list[RawClaim],
    faceted: list[FacetedClaim],
    assignments: list[TreeNodeAssignment],
    parent_hints: dict[str, str] | None = None,
) -> list[ClaimNode]:
    claim_by_id: dict[str, RawClaim] = {}
    claim_order: list[str] = []
    for claim in claims:
        if claim.claim_id in claim_by_id:
            continue
        claim_by_id[claim.claim_id] = claim
        claim_order.append(claim.claim_id)

    valid_ids = set(claim_by_id)
    assignment_by_id: dict[str, TreeNodeAssignment] = {}
    for assignment in assignments:
        claim_id = assignment.claim_id.strip()
        if claim_id in valid_ids and claim_id not in assignment_by_id:
            assignment_by_id[claim_id] = assignment

    depends_by_claim: dict[str, list[str]] = {claim_id: [] for claim_id in claim_order}
    parent_by_claim: dict[str, str] = {}

    if parent_hints:
        for child_id, parent_id in parent_hints.items():
            if child_id not in valid_ids or parent_id not in valid_ids or child_id == parent_id:
                continue
            if _would_create_parent_cycle(child_id, parent_id, parent_by_claim):
                continue
            parent_by_claim[child_id] = parent_id
            depends_by_claim[child_id] = [parent_id]

    for claim_id in claim_order:
        assignment = assignment_by_id.get(claim_id)
        if assignment is None:
            continue
        parent_id, depends_on = _clean_assignment(assignment, valid_ids)
        depends_by_claim[claim_id] = depends_on
        if parent_id is None:
            continue
        if _would_create_parent_cycle(claim_id, parent_id, parent_by_claim):
            continue
        parent_by_claim[claim_id] = parent_id

    context_ids = [claim_id for claim_id in claim_order if claim_by_id[claim_id].claim_type == ClaimType.context]
    context_claims = [claim_by_id[claim_id] for claim_id in context_ids]
    method_ids = [claim_id for claim_id in claim_order if claim_by_id[claim_id].claim_type == ClaimType.method]
    method_claims = [claim_by_id[claim_id] for claim_id in method_ids]
    result_ids = [claim_id for claim_id in claim_order if claim_by_id[claim_id].claim_type == ClaimType.result]

    # Enforce structural grammar on provided parent assignments.
    for child_id, parent_id in list(parent_by_claim.items()):
        child_claim = claim_by_id[child_id]
        parent_claim = claim_by_id.get(parent_id)
        if parent_claim is None:
            del parent_by_claim[child_id]
            continue

        allowed_parent_types = _ALLOWED_PARENT_TYPES.get(child_claim.claim_type, set())
        if parent_claim.claim_type in allowed_parent_types:
            continue

        del parent_by_claim[child_id]
        if depends_by_claim.get(child_id) == [parent_id]:
            depends_by_claim[child_id] = []

    # Backfill missing parents for non-context claims whenever a compatible anchor exists.
    for claim_id in claim_order:
        if claim_id in parent_by_claim:
            continue

        claim = claim_by_id[claim_id]
        if claim.claim_type == ClaimType.context:
            continue

        hinted_parent = _hinted_parent_candidate(
            claim,
            claim_by_id=claim_by_id,
            method_claims=method_claims,
            context_claims=context_claims,
        )
        if hinted_parent is not None and not _would_create_parent_cycle(claim_id, hinted_parent, parent_by_claim):
            parent_by_claim[claim_id] = hinted_parent
            continue

        fallback_parent: str | None = None
        if claim.claim_type == ClaimType.method:
            if context_claims:
                best_context_id, context_score = _best_context_parent(claim, context_claims)
                fallback_parent = best_context_id if best_context_id is not None and context_score > 0 else context_ids[0]
        elif claim.claim_type == ClaimType.result:
            if method_ids:
                best_method_id, score = _best_method_parent(claim, method_claims)
                fallback_parent = best_method_id if best_method_id is not None and score > 0 else method_ids[0]
            elif context_ids:
                fallback_parent = context_ids[0]
        elif claim.claim_type == ClaimType.assumption:
            if method_ids:
                best_method_id, score = _best_method_parent(claim, method_claims)
                fallback_parent = best_method_id if best_method_id is not None and score > 0 else method_ids[0]
            elif result_ids:
                fallback_parent = result_ids[0]
            elif context_ids:
                fallback_parent = context_ids[0]
        elif claim.claim_type == ClaimType.negative:
            if method_ids:
                best_method_id, score = _best_method_parent(claim, method_claims)
                if best_method_id is not None and score > 0:
                    fallback_parent = best_method_id
            if fallback_parent is None and context_claims:
                best_context_id, score = _best_context_parent(claim, context_claims)
                fallback_parent = best_context_id if best_context_id is not None and score > 0 else context_ids[0]

        if fallback_parent is None:
            continue
        if _would_create_parent_cycle(claim_id, fallback_parent, parent_by_claim):
            continue
        parent_by_claim[claim_id] = fallback_parent

    related_context_roots = [
        claim_id
        for claim_id in context_ids
        if claim_id not in parent_by_claim and _related_work_source(claim_by_id[claim_id].source_section)
    ]
    if len(related_context_roots) > 1:
        umbrella = related_context_roots[0]
        for claim_id in related_context_roots[1:]:
            if _would_create_parent_cycle(claim_id, umbrella, parent_by_claim):
                continue
            parent_by_claim[claim_id] = umbrella
            depends_by_claim[claim_id] = [umbrella]

    child_count: dict[str, int] = {}
    for parent_id in parent_by_claim.values():
        child_count[parent_id] = child_count.get(parent_id, 0) + 1

    # Low-level implementation claims should prefer method parents over broad context roots.
    for claim_id in method_ids:
        if len(method_claims) <= 1:
            break

        claim = claim_by_id[claim_id]
        current_parent = parent_by_claim.get(claim_id)
        best_method_id, best_score = _best_method_parent(claim, method_claims)
        hints = claim.structural_hints
        hinted_parent = hints.elaborates_seed_id if hints is not None else None
        hint_requests_method_parent = (
            hints is not None and hints.preferred_parent_type == ParentPreference.method
        )
        if best_method_id is None or best_method_id == current_parent:
            continue
        if best_score < 7 and not (
            hint_requests_method_parent and hinted_parent == best_method_id
        ):
            continue

        current_parent_claim = claim_by_id.get(current_parent) if current_parent else None
        parent_is_context = current_parent_claim is not None and current_parent_claim.claim_type == ClaimType.context
        parent_is_method = current_parent_claim is not None and current_parent_claim.claim_type == ClaimType.method

        should_reattach = False
        if current_parent is None or parent_is_context:
            should_reattach = _is_low_level_method(claim)
        elif parent_is_method:
            current_score = _method_affinity_score(claim, current_parent_claim)
            overloaded_parent = child_count.get(current_parent, 0) >= 8
            should_reattach = best_score >= current_score + 4 or (
                overloaded_parent and best_score > current_score
            )

        if not should_reattach:
            continue
        if _would_create_parent_cycle(claim_id, best_method_id, parent_by_claim):
            continue

        if current_parent:
            child_count[current_parent] = max(0, child_count.get(current_parent, 1) - 1)
        parent_by_claim[claim_id] = best_method_id
        child_count[best_method_id] = child_count.get(best_method_id, 0) + 1

    # Reattach results/assumptions/negatives to the most specific method when warranted.
    for claim_id in claim_order:
        claim = claim_by_id[claim_id]
        if claim.claim_type not in {ClaimType.result, ClaimType.assumption, ClaimType.negative}:
            continue
        if not method_claims:
            continue

        current_parent = parent_by_claim.get(claim_id)
        best_method_id, best_score = _best_method_parent(claim, method_claims)
        if best_method_id is None:
            continue

        if current_parent is None:
            if _would_create_parent_cycle(claim_id, best_method_id, parent_by_claim):
                continue
            parent_by_claim[claim_id] = best_method_id
            child_count[best_method_id] = child_count.get(best_method_id, 0) + 1
            continue

        current_parent_claim = claim_by_id.get(current_parent)
        if current_parent_claim is None or current_parent_claim.claim_type != ClaimType.method:
            if _would_create_parent_cycle(claim_id, best_method_id, parent_by_claim):
                continue
            child_count[current_parent] = max(0, child_count.get(current_parent, 1) - 1)
            parent_by_claim[claim_id] = best_method_id
            child_count[best_method_id] = child_count.get(best_method_id, 0) + 1
            continue

        current_score = _method_affinity_score(claim, current_parent_claim)
        overloaded_parent = child_count.get(current_parent, 0) >= 8
        better_specific_match = best_score >= current_score + 3
        should_reattach = best_method_id != current_parent and (
            better_specific_match or (overloaded_parent and best_score > current_score)
        )
        if not should_reattach:
            continue
        if _would_create_parent_cycle(claim_id, best_method_id, parent_by_claim):
            continue
        child_count[current_parent] = max(0, child_count.get(current_parent, 1) - 1)
        parent_by_claim[claim_id] = best_method_id
        child_count[best_method_id] = child_count.get(best_method_id, 0) + 1

    for claim_id in claim_order:
        claim = claim_by_id[claim_id]
        if claim.claim_type not in {ClaimType.result, ClaimType.assumption, ClaimType.negative}:
            continue
        parent_id = parent_by_claim.get(claim_id)
        if parent_id is None:
            continue
        parent_claim = claim_by_id.get(parent_id)
        if parent_claim is None or parent_claim.claim_type != ClaimType.method:
            continue
        depends_by_claim[claim_id] = [parent_id]

    for claim_id, parent_id in parent_by_claim.items():
        if not depends_by_claim[claim_id]:
            depends_by_claim[claim_id] = [parent_id]

    facets_by_id = {faceted_claim.claim.claim_id: faceted_claim for faceted_claim in faceted}
    nodes = {
        claim_id: ClaimNode(
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            statement=claim.statement,
            evidence=list(claim.evidence),
            facets=facets_by_id.get(claim_id),
            children=[],
            depends_on=list(depends_by_claim.get(claim_id, [])),
            rejected_what=claim.rejected_what,
            rejected_why=claim.rejected_why,
        )
        for claim_id, claim in claim_by_id.items()
    }

    for claim_id in claim_order:
        parent_id = parent_by_claim.get(claim_id)
        if parent_id is None:
            continue
        parent_node = nodes.get(parent_id)
        child_node = nodes.get(claim_id)
        if parent_node is None or child_node is None:
            continue
        parent_node.children.append(child_node)

    return [nodes[claim_id] for claim_id in claim_order if claim_id not in parent_by_claim]


async def assemble_tree(
    metadata: PaperMetadata,
    claims: list[RawClaim],
    faceted: list[FacetedClaim],
    negatives: list[RawClaim],
    artifacts: list[EvidenceArtifact],
    config: Any,
    claim_groups: list[ClaimGroup] | None = None,
) -> PaperDecomposition:
    messages = build_tree_prompt(
        metadata,
        claims,
        faceted,
        negatives,
        artifacts,
        claim_groups=claim_groups,
    )
    normalized_negatives = list(negatives) or [claim for claim in claims if claim.claim_type == ClaimType.negative]

    combined_claims = list(claims)
    known_claim_ids = {claim.claim_id for claim in combined_claims}
    for negative_claim in normalized_negatives:
        if negative_claim.claim_id in known_claim_ids:
            continue
        combined_claims.append(negative_claim)
        known_claim_ids.add(negative_claim.claim_id)

    result = await call_model(
        tier=_resolve_model_tier(config),
        messages=messages,
        response_schema=TreeAssemblyOutput,
        config=config,
    )
    if not isinstance(result, TreeAssemblyOutput):
        raise TypeError("Expected TreeAssemblyOutput from structured model call.")

    parent_hints = _canonical_parent_hints(combined_claims, claim_groups)
    claim_tree = _build_tree_nodes(combined_claims, faceted, result.nodes, parent_hints=parent_hints)
    return PaperDecomposition(
        metadata=metadata,
        one_liner=_one_liner_or_unspecified(result.one_liner),
        claim_tree=claim_tree,
        negative_claims=normalized_negatives,
        all_artifacts=list(artifacts),
    )


__all__ = ["TREE_SYSTEM_PROMPT", "build_tree_prompt", "assemble_tree"]
