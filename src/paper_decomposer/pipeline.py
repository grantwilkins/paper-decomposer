from __future__ import annotations

import asyncio
import re
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar, cast

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .config import load_config
from .models import get_cost_tracker, preflight_model_tiers, reset_cost_tracker
from .pdf_parser import parse_pdf
from .prompts.dedup import chunked_dedup, hybrid_dedup_promoted
from .prompts.facets import extract_facets
from .prompts.seed import extract_seed, extract_skeleton, repair_skeleton
from .prompts.section import (
    _claim_worthiness_score,
    _worth_score_threshold,
    extract_section_claims,
    extract_section_digest,
)
from .prompts.tree import assemble_tree, assemble_tree_deterministic
from .schema import (
    ClaimGroup,
    ClaimLocalRole,
    ClaimNode,
    ClaimStructuralHints,
    ClaimType,
    FacetedClaim,
    ModelTier,
    OneLiner,
    PaperDecomposition,
    PaperDocument,
    PaperSkeletonCandidate,
    ParentPreference,
    RawClaim,
    SectionArgumentCandidate,
    Section,
    SupportDetail,
)

console = Console()

T = TypeVar("T")
_VALID_TIERS: set[ModelTier] = {"small", "medium", "heavy"}
_CLAIM_ID_PREFIX_BY_TYPE: dict[ClaimType, str] = {
    ClaimType.context: "C",
    ClaimType.method: "M",
    ClaimType.result: "R",
    ClaimType.assumption: "A",
    ClaimType.negative: "N",
}
_CLAIM_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_CLAIM_STOPWORDS = {
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
_RESULT_HINT_RE = re.compile(
    r"(achiev|improv|increase|decrease|reduc|outperform|speedup|throughput|latenc|memory waste|oom|accuracy)",
    re.IGNORECASE,
)
_METHOD_HINT_RE = re.compile(
    r"(introduc|propos|present|design|build|implement|allocate|schedul|kernel|map|partition|mechanism|algorithm)",
    re.IGNORECASE,
)
_CONTEXT_HINT_RE = re.compile(
    r"(challenge|problem|gap|bottleneck|limitation|insufficient|fragmentation|existing approach|struggle|fails?)",
    re.IGNORECASE,
)
_NUMERIC_SIGNAL_RE = re.compile(r"(\d+(?:\.\d+)?\s*(x|%|percent|ms|s|gb|mb|tok/s|tokens/s))|(\bvs\.?\b)", re.IGNORECASE)


def _collapse_ws(value: str) -> str:
    return " ".join(value.split())


def _statement_tokens(text: str) -> set[str]:
    lowered = _collapse_ws(text).lower()
    return {
        token
        for token in _CLAIM_TOKEN_SPLIT_RE.split(lowered)
        if token and len(token) > 2 and token not in _CLAIM_STOPWORDS
    }


def _evidence_ids(claim: RawClaim) -> set[str]:
    return {pointer.artifact_id.strip().lower() for pointer in claim.evidence if pointer.artifact_id.strip()}


def _claim_quality_score(claim: RawClaim) -> tuple[int, int, int]:
    evidence_score = len(_evidence_ids(claim))
    entity_score = len({entity.strip().lower() for entity in claim.entity_names if entity.strip()})
    token_score = len(_statement_tokens(claim.statement))
    return evidence_score, entity_score, token_score


def _fallback_claim_strength(claim: RawClaim) -> float:
    evidence_score, entity_score, token_score = _claim_quality_score(claim)
    lowered = _collapse_ws(claim.statement).lower()

    score = float(token_score) / 4.0
    score += float(evidence_score)
    score += float(min(entity_score, 2)) * 0.6

    if claim.claim_type in {ClaimType.result, ClaimType.assumption, ClaimType.negative}:
        score += 1.0
    elif claim.claim_type == ClaimType.context:
        score += 0.4

    if _RESULT_HINT_RE.search(lowered):
        score += 1.0
    if _METHOD_HINT_RE.search(lowered):
        score += 0.8
    if _CONTEXT_HINT_RE.search(lowered):
        score += 0.4

    return score


def _refined_claim_strength(claim: RawClaim, source: Section | str) -> float:
    score = (
        _claim_worthiness_score(claim, source)
        if isinstance(source, Section)
        else _fallback_claim_strength(claim)
    )
    hints = claim.structural_hints
    if hints is not None:
        if hints.local_role == ClaimLocalRole.top_level:
            score += 0.8
        elif hints.local_role == ClaimLocalRole.mechanism:
            score += 0.3
        elif hints.local_role == ClaimLocalRole.implementation_detail:
            score -= 0.8
        if hints.elaborates_seed_id and claim.claim_type == ClaimType.method:
            score += 0.2
    return round(score, 3)


def _claim_strength_threshold(claim: RawClaim, source: Section | str) -> float:
    if isinstance(source, Section):
        threshold = _worth_score_threshold(source, stage="post")
    elif claim.claim_type == ClaimType.method:
        threshold = 2.4
    elif claim.claim_type in {ClaimType.result, ClaimType.assumption, ClaimType.negative}:
        threshold = 1.8
    else:
        threshold = 1.4

    hints = claim.structural_hints
    if claim.claim_type == ClaimType.method and hints is not None:
        if hints.local_role == ClaimLocalRole.top_level:
            threshold -= 0.6
        elif hints.local_role == ClaimLocalRole.mechanism:
            threshold -= 0.2
        elif hints.local_role == ClaimLocalRole.implementation_detail:
            threshold += 0.6
    return threshold


def _refine_tree_candidate_claims(claims: list[RawClaim], sections: list[Section]) -> list[RawClaim]:
    refined_claims: list[RawClaim] = []
    retained_method_ids: set[str] = set()
    strongest_method_id: str | None = None
    strongest_method_score = float("-inf")

    for claim in claims:
        source = _find_section_for_claim(claim, sections)
        strength = _refined_claim_strength(claim, source)
        refined_claim = claim.model_copy(update={"claim_strength": strength})
        refined_claims.append(refined_claim)

        if refined_claim.claim_type != ClaimType.method:
            continue

        if strength > strongest_method_score:
            strongest_method_score = strength
            strongest_method_id = refined_claim.claim_id

        if strength >= _claim_strength_threshold(refined_claim, source):
            retained_method_ids.add(refined_claim.claim_id)

    if strongest_method_id is not None and not retained_method_ids:
        retained_method_ids.add(strongest_method_id)

    if strongest_method_id is None:
        return refined_claims

    return [
        claim
        for claim in refined_claims
        if claim.claim_type != ClaimType.method or claim.claim_id in retained_method_ids
    ]


def _retag_claim_type(claim: RawClaim) -> ClaimType:
    if claim.claim_type == ClaimType.negative:
        return claim.claim_type

    text = _collapse_ws(claim.statement).lower()
    if not text:
        return claim.claim_type

    result_score = len(_RESULT_HINT_RE.findall(text)) + (2 if _NUMERIC_SIGNAL_RE.search(text) else 0)
    method_score = len(_METHOD_HINT_RE.findall(text))
    context_score = len(_CONTEXT_HINT_RE.findall(text))

    if claim.claim_type == ClaimType.context:
        if result_score >= 3 and result_score >= context_score + 2 and result_score >= method_score + 1:
            return ClaimType.result
        if method_score >= 3 and method_score >= context_score + 2 and result_score <= 2:
            return ClaimType.method
    elif claim.claim_type == ClaimType.method:
        if result_score >= 4 and result_score >= method_score + 2:
            return ClaimType.result
    elif claim.claim_type == ClaimType.result:
        if method_score >= 4 and method_score >= result_score + 2 and not _NUMERIC_SIGNAL_RE.search(text):
            return ClaimType.method

    return claim.claim_type


def _hint_defaults_for_type(claim_type: ClaimType) -> tuple[ClaimLocalRole, ParentPreference]:
    if claim_type == ClaimType.context:
        return ClaimLocalRole.context_gap, ParentPreference.context
    if claim_type == ClaimType.method:
        return ClaimLocalRole.mechanism, ParentPreference.method
    if claim_type == ClaimType.result:
        return ClaimLocalRole.empirical_finding, ParentPreference.method
    if claim_type == ClaimType.assumption:
        return ClaimLocalRole.assumption, ParentPreference.method
    return ClaimLocalRole.limitation, ParentPreference.method


def _normalize_claim_semantics(claims: list[RawClaim]) -> list[RawClaim]:
    normalized: list[RawClaim] = []
    for claim in claims:
        statement = _collapse_ws(claim.statement).strip()
        if not statement:
            continue
        source_section = _collapse_ws(claim.source_section).strip() or claim.source_section
        entity_names = [entity for entity in (_collapse_ws(name).strip() for name in claim.entity_names) if entity]

        target_type = _retag_claim_type(claim)
        hints = claim.structural_hints
        if target_type != claim.claim_type:
            default_local_role, default_parent_pref = _hint_defaults_for_type(target_type)
            hints = ClaimStructuralHints(
                elaborates_seed_id=hints.elaborates_seed_id if hints is not None else None,
                local_role=default_local_role,
                preferred_parent_type=default_parent_pref,
            )

        normalized.append(
            claim.model_copy(
                update={
                    "statement": statement,
                    "source_section": source_section,
                    "entity_names": entity_names,
                    "claim_type": target_type,
                    "structural_hints": hints,
                }
            )
        )
    return normalized


def _claims_near_duplicate(left: RawClaim, right: RawClaim) -> bool:
    if left.claim_type != right.claim_type:
        return False

    left_tokens = _statement_tokens(left.statement)
    right_tokens = _statement_tokens(right.statement)
    if not left_tokens or not right_tokens:
        return False

    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = overlap / union if union else 0.0
    containment = max(overlap / len(left_tokens), overlap / len(right_tokens))

    left_source = _collapse_ws(left.source_section).lower()
    right_source = _collapse_ws(right.source_section).lower()
    same_source = bool(left_source and right_source and (left_source == right_source or left_source in right_source or right_source in left_source))
    evidence_overlap = bool(_evidence_ids(left) & _evidence_ids(right))

    if jaccard >= 0.92:
        return True
    if containment >= 0.90 and (same_source or evidence_overlap or jaccard >= 0.84):
        return True
    return False


def _collapse_claim_duplicates(claims: list[RawClaim]) -> list[RawClaim]:
    if len(claims) < 2:
        return claims

    by_exact_key: dict[tuple[ClaimType, str], RawClaim] = {}
    kept_exact: list[RawClaim] = []
    for claim in claims:
        key = (claim.claim_type, _collapse_ws(claim.statement).lower())
        existing = by_exact_key.get(key)
        if existing is None:
            by_exact_key[key] = claim
            kept_exact.append(claim)
            continue
        if _claim_quality_score(claim) > _claim_quality_score(existing):
            by_exact_key[key] = claim
            idx = kept_exact.index(existing)
            kept_exact[idx] = claim

    candidates = [(idx, claim) for idx, claim in enumerate(kept_exact)]
    ranked = sorted(
        candidates,
        key=lambda item: (
            -_claim_quality_score(item[1])[0],
            -_claim_quality_score(item[1])[1],
            -_claim_quality_score(item[1])[2],
            item[0],
        ),
    )

    selected: list[tuple[int, RawClaim]] = []
    for original_idx, claim in ranked:
        duplicate_of_selected = any(_claims_near_duplicate(claim, existing) for _, existing in selected)
        if duplicate_of_selected:
            continue
        selected.append((original_idx, claim))

    selected.sort(key=lambda item: item[0])
    return [claim for _, claim in selected]


def _section_label(section: Section) -> str:
    if section.section_number:
        return f"{section.section_number} {section.title}"
    return section.title


def _is_reference_section(section: Section) -> bool:
    lowered = section.title.lower()
    return "reference" in lowered or "bibliograph" in lowered


def _sanitize_filename(stem: str, fallback: str) -> str:
    candidate = re.sub(r"\s+", " ", stem).strip()[:60]
    candidate = re.sub(r"[^A-Za-z0-9._ -]+", "", candidate).strip(" .")
    if not candidate:
        candidate = fallback
    candidate = candidate.replace(" ", "_")
    return candidate or fallback


def _resolve_output_dir(config: Any) -> Path:
    pipeline = getattr(config, "pipeline", None)
    output_cfg = getattr(pipeline, "output", None) if pipeline is not None else None
    if isinstance(output_cfg, dict):
        output_dir = output_cfg.get("output_dir", "output")
    else:
        output_dir = "output"
    return Path(str(output_dir))


def _find_seed_text(document: PaperDocument) -> str:
    abstract_sections = [section for section in document.sections if section.role.value == "abstract"]
    if abstract_sections:
        return "\n\n".join(section.body_text for section in abstract_sections)

    intro_sections = [section for section in document.sections if section.role.value == "introduction"]
    if intro_sections:
        return "\n\n".join(section.body_text for section in intro_sections)

    if not document.sections:
        return ""
    return document.sections[0].body_text


def _find_section_for_claim(claim: RawClaim, sections: list[Section]) -> Section | str:
    source = claim.source_section.strip().lower()
    if not source:
        return claim.source_section

    def score(section: Section) -> int:
        title = section.title.strip().lower()
        label = _section_label(section).strip().lower()
        number = (section.section_number or "").strip().lower()

        if source == label or source == title:
            return 100

        value = 0
        if number and number in source:
            value += 20
        if title and title in source:
            value += 15
        if source in label:
            value += 10

        source_tokens = {token for token in re.split(r"\W+", source) if token}
        title_tokens = {token for token in re.split(r"\W+", title) if token}
        overlap = len(source_tokens & title_tokens)
        value += overlap
        return value

    best = max(sections, key=score, default=None)
    if best is None or score(best) == 0:
        return claim.source_section
    return best


def _claim_type_counts(claims: list[RawClaim]) -> dict[str, int]:
    counts = Counter(claim.claim_type.value for claim in claims)
    return {claim_type.value: counts.get(claim_type.value, 0) for claim_type in ClaimType}


def _uniquify_claim_ids(claims: list[RawClaim]) -> tuple[list[RawClaim], dict[str, str]]:
    seen: dict[str, int] = {}
    used: set[str] = set()
    unique_claims: list[RawClaim] = []
    id_map: dict[str, str] = {}

    for claim in claims:
        base_id = claim.claim_id.strip() or "claim"
        current_count = seen.get(base_id, 0)

        if current_count == 0 and base_id not in used:
            next_id = base_id
        else:
            suffix = current_count + 1
            next_id = f"{base_id}_{suffix}"
            while next_id in used:
                suffix += 1
                next_id = f"{base_id}_{suffix}"

        seen[base_id] = current_count + 1
        used.add(next_id)
        if base_id not in id_map:
            id_map[base_id] = next_id

        if next_id == claim.claim_id:
            unique_claims.append(claim)
        else:
            unique_claims.append(claim.model_copy(update={"claim_id": next_id}))

    return unique_claims, id_map


def _normalize_claim_id_scheme(claims: list[RawClaim]) -> tuple[list[RawClaim], dict[str, str]]:
    counters: Counter[str] = Counter()
    normalized: list[RawClaim] = []
    id_map: dict[str, str] = {}
    for claim in claims:
        original_id = claim.claim_id
        prefix = _CLAIM_ID_PREFIX_BY_TYPE.get(claim.claim_type, "X")
        counters[prefix] += 1
        normalized_id = f"{prefix}{counters[prefix]}"
        id_map[original_id] = normalized_id
        if claim.claim_id == normalized_id:
            normalized.append(claim)
        else:
            normalized.append(claim.model_copy(update={"claim_id": normalized_id}))
    return normalized, id_map


def _remap_structural_hint_ids(claims: list[RawClaim], id_map: dict[str, str]) -> list[RawClaim]:
    if not claims or not id_map:
        return claims

    remapped: list[RawClaim] = []
    for claim in claims:
        hints = claim.structural_hints
        if hints is None or not hints.elaborates_seed_id:
            remapped.append(claim)
            continue

        remapped_id = id_map.get(hints.elaborates_seed_id, hints.elaborates_seed_id)
        if remapped_id == hints.elaborates_seed_id:
            remapped.append(claim)
            continue

        remapped.append(
            claim.model_copy(
                update={
                    "structural_hints": hints.model_copy(update={"elaborates_seed_id": remapped_id}),
                }
            )
        )
    return remapped


def _merge_structural_hints(
    claims: list[RawClaim],
    groups: list[ClaimGroup],
) -> list[RawClaim]:
    if not claims or not groups:
        return claims

    claim_by_id = {claim.claim_id: claim for claim in claims}
    merged_by_id = dict(claim_by_id)

    for group in groups:
        member_ids = {group.canonical_id, *group.member_ids}
        member_claims = [claim_by_id[member_id] for member_id in member_ids if member_id in claim_by_id]
        if not member_claims:
            continue

        seed_votes = Counter(
            claim.structural_hints.elaborates_seed_id
            for claim in member_claims
            if claim.structural_hints is not None and claim.structural_hints.elaborates_seed_id
        )
        local_roles = {
            claim.structural_hints.local_role
            for claim in member_claims
            if claim.structural_hints is not None and claim.structural_hints.local_role is not None
        }
        parent_prefs = {
            claim.structural_hints.preferred_parent_type
            for claim in member_claims
            if claim.structural_hints is not None and claim.structural_hints.preferred_parent_type is not None
        }

        merged_seed_id = seed_votes.most_common(1)[0][0] if seed_votes else None
        merged_local_role: ClaimLocalRole | None = None
        for candidate in (
            ClaimLocalRole.implementation_detail,
            ClaimLocalRole.mechanism,
            ClaimLocalRole.top_level,
            ClaimLocalRole.empirical_finding,
            ClaimLocalRole.assumption,
            ClaimLocalRole.limitation,
            ClaimLocalRole.context_gap,
            ClaimLocalRole.positioning,
            ClaimLocalRole.other,
        ):
            if candidate in local_roles:
                merged_local_role = candidate
                break

        merged_parent_pref: ParentPreference | None = None
        if ParentPreference.method in parent_prefs:
            merged_parent_pref = ParentPreference.method
        elif ParentPreference.context in parent_prefs:
            merged_parent_pref = ParentPreference.context
        elif ParentPreference.none in parent_prefs:
            merged_parent_pref = ParentPreference.none

        if (
            merged_seed_id is None
            and merged_local_role is None
            and merged_parent_pref is None
        ):
            continue

        canonical_id = group.canonical_id
        canonical_claim = merged_by_id.get(canonical_id)
        if canonical_claim is None:
            continue

        existing = canonical_claim.structural_hints
        merged_hints = ClaimStructuralHints(
            elaborates_seed_id=merged_seed_id or (existing.elaborates_seed_id if existing else None),
            local_role=merged_local_role or (existing.local_role if existing else None),
            preferred_parent_type=merged_parent_pref or (existing.preferred_parent_type if existing else None),
        )
        merged_by_id[canonical_id] = canonical_claim.model_copy(update={"structural_hints": merged_hints})

    return [merged_by_id[claim.claim_id] for claim in claims if claim.claim_id in merged_by_id]


def _tree_stats(nodes: list[ClaimNode]) -> tuple[int, int, int]:
    def walk(node: ClaimNode) -> tuple[int, int, int]:
        child_nodes = 0
        max_depth = 1
        edge_count = len(node.depends_on)
        for child in node.children:
            n_count, n_depth, n_edges = walk(child)
            child_nodes += n_count
            max_depth = max(max_depth, 1 + n_depth)
            edge_count += 1 + n_edges
        return 1 + child_nodes, max_depth, edge_count

    total_nodes = 0
    depth = 0
    edges = 0
    for root in nodes:
        count, root_depth, root_edges = walk(root)
        total_nodes += count
        depth = max(depth, root_depth)
        edges += root_edges
    return total_nodes, depth, edges


def _shorten(text: str, limit: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3].rstrip()}..."


def _build_fallback_decomposition(
    *,
    document: PaperDocument,
    claims: list[RawClaim],
    faceted_claims: list[FacetedClaim],
    negative_claims: list[RawClaim],
    extraction_cost_usd: float,
) -> PaperDecomposition:
    facets_by_id = {faceted.claim.claim_id: faceted for faceted in faceted_claims}

    def make_node(claim: RawClaim) -> ClaimNode:
        return ClaimNode(
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            statement=claim.statement,
            evidence=list(claim.evidence),
            facets=facets_by_id.get(claim.claim_id),
            children=[],
            depends_on=[],
            rejected_what=claim.rejected_what,
            rejected_why=claim.rejected_why,
        )

    context_claims = [claim for claim in claims if claim.claim_type == ClaimType.context]
    method_claims = [claim for claim in claims if claim.claim_type == ClaimType.method]
    result_claims = [claim for claim in claims if claim.claim_type == ClaimType.result]
    assumption_claims = [claim for claim in claims if claim.claim_type == ClaimType.assumption]

    roots = [make_node(claim) for claim in context_claims]
    method_nodes = [make_node(claim) for claim in method_claims]
    result_nodes = [make_node(claim) for claim in result_claims]
    assumption_nodes = [make_node(claim) for claim in assumption_claims]
    negative_nodes = [make_node(claim) for claim in negative_claims]

    if roots:
        for idx, method_node in enumerate(method_nodes):
            parent = roots[idx % len(roots)]
            method_node.depends_on = [parent.claim_id]
            parent.children.append(method_node)

        parents_for_results = method_nodes if method_nodes else roots
        for idx, result_node in enumerate(result_nodes):
            parent = parents_for_results[idx % len(parents_for_results)]
            result_node.depends_on = [parent.claim_id]
            parent.children.append(result_node)

        attachment_parent = method_nodes[0] if method_nodes else roots[0]
        for node in [*assumption_nodes, *negative_nodes]:
            node.depends_on = [attachment_parent.claim_id]
            attachment_parent.children.append(node)
    else:
        ordered = [*method_nodes, *result_nodes, *assumption_nodes, *negative_nodes]
        if not ordered:
            ordered = [make_node(claim) for claim in claims]

        root_count = min(3, len(ordered))
        roots = ordered[:root_count]
        if roots:
            for node in ordered[root_count:]:
                node.depends_on = [roots[0].claim_id]
                roots[0].children.append(node)

    achieved = _shorten(result_claims[0].statement) if result_claims else _shorten(claims[0].statement) if claims else "UNSPECIFIED"
    via = _shorten(method_claims[0].statement) if method_claims else "UNSPECIFIED"
    because = _shorten(context_claims[0].statement) if context_claims else "UNSPECIFIED"

    return PaperDecomposition(
        metadata=document.metadata,
        one_liner=OneLiner(achieved=achieved, via=via, because=because),
        claim_tree=roots,
        negative_claims=list(negative_claims),
        all_artifacts=list(document.all_artifacts),
        extraction_cost_usd=extraction_cost_usd,
    )


def _read_max_concurrent(config: Any) -> int:
    pipeline = getattr(config, "pipeline", None)
    section_cfg = getattr(pipeline, "section_extraction", None) if pipeline is not None else None
    if isinstance(section_cfg, dict):
        value = section_cfg.get("max_concurrent")
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 8
    return 8


def _read_phase_tier(config: Any, phase_key: str, default: ModelTier) -> ModelTier:
    pipeline = getattr(config, "pipeline", None)
    phase_cfg = getattr(pipeline, phase_key, None) if pipeline is not None else None
    if isinstance(phase_cfg, dict):
        tier = phase_cfg.get("model_tier")
        if isinstance(tier, str) and tier in _VALID_TIERS:
            return cast(ModelTier, tier)
    return default


def _required_preflight_tiers(config: Any) -> list[ModelTier]:
    tiers = [
        _read_phase_tier(config, "seed", "small"),
        _read_phase_tier(config, "section_extraction", "small"),
        _read_phase_tier(config, "dedup", "medium"),
        _read_phase_tier(config, "tree", "heavy"),
    ]

    unique: list[ModelTier] = []
    for tier in tiers:
        if tier in unique:
            continue
        unique.append(tier)
    return unique


def _cost_delta(before: dict[str, float | int], after: dict[str, float | int]) -> float:
    return float(after.get("total_cost_usd", 0.0)) - float(before.get("total_cost_usd", 0.0))


def _read_promotion_float(config: Any, key: str, default: float) -> float:
    pipeline = getattr(config, "pipeline", None)
    section_cfg = getattr(pipeline, "section_extraction", None) if pipeline is not None else None
    value: Any = None
    if isinstance(section_cfg, dict):
        promotion_cfg = section_cfg.get("promotion")
        if isinstance(promotion_cfg, dict):
            value = promotion_cfg.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _section_label_key(section: Section) -> str:
    if section.section_number:
        return f"{section.section_number} {section.title}".strip().lower()
    return section.title.strip().lower()


def _candidate_to_raw_claim(candidate: SectionArgumentCandidate) -> RawClaim:
    hints = ClaimStructuralHints(
        elaborates_seed_id=candidate.elaborates_seed_id,
        local_role=candidate.local_role,
        preferred_parent_type=candidate.preferred_parent_type,
    )
    return RawClaim(
        claim_id=candidate.claim_id,
        claim_type=candidate.claim_type,
        statement=candidate.statement,
        source_section=candidate.source_section,
        evidence=[{"artifact_id": artifact_id, "role": "supports"} for artifact_id in candidate.evidence_ids],
        entity_names=list(candidate.entity_names),
        rejected_what=candidate.rejected_what,
        rejected_why=candidate.rejected_why,
        structural_hints=hints,
        claim_strength=candidate.strength,
    )


def _claim_similarity(left: RawClaim, right: RawClaim) -> float:
    left_tokens = _statement_tokens(left.statement)
    right_tokens = _statement_tokens(right.statement)
    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    token_sim = overlap / union if union else 0.0

    left_entities = {entity.strip().lower() for entity in left.entity_names if entity.strip()}
    right_entities = {entity.strip().lower() for entity in right.entity_names if entity.strip()}
    entity_sim = 0.0
    if left_entities and right_entities:
        entity_sim = len(left_entities & right_entities) / max(1, len(left_entities | right_entities))

    left_evidence = _evidence_ids(left)
    right_evidence = _evidence_ids(right)
    evidence_sim = 0.0
    if left_evidence and right_evidence:
        evidence_sim = len(left_evidence & right_evidence) / max(1, len(left_evidence | right_evidence))

    return 0.6 * token_sim + 0.25 * entity_sim + 0.15 * evidence_sim


def _type_compatibility(child: RawClaim, parent: RawClaim) -> float:
    if child.claim_type == parent.claim_type:
        return 1.0
    allowed: dict[ClaimType, set[ClaimType]] = {
        ClaimType.method: {ClaimType.context, ClaimType.method},
        ClaimType.result: {ClaimType.method, ClaimType.result},
        ClaimType.assumption: {ClaimType.method, ClaimType.result, ClaimType.context},
        ClaimType.negative: {ClaimType.method, ClaimType.context},
        ClaimType.context: {ClaimType.context},
    }
    return 1.0 if parent.claim_type in allowed.get(child.claim_type, set()) else 0.0


def _anchor_match(candidate: RawClaim, anchors: list[RawClaim], source_by_section: dict[str, Section]) -> float:
    best = 0.0
    candidate_source = source_by_section.get(candidate.source_section.strip().lower())
    for anchor in anchors:
        type_score = _type_compatibility(candidate, anchor)
        if type_score <= 0.0:
            continue
        similarity = _claim_similarity(candidate, anchor)
        section_bonus = 0.0
        if candidate_source is not None:
            if anchor.source_section.strip().lower() == _section_label_key(candidate_source):
                section_bonus = 0.2
            elif candidate_source.role.value in {"method", "theory"} and anchor.claim_type == ClaimType.method:
                section_bonus = 0.1
            elif candidate_source.role.value in {"evaluation", "appendix"} and anchor.claim_type == ClaimType.result:
                section_bonus = 0.1
        best = max(best, min(1.0, 0.55 * type_score + 0.35 * similarity + section_bonus))
    return best


def _novelty(candidate: RawClaim, references: list[RawClaim]) -> float:
    comparable = [reference for reference in references if reference.claim_type == candidate.claim_type]
    if not comparable:
        return 1.0
    max_similarity = max((_claim_similarity(candidate, reference) for reference in comparable), default=0.0)
    return max(0.0, 1.0 - max_similarity)


def _strength(candidate: RawClaim, source_by_section: dict[str, Section]) -> float:
    source = source_by_section.get(candidate.source_section.strip().lower(), candidate.source_section)
    return _refined_claim_strength(candidate, source)


def _necessity(
    candidate: RawClaim,
    promoted_claims: list[RawClaim],
    skeleton: PaperSkeletonCandidate,
) -> float:
    if candidate.claim_type == ClaimType.context:
        return 1.0

    method_pool = [
        claim
        for claim in [*promoted_claims, *skeleton.core_methods]
        if claim.claim_type == ClaimType.method and claim.claim_id != candidate.claim_id
    ]
    result_pool = [
        claim
        for claim in [*promoted_claims, *skeleton.topline_results]
        if claim.claim_type == ClaimType.result and claim.claim_id != candidate.claim_id
    ]

    if candidate.claim_type == ClaimType.method:
        if not result_pool:
            return 0.35
        return max((_claim_similarity(candidate, result) for result in result_pool), default=0.0)

    if candidate.claim_type in {ClaimType.result, ClaimType.assumption, ClaimType.negative}:
        if not method_pool:
            return 0.25
        return max((_claim_similarity(candidate, method) for method in method_pool), default=0.0)

    return 0.2


def _promote_argument_candidates(
    candidates: list[SectionArgumentCandidate],
    skeleton: PaperSkeletonCandidate,
    sections: list[Section],
    config: Any,
) -> tuple[list[RawClaim], list[RawClaim], dict[str, float]]:
    t_anchor = _read_promotion_float(config, "t_anchor", 0.55)
    t_novel = _read_promotion_float(config, "t_novel", 0.40)
    t_strength = _read_promotion_float(config, "t_strength", 2.2)
    t_need = _read_promotion_float(config, "t_need", 0.25)

    source_by_section = {_section_label_key(section): section for section in sections}
    anchors = skeleton.claims()
    promoted: list[RawClaim] = []
    unmatched_high_strength: list[RawClaim] = []
    diagnostics: dict[str, float] = {}

    for candidate in candidates:
        raw = _candidate_to_raw_claim(candidate)
        anchor = _anchor_match(raw, anchors, source_by_section)
        novelty = _novelty(raw, [*anchors, *promoted])
        strength = _strength(raw, source_by_section)
        need = _necessity(raw, promoted, skeleton)
        raw = raw.model_copy(update={"claim_strength": round(strength, 3)})

        admitted = anchor >= t_anchor or (novelty >= t_novel and strength >= t_strength and need >= t_need)
        if admitted:
            promoted.append(raw)
        elif raw.claim_type == ClaimType.method and strength >= t_strength:
            unmatched_high_strength.append(raw)

        diagnostics[f"{raw.claim_id}:anchor"] = round(anchor, 3)
        diagnostics[f"{raw.claim_id}:novelty"] = round(novelty, 3)
        diagnostics[f"{raw.claim_id}:strength"] = round(strength, 3)
        diagnostics[f"{raw.claim_id}:need"] = round(need, 3)

    promoted = _normalize_claim_semantics(promoted)
    promoted = _collapse_claim_duplicates(promoted)
    promoted, unique_map = _uniquify_claim_ids(promoted)
    promoted = _remap_structural_hint_ids(promoted, unique_map)
    promoted, norm_map = _normalize_claim_id_scheme(promoted)
    promoted = _remap_structural_hint_ids(promoted, norm_map)
    return promoted, unmatched_high_strength, diagnostics


def _needs_skeleton_repair(
    promoted_claims: list[RawClaim],
    unmatched_high_strength_methods: list[RawClaim],
    skeleton: PaperSkeletonCandidate,
) -> bool:
    skeleton_ids = {claim.claim_id for claim in skeleton.claims()}
    matched_promoted = [
        claim
        for claim in promoted_claims
        if claim.structural_hints is not None
        and claim.structural_hints.elaborates_seed_id is not None
        and claim.structural_hints.elaborates_seed_id in skeleton_ids
    ]
    anchor_coverage = len(matched_promoted) / max(1, len(promoted_claims))

    top_line_results = [
        claim
        for claim in promoted_claims
        if claim.claim_type == ClaimType.result and (claim.claim_strength or 0.0) >= 2.6
    ]
    method_claims = [claim for claim in promoted_claims if claim.claim_type == ClaimType.method]
    orphan_top_line_results = 0
    for result in top_line_results:
        if not method_claims:
            orphan_top_line_results += 1
            continue
        affinity = max((_claim_similarity(result, method) for method in method_claims), default=0.0)
        if affinity < 0.25:
            orphan_top_line_results += 1
    orphan_rate = orphan_top_line_results / max(1, len(top_line_results))

    return (
        anchor_coverage < 0.65
        or orphan_rate > 0.30
        or len(unmatched_high_strength_methods) >= 2
    )


def _attach_support_details(
    support_details: list[SupportDetail],
    claims: list[RawClaim],
) -> list[SupportDetail]:
    if not support_details:
        return []
    if not claims:
        return support_details

    claim_by_id = {claim.claim_id: claim for claim in claims}
    remapped: list[SupportDetail] = []
    for detail in support_details:
        if detail.anchor_claim_id and detail.anchor_claim_id in claim_by_id:
            remapped.append(detail)
            continue

        pseudo_claim = RawClaim(
            claim_id=detail.support_detail_id,
            claim_type=ClaimType.method,
            statement=detail.text,
            source_section=detail.source_section,
            evidence=[{"artifact_id": evidence_id, "role": "supports"} for evidence_id in detail.evidence_ids],
            entity_names=[],
        )
        scored = sorted(
            ((claim, _claim_similarity(pseudo_claim, claim)) for claim in claims),
            key=lambda item: item[1],
            reverse=True,
        )
        candidate_ids = [claim.claim_id for claim, score in scored[:3] if score > 0.15]
        anchor_id = candidate_ids[0] if candidate_ids else None
        confidence = detail.confidence
        if scored:
            confidence = round(max(confidence, min(1.0, scored[0][1])), 3)

        remapped.append(
            detail.model_copy(
                update={
                    "anchor_claim_id": anchor_id,
                    "candidate_anchor_ids": candidate_ids,
                    "confidence": confidence,
                }
            )
        )
    return remapped


async def _run_parallel_with_progress(
    items: list[T],
    *,
    progress: Progress,
    description: str,
    limit: int,
    worker: Callable[[T], Awaitable[Any]],
) -> list[Any | Exception]:
    if not items:
        return []

    task_id = progress.add_task(description, total=len(items))
    semaphore = asyncio.Semaphore(max(1, limit))

    async def invoke(index: int, item: T) -> tuple[int, Any | Exception]:
        try:
            async with semaphore:
                result = await worker(item)
            return index, result
        except Exception as exc:  # pragma: no cover - exercised in API integration paths
            return index, exc

    tasks = [asyncio.create_task(invoke(index, item)) for index, item in enumerate(items)]
    ordered: list[Any | Exception] = [RuntimeError("uninitialized parallel slot")] * len(items)

    for task in asyncio.as_completed(tasks):
        index, result = await task
        ordered[index] = result
        progress.advance(task_id)

    return ordered


async def decompose_paper(pdf_path: str, config_path: str = "config.yaml") -> PaperDecomposition:
    """Run the full decomposition pipeline for a single paper PDF."""
    config = load_config(config_path)
    reset_cost_tracker()

    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    console.print(f"[bold]Decomposing[/bold] {pdf}")

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        parse_task = progress.add_task("Phase 0/6: Parse PDF", total=1)
        document = parse_pdf(str(pdf), config)
        progress.advance(parse_task)

        section_count = len(document.sections)
        console.print(
            "Parsed "
            f"{section_count} sections and {len(document.all_artifacts)} artifacts "
            f"for '{document.metadata.title}'."
        )

        preflight_task = progress.add_task("Phase 0b/6: Model preflight", total=1)
        required_tiers = _required_preflight_tiers(config)
        await preflight_model_tiers(required_tiers, config=config)
        progress.advance(preflight_task)
        console.print(f"Preflight passed for tiers: {', '.join(required_tiers)}")

        phase1_task = progress.add_task("Phase 1/6: Skeleton extraction", total=1)
        phase1_before = get_cost_tracker()
        seed_text = _find_seed_text(document)
        try:
            skeleton = await extract_skeleton(seed_text, config)
        except Exception as exc:
            console.print(f"[yellow]Skeleton extraction failed:[/yellow] {exc}")
            seed_fallback = await extract_seed(seed_text, config)
            fallback_claims = list(seed_fallback.claims)
            skeleton = PaperSkeletonCandidate(
                context_roots=[claim for claim in fallback_claims if claim.claim_type == ClaimType.context][:2],
                core_methods=[claim for claim in fallback_claims if claim.claim_type == ClaimType.method][:3],
                topline_results=[claim for claim in fallback_claims if claim.claim_type == ClaimType.result][:3],
                assumptions=[claim for claim in fallback_claims if claim.claim_type == ClaimType.assumption][:2],
                negatives=[claim for claim in fallback_claims if claim.claim_type == ClaimType.negative][:2],
            )
        progress.advance(phase1_task)
        phase1_after = get_cost_tracker()
        console.print(
            "Phase 1 skeleton: "
            f"context={len(skeleton.context_roots)} method={len(skeleton.core_methods)} "
            f"result={len(skeleton.topline_results)} assumption={len(skeleton.assumptions)} "
            f"negative={len(skeleton.negatives)} "
            f"| cost +${_cost_delta(phase1_before, phase1_after):.4f}"
        )

        section_inputs = [
            section
            for section in document.sections
            if section.role.value != "abstract" and not _is_reference_section(section)
        ]
        phase2_before = get_cost_tracker()

        async def section_worker(section: Section):
            section_artifacts = list(section.artifacts) if section.artifacts else list(document.all_artifacts)
            return await extract_section_digest(
                section=section,
                skeleton=skeleton,
                artifacts=section_artifacts,
                config=config,
            )

        section_results = await _run_parallel_with_progress(
            section_inputs,
            progress=progress,
            description="Phase 2/6: Section digests",
            limit=_read_max_concurrent(config),
            worker=section_worker,
        )

        argument_candidates: list[SectionArgumentCandidate] = []
        support_details: list[SupportDetail] = []
        for section_obj, result in zip(section_inputs, section_results, strict=False):
            if isinstance(result, Exception):
                console.print(
                    f"[red]Section digest failed[/red] ({_section_label(section_obj)}): {result}"
                )
                continue
            argument_candidates.extend(result.argument_candidates)
            support_details.extend(result.support_details)

        if section_inputs and not argument_candidates and not skeleton.claims():
            raise RuntimeError("Phase 2 produced zero argument candidates and empty skeleton; aborting run.")

        promoted_claims, unmatched_high_strength, _ = _promote_argument_candidates(
            argument_candidates,
            skeleton,
            document.sections,
            config,
        )

        repaired = False
        if _needs_skeleton_repair(promoted_claims, unmatched_high_strength, skeleton):
            repaired = True
            skeleton = await repair_skeleton(skeleton, unmatched_high_strength, config)
            promoted_claims, unmatched_high_strength, _ = _promote_argument_candidates(
                argument_candidates,
                skeleton,
                document.sections,
                config,
            )

        all_promoted = _normalize_claim_semantics([*skeleton.claims(), *promoted_claims])
        all_promoted = _collapse_claim_duplicates(all_promoted)
        all_promoted, unique_map = _uniquify_claim_ids(all_promoted)
        all_promoted = _remap_structural_hint_ids(all_promoted, unique_map)
        all_promoted, norm_map = _normalize_claim_id_scheme(all_promoted)
        all_promoted = _remap_structural_hint_ids(all_promoted, norm_map)

        phase2_after = get_cost_tracker()
        console.print(
            "Phase 2 promoted claims: "
            f"{len(all_promoted)} | candidates={len(argument_candidates)} "
            f"support_details={len(support_details)} repaired={repaired} "
            f"| cost +${_cost_delta(phase2_before, phase2_after):.4f}"
        )

        phase3_task = progress.add_task("Phase 3/6: Hybrid dedup", total=1)
        phase3_before = get_cost_tracker()
        try:
            deduplicated_claims, dedup_groups = await hybrid_dedup_promoted(all_promoted, config)
            deduplicated_claims = _merge_structural_hints(deduplicated_claims, dedup_groups)
            deduplicated_claims = _refine_tree_candidate_claims(deduplicated_claims, document.sections)
        except Exception as exc:
            console.print(f"[yellow]Hybrid dedup failed:[/yellow] {exc}")
            deduplicated_claims = list(all_promoted)
            dedup_groups = []
        progress.advance(phase3_task)
        phase3_after = get_cost_tracker()
        console.print(
            f"Phase 3 claims: {len(all_promoted)} -> {len(deduplicated_claims)} "
            f"({len(dedup_groups)} groups) | cost +${_cost_delta(phase3_before, phase3_after):.4f}"
        )

        phase4_before = get_cost_tracker()
        method_claims = [claim for claim in deduplicated_claims if claim.claim_type == ClaimType.method]

        async def facet_worker(claim: RawClaim) -> FacetedClaim:
            source = _find_section_for_claim(claim, document.sections)
            return await extract_facets(claim, source, config)

        facet_results = await _run_parallel_with_progress(
            method_claims,
            progress=progress,
            description="Phase 4/6: Method facets",
            limit=_read_max_concurrent(config),
            worker=facet_worker,
        )

        faceted_claims: list[FacetedClaim] = []
        for claim, result in zip(method_claims, facet_results, strict=False):
            if isinstance(result, Exception):
                console.print(f"[red]Facet extraction failed[/red] ({claim.claim_id}): {result}")
                continue
            faceted_claims.append(result)

        phase4_after = get_cost_tracker()
        console.print(
            "Phase 4 faceted claims: "
            f"{len(faceted_claims)}/{len(method_claims)} | cost +${_cost_delta(phase4_before, phase4_after):.4f}"
        )

        phase5_task = progress.add_task("Phase 5/6: Deterministic tree", total=1)
        phase5_before = get_cost_tracker()
        negative_claims = [claim for claim in deduplicated_claims if claim.claim_type == ClaimType.negative]
        attached_support_details = _attach_support_details(support_details, deduplicated_claims)
        try:
            decomposition = await assemble_tree_deterministic(
                metadata=document.metadata,
                claims=deduplicated_claims,
                faceted=faceted_claims,
                negatives=negative_claims,
                artifacts=document.all_artifacts,
                config=config,
                claim_groups=dedup_groups,
                support_details=attached_support_details,
            )
        except Exception as exc:
            console.print(f"[yellow]Deterministic tree assembly failed:[/yellow] {exc}")
            fallback_cost = float(get_cost_tracker().get("total_cost_usd", 0.0))
            decomposition = _build_fallback_decomposition(
                document=document,
                claims=deduplicated_claims,
                faceted_claims=faceted_claims,
                negative_claims=negative_claims,
                extraction_cost_usd=fallback_cost,
            ).model_copy(update={"support_details": attached_support_details})
        progress.advance(phase5_task)

        phase5_after = get_cost_tracker()
        total_cost = float(phase5_after.get("total_cost_usd", 0.0))
        decomposition = decomposition.model_copy(update={"extraction_cost_usd": total_cost})
        node_count, depth, edge_count = _tree_stats(decomposition.claim_tree)
        console.print(
            f"Phase 5 tree nodes={node_count} depth={depth} edges={edge_count} "
            f"| support_details={len(decomposition.support_details)} "
            f"| cost +${_cost_delta(phase5_before, phase5_after):.4f}"
        )

    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _sanitize_filename(document.metadata.title, pdf.stem)
    output_path = output_dir / f"{filename}.json"
    suffix = 2
    while output_path.exists():
        output_path = output_dir / f"{filename}_{suffix}.json"
        suffix += 1

    output_path.write_text(decomposition.model_dump_json(indent=2), encoding="utf-8")
    total_cost = float(get_cost_tracker().get("total_cost_usd", 0.0))
    console.print(f"Saved decomposition to {output_path} | total extraction cost=${total_cost:.4f}")

    return decomposition


__all__ = ["decompose_paper"]
