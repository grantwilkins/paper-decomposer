from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from ..models import call_model
from ..schema import (
    ClaimGroup,
    CrossTypeOutput,
    DedupBatchOutput,
    DedupOutput,
    ModelTier,
    ParentChildLink,
    RawClaim,
)

WITHIN_TYPE_DEDUP_PROMPT = """You are given {n} {claim_type} claims
from the same paper. Some may be restatements of the same finding.

Canonicalization rules:
- Group claims that assert the SAME proposition, including close paraphrases
  and seed/abstract placeholders repeated later in sections.
- Ignore claim_id naming style when deciding semantic overlap.
- For each duplicate group, choose exactly one canonical_id.
- Prefer canonicals with concrete provenance:
  section-specific source (not seed/abstract placeholders),
  richer evidence pointers, and more concrete detail.
- If one claim is a broad summary and another is a true sub-claim
  (not a duplicate), keep separate groups and set child.parent_id
  to the broader canonical.
- Repeated method summaries should collapse to one main canonical
  plus child mechanism groups when appropriate.

If a claim has no duplicates, it forms a group of size 1.
Return JSON only.
"""

CROSS_TYPE_PROMPT = """You are given the deduplicated claims from
a research paper, organized by type. Identify PARENT-CHILD
relationships across types.

Rules:
- A general RESULT is parent of specific per-experiment RESULTS.
- A top-level METHOD is parent of its sub-mechanisms.
- Do NOT link CONTEXT -> METHOD here.

Return JSON only.
"""

DEDUP_PROMPT = WITHIN_TYPE_DEDUP_PROMPT

_VALID_TIERS: set[ModelTier] = {"small", "medium", "heavy"}
_MAX_WITHIN_TYPE_BATCH = 20
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
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "we",
    "with",
}
_PLACEHOLDER_SOURCE_HINTS = {"abstract", "seed", "summary", "overview"}


def _resolve_model_tier(config: Any) -> ModelTier:
    if config is None:
        return "medium"

    pipeline = getattr(config, "pipeline", None)
    if pipeline is not None:
        dedup_cfg = getattr(pipeline, "dedup", None)
        if isinstance(dedup_cfg, Mapping):
            tier = dedup_cfg.get("model_tier")
            if isinstance(tier, str) and tier in _VALID_TIERS:
                return cast(ModelTier, tier)

    if isinstance(config, Mapping):
        pipeline_cfg = config.get("pipeline")
        if isinstance(pipeline_cfg, Mapping):
            dedup_cfg = pipeline_cfg.get("dedup")
            if isinstance(dedup_cfg, Mapping):
                tier = dedup_cfg.get("model_tier")
                if isinstance(tier, str) and tier in _VALID_TIERS:
                    return cast(ModelTier, tier)

    return "medium"


def _format_claim_line(idx: int, claim: RawClaim) -> str:
    statement = claim.statement.strip().replace("\n", " ")
    hint_parts: list[str] = []
    hints = claim.structural_hints
    if hints is not None:
        if hints.elaborates_seed_id:
            hint_parts.append(f"seed={hints.elaborates_seed_id}")
        if hints.local_role is not None:
            hint_parts.append(f"role={hints.local_role.value}")
        if hints.preferred_parent_type is not None:
            hint_parts.append(f"pref_parent={hints.preferred_parent_type.value}")
    hint_suffix = f" [hints: {', '.join(hint_parts)}]" if hint_parts else ""
    return f'{idx}. [{claim.claim_id}] {claim.claim_type.value.upper()}: "{statement}"{hint_suffix}'


def _singleton_groups(claims: list[RawClaim]) -> list[ClaimGroup]:
    return [ClaimGroup(canonical_id=claim.claim_id, member_ids=[claim.claim_id], parent_id=None) for claim in claims]


def _chunk_claims(claims: list[RawClaim], size: int) -> list[list[RawClaim]]:
    if size <= 0:
        return [claims]
    return [claims[idx : idx + size] for idx in range(0, len(claims), size)]


def _dedupe_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in ids:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _claim_tokens(text: str) -> set[str]:
    lowered = " ".join(text.strip().lower().split())
    return {
        token
        for token in _TOKEN_SPLIT_RE.split(lowered)
        if token and len(token) > 2 and token not in _STOPWORDS
    }


def _source_section_score(source_section: str) -> int:
    lowered = source_section.strip().lower()
    if not lowered:
        return 0

    score = 1
    if any(marker in lowered for marker in _PLACEHOLDER_SOURCE_HINTS):
        score -= 6
    if "introduction" in lowered:
        score -= 2
    if re.search(r"\b\d", lowered):
        score += 4
    if any(
        marker in lowered
        for marker in (
            "method",
            "design",
            "implementation",
            "experiment",
            "evaluation",
            "ablation",
            "results",
            "analysis",
        )
    ):
        score += 2
    return score


def _canonical_score(claim: RawClaim) -> float:
    evidence_ids = {pointer.artifact_id.strip().lower() for pointer in claim.evidence if pointer.artifact_id.strip()}
    token_count = len(_claim_tokens(claim.statement))
    entity_count = len({entity.strip().lower() for entity in claim.entity_names if entity.strip()})
    hint = claim.structural_hints
    detail_penalty = 0.0
    if hint is not None and hint.local_role is not None and hint.local_role.value == "implementation_detail":
        detail_penalty = 4.0
    return float(
        _source_section_score(claim.source_section) * 12
        + len(evidence_ids) * 7
        + min(token_count, 40)
        + min(entity_count, 5) * 2
        - detail_penalty
    )


def _statement_numbers(statement: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", statement))


def _is_near_duplicate(left: RawClaim, right: RawClaim) -> bool:
    if left.claim_type != right.claim_type:
        return False

    left_tokens = _claim_tokens(left.statement)
    right_tokens = _claim_tokens(right.statement)
    if not left_tokens or not right_tokens:
        return False

    overlap = left_tokens & right_tokens
    if not overlap:
        return False

    union = left_tokens | right_tokens
    jaccard = len(overlap) / len(union)
    coverage = len(overlap) / min(len(left_tokens), len(right_tokens))

    left_numbers = _statement_numbers(left.statement)
    right_numbers = _statement_numbers(right.statement)
    if left_numbers and right_numbers and left_numbers.isdisjoint(right_numbers):
        return False

    left_text = " ".join(left.statement.lower().split())
    right_text = " ".join(right.statement.lower().split())
    containment = left_text in right_text or right_text in left_text

    if jaccard >= 0.84:
        return True
    if coverage >= 0.9:
        return True
    return containment and coverage >= 0.75


def _collapse_residual_duplicates(claims: list[RawClaim]) -> tuple[list[RawClaim], list[ClaimGroup]]:
    if len(claims) <= 1:
        return claims, []

    parent = list(range(len(claims)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left_idx: int, right_idx: int) -> None:
        root_left = find(left_idx)
        root_right = find(right_idx)
        if root_left == root_right:
            return
        parent[root_right] = root_left

    for left_idx in range(len(claims)):
        for right_idx in range(left_idx + 1, len(claims)):
            if _is_near_duplicate(claims[left_idx], claims[right_idx]):
                union(left_idx, right_idx)

    clusters: dict[int, list[RawClaim]] = defaultdict(list)
    for idx, claim in enumerate(claims):
        clusters[find(idx)].append(claim)

    collapsed_claims: list[RawClaim] = []
    groups: list[ClaimGroup] = []
    for member_claims in clusters.values():
        if len(member_claims) == 1:
            collapsed_claims.append(member_claims[0])
            continue

        canonical = max(member_claims, key=_canonical_score)
        member_ids = sorted(
            {
                claim.claim_id
                for claim in member_claims
                if claim.claim_id != canonical.claim_id
            }
        )
        collapsed_claims.append(canonical)
        groups.append(
            ClaimGroup(
                canonical_id=canonical.claim_id,
                member_ids=member_ids,
                parent_id=None,
            )
        )

    ordered_collapsed = sorted(
        collapsed_claims,
        key=lambda claim: claims.index(claim),
    )
    return ordered_collapsed, groups


def _canonicalize_groups(
    groups: list[ClaimGroup],
    claim_by_id: dict[str, RawClaim],
) -> list[ClaimGroup]:
    if not groups:
        return []

    preliminary: list[ClaimGroup] = []
    member_to_canonical: dict[str, str] = {}

    for group in groups:
        original_canonical = group.canonical_id.strip()
        if original_canonical not in claim_by_id:
            raise ValueError(
                f"Canonical claim_id '{group.canonical_id}' not found in original claims."
            )

        member_ids = _dedupe_ids([group.canonical_id, *group.member_ids])
        available_ids = [member_id for member_id in member_ids if member_id in claim_by_id]
        if not available_ids:
            raise ValueError(
                f"Canonical claim_id '{group.canonical_id}' not found in original claims."
            )

        selected = max(
            available_ids,
            key=lambda claim_id: (
                _canonical_score(claim_by_id[claim_id]),
                claim_id == original_canonical,
                claim_id,
            ),
        )
        merged_member_ids = [member_id for member_id in member_ids if member_id != selected]
        preliminary.append(
            ClaimGroup(
                canonical_id=selected,
                member_ids=merged_member_ids,
                parent_id=group.parent_id,
            )
        )

        for member_id in member_ids:
            member_to_canonical[member_id] = selected

    merged_by_canonical: dict[str, ClaimGroup] = {}
    order: list[str] = []

    for group in preliminary:
        canonical_id = group.canonical_id
        normalized_parent = None
        if isinstance(group.parent_id, str) and group.parent_id.strip():
            parent_candidate = member_to_canonical.get(group.parent_id.strip(), group.parent_id.strip())
            if parent_candidate != canonical_id:
                normalized_parent = parent_candidate

        if canonical_id not in merged_by_canonical:
            merged_by_canonical[canonical_id] = ClaimGroup(
                canonical_id=canonical_id,
                member_ids=_dedupe_ids(group.member_ids),
                parent_id=normalized_parent,
            )
            order.append(canonical_id)
            continue

        existing = merged_by_canonical[canonical_id]
        existing_members = _dedupe_ids([*existing.member_ids, *group.member_ids])
        parent_id = existing.parent_id or normalized_parent
        merged_by_canonical[canonical_id] = existing.model_copy(
            update={
                "member_ids": [member_id for member_id in existing_members if member_id != canonical_id],
                "parent_id": parent_id if parent_id != canonical_id else None,
            }
        )

    merged_groups = [merged_by_canonical[canonical_id] for canonical_id in order]
    canonical_ids = {group.canonical_id for group in merged_groups}
    normalized: list[ClaimGroup] = []

    for group in merged_groups:
        parent_id = group.parent_id
        if parent_id is not None and parent_id not in canonical_ids:
            raise ValueError(
                f"Dedup parent_id '{parent_id}' for canonical '{group.canonical_id}' "
                "does not resolve to a canonical group."
            )
        normalized.append(group)

    return normalized


def _members_by_canonical(groups: list[ClaimGroup]) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {}
    for group in groups:
        members[group.canonical_id] = {group.canonical_id, *group.member_ids}
    return members


def _seed_like_source(source_section: str) -> bool:
    lowered = source_section.strip().lower()
    return any(marker in lowered for marker in ("abstract", "introduction", "seed", "overview", "summary"))


def _infer_within_type_parent_links(claims: list[RawClaim]) -> CrossTypeOutput:
    by_type: dict[str, list[RawClaim]] = defaultdict(list)
    for claim in claims:
        by_type[claim.claim_type.value].append(claim)

    links: list[ParentChildLink] = []
    for typed_claims in by_type.values():
        token_cache = {claim.claim_id: _claim_tokens(claim.statement) for claim in typed_claims}
        for child in typed_claims:
            child_tokens = token_cache.get(child.claim_id, set())
            if len(child_tokens) < 4:
                continue

            best_parent: RawClaim | None = None
            best_score = 0.0
            child_specificity = _canonical_score(child)

            for parent in typed_claims:
                if parent.claim_id == child.claim_id:
                    continue
                parent_tokens = token_cache.get(parent.claim_id, set())
                if len(parent_tokens) < 3 or len(parent_tokens) >= len(child_tokens):
                    continue

                overlap = parent_tokens & child_tokens
                if len(overlap) < 3:
                    continue

                coverage = len(overlap) / len(parent_tokens)
                if coverage < 0.65:
                    continue

                specificity_gap = child_specificity - _canonical_score(parent)
                if specificity_gap < 6:
                    continue

                seed_to_section_bonus = 2.0 if _seed_like_source(parent.source_section) and not _seed_like_source(child.source_section) else 0.0
                score = coverage * 10.0 + specificity_gap + seed_to_section_bonus
                if score > best_score:
                    best_score = score
                    best_parent = parent

            if best_parent is None:
                continue

            links.append(
                ParentChildLink(
                    parent_id=best_parent.claim_id,
                    child_id=child.claim_id,
                    relationship="abstraction_refinement",
                )
            )

    return CrossTypeOutput(parent_child_links=links)


def build_within_type_dedup_prompt(claims: list[RawClaim], claim_type: str) -> list[dict[str, str]]:
    claim_lines = (
        "\n".join(_format_claim_line(idx, claim) for idx, claim in enumerate(claims, start=1))
        if claims
        else "(no claims provided)"
    )
    return [
        {
            "role": "system",
            "content": WITHIN_TYPE_DEDUP_PROMPT.format(n=len(claims), claim_type=claim_type),
        },
        {
            "role": "user",
            "content": (
                "CLAIMS:\n"
                f"{claim_lines}\n\n"
                "Output `groups` with canonical_id/member_ids/parent_id."
            ),
        },
    ]


def build_cross_type_prompt(claims: list[RawClaim]) -> list[dict[str, str]]:
    claim_lines = (
        "\n".join(_format_claim_line(idx, claim) for idx, claim in enumerate(claims, start=1))
        if claims
        else "(no claims provided)"
    )
    return [
        {"role": "system", "content": CROSS_TYPE_PROMPT},
        {
            "role": "user",
            "content": (
                "CLAIMS:\n"
                f"{claim_lines}\n\n"
                "Return `parent_child_links` with parent_id, child_id, relationship."
            ),
        },
    ]


def build_dedup_prompt(claims: list[RawClaim]) -> list[dict[str, str]]:
    return build_within_type_dedup_prompt(claims, claim_type="mixed")


def _normalize_groups(groups: list[ClaimGroup]) -> list[ClaimGroup]:
    if not groups:
        return []

    member_to_canonical: dict[str, str] = {}
    canonical_ids: set[str] = set()
    for group in groups:
        canonical_ids.add(group.canonical_id)
        for member_id in {group.canonical_id, *group.member_ids}:
            member_to_canonical[member_id] = group.canonical_id

    normalized: list[ClaimGroup] = []
    for group in groups:
        if group.parent_id is None:
            normalized.append(group)
            continue

        resolved_parent = member_to_canonical.get(group.parent_id, group.parent_id)
        if resolved_parent not in canonical_ids:
            raise ValueError(
                f"Dedup parent_id '{group.parent_id}' for canonical '{group.canonical_id}' "
                "does not resolve to a canonical group."
            )

        normalized.append(
            group.model_copy(
                update={
                    "parent_id": resolved_parent,
                    "member_ids": _dedupe_ids(group.member_ids),
                }
            )
        )

    return normalized


def _apply_parent_links(groups: list[ClaimGroup], links: CrossTypeOutput) -> list[ClaimGroup]:
    by_canonical = {group.canonical_id: group for group in groups}
    for link in links.parent_child_links:
        parent_id = link.parent_id.strip()
        child_id = link.child_id.strip()
        if not parent_id or not child_id or parent_id == child_id:
            continue
        if parent_id not in by_canonical or child_id not in by_canonical:
            continue
        child = by_canonical[child_id]
        by_canonical[child_id] = child.model_copy(update={"parent_id": parent_id})
    return list(by_canonical.values())


def apply_dedup(original_claims: list[RawClaim], dedup_output: DedupOutput) -> list[RawClaim]:
    if not dedup_output.groups:
        return list(original_claims)

    claim_by_id = {claim.claim_id: claim for claim in original_claims}
    normalized_groups = _canonicalize_groups(dedup_output.groups, claim_by_id)

    canonical_claims: list[RawClaim] = []
    seen: set[str] = set()

    for group in normalized_groups:
        if group.canonical_id in seen:
            continue
        if group.canonical_id not in claim_by_id:
            raise ValueError(f"Canonical claim_id '{group.canonical_id}' not found in original claims.")

        canonical_claims.append(claim_by_id[group.canonical_id])
        seen.add(group.canonical_id)

    return canonical_claims


@dataclass(slots=True)
class DedupBatchResult:
    claim_type: str
    input_claims: list[RawClaim]
    canonical_claims: list[RawClaim]
    groups: list[ClaimGroup]


async def dedup_type_batch(
    claims: list[RawClaim],
    claim_type: str,
    config: Any,
) -> DedupBatchResult:
    messages = build_within_type_dedup_prompt(claims, claim_type)
    result = await call_model(
        tier=_resolve_model_tier(config),
        messages=messages,
        response_schema=DedupBatchOutput,
        config=config,
    )
    if not isinstance(result, DedupBatchOutput):
        raise TypeError("Expected DedupBatchOutput from structured model call.")

    claim_by_id = {claim.claim_id: claim for claim in claims}
    groups = _canonicalize_groups(result.groups, claim_by_id)
    if not groups:
        groups = _singleton_groups(claims)

    canonical_claims = apply_dedup(claims, DedupOutput(groups=groups))
    return DedupBatchResult(
        claim_type=claim_type,
        input_claims=claims,
        canonical_claims=canonical_claims,
        groups=groups,
    )


async def chunked_dedup(
    all_claims: list[RawClaim],
    config: Any,
) -> tuple[list[RawClaim], list[ClaimGroup]]:
    if not all_claims:
        return [], []

    by_type: dict[str, list[RawClaim]] = defaultdict(list)
    for claim in all_claims:
        by_type[claim.claim_type.value].append(claim)

    all_claims_by_id = {claim.claim_id: claim for claim in all_claims}

    pass1_tasks: list[asyncio.Task[DedupBatchResult]] = []
    pass1_inputs: list[tuple[str, list[RawClaim]]] = []
    pass1_groups: list[ClaimGroup] = []
    for claim_type, claims in by_type.items():
        if len(claims) <= 1:
            pass1_groups.extend(_singleton_groups(claims))
            continue
        batches = _chunk_claims(claims, _MAX_WITHIN_TYPE_BATCH)
        for batch in batches:
            pass1_inputs.append((claim_type, batch))
            pass1_tasks.append(asyncio.create_task(dedup_type_batch(batch, claim_type, config)))

    pass1_results: list[DedupBatchResult | Exception] = []
    if pass1_tasks:
        pass1_results = cast(
            list[DedupBatchResult | Exception],
            await asyncio.gather(*pass1_tasks, return_exceptions=True),
        )

    for (_, claims), result in zip(pass1_inputs, pass1_results, strict=False):
        if isinstance(result, Exception):
            pass1_groups.extend(_singleton_groups(claims))
            continue
        pass1_groups.extend(result.groups)

    normalized_pass1_groups = _canonicalize_groups(pass1_groups, all_claims_by_id)
    pass1_canonical_claims = apply_dedup(all_claims, DedupOutput(groups=normalized_pass1_groups))
    pass1_members = _members_by_canonical(normalized_pass1_groups)

    pass2_by_type: dict[str, list[RawClaim]] = defaultdict(list)
    for claim in pass1_canonical_claims:
        pass2_by_type[claim.claim_type.value].append(claim)

    pass1_canonical_by_id = {claim.claim_id: claim for claim in pass1_canonical_claims}
    pass2_tasks: list[asyncio.Task[DedupBatchResult]] = []
    pass2_inputs: list[tuple[str, list[RawClaim]]] = []
    pass2_groups: list[ClaimGroup] = []

    for claim_type, claims in pass2_by_type.items():
        if len(claims) <= 1:
            pass2_groups.extend(_singleton_groups(claims))
            continue
        pass2_inputs.append((claim_type, claims))
        pass2_tasks.append(asyncio.create_task(dedup_type_batch(claims, claim_type, config)))

    pass2_results: list[DedupBatchResult | Exception] = []
    if pass2_tasks:
        pass2_results = cast(
            list[DedupBatchResult | Exception],
            await asyncio.gather(*pass2_tasks, return_exceptions=True),
        )

    for (_, claims), result in zip(pass2_inputs, pass2_results, strict=False):
        if isinstance(result, Exception):
            pass2_groups.extend(_singleton_groups(claims))
            continue
        pass2_groups.extend(result.groups)

    normalized_pass2_groups = _canonicalize_groups(pass2_groups, pass1_canonical_by_id)

    expanded_groups: list[ClaimGroup] = []
    for group in normalized_pass2_groups:
        contributor_ids = {group.canonical_id, *group.member_ids}
        expanded_members: set[str] = set()
        for contributor_id in contributor_ids:
            expanded_members.update(pass1_members.get(contributor_id, {contributor_id}))
        expanded_groups.append(
            ClaimGroup(
                canonical_id=group.canonical_id,
                member_ids=sorted(member_id for member_id in expanded_members if member_id != group.canonical_id),
                parent_id=group.parent_id,
            )
        )

    final_groups = _canonicalize_groups(expanded_groups, all_claims_by_id)
    final_canonical_claims = apply_dedup(all_claims, DedupOutput(groups=final_groups))

    inferred = _infer_within_type_parent_links(final_canonical_claims)
    if inferred.parent_child_links:
        final_groups = _apply_parent_links(final_groups, inferred)

    if len(final_canonical_claims) <= 40 and final_canonical_claims:
        try:
            cross = await call_model(
                tier=_resolve_model_tier(config),
                messages=build_cross_type_prompt(final_canonical_claims),
                response_schema=CrossTypeOutput,
                config=config,
            )
            if isinstance(cross, CrossTypeOutput):
                final_groups = _apply_parent_links(final_groups, cross)
        except Exception:
            pass

    normalized_final_groups = _normalize_groups(final_groups)
    deduped_canonical = apply_dedup(all_claims, DedupOutput(groups=normalized_final_groups))
    _, residual_groups = _collapse_residual_duplicates(deduped_canonical)
    if residual_groups:
        merged_groups = _canonicalize_groups([*normalized_final_groups, *residual_groups], all_claims_by_id)
        normalized_final_groups = _normalize_groups(merged_groups)
        deduped_canonical = apply_dedup(all_claims, DedupOutput(groups=normalized_final_groups))
    return deduped_canonical, normalized_final_groups


async def deduplicate_claims(claims: list[RawClaim], config: Any) -> DedupOutput:
    _, groups = await chunked_dedup(claims, config)
    return DedupOutput(groups=groups)


__all__ = [
    "DEDUP_PROMPT",
    "WITHIN_TYPE_DEDUP_PROMPT",
    "CROSS_TYPE_PROMPT",
    "build_dedup_prompt",
    "build_within_type_dedup_prompt",
    "build_cross_type_prompt",
    "apply_dedup",
    "dedup_type_batch",
    "chunked_dedup",
    "deduplicate_claims",
]
