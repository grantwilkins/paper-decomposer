from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

from ..models import call_model_with_fallback, flat_claim_to_raw
from ..schema import FlatSeedOutput, ModelTier, PaperSkeletonCandidate, RawClaim, SeedOutput

SEED_SYSTEM_PROMPT = """You are a research paper analyst. Extract a compact paper skeleton
from the abstract or introduction text.

Output 3-10 claims spanning these roles when present:
- CONTEXT: specific gap, bottleneck, or limitation
- METHOD: the core method/system contribution(s)
- RESULT: top-line outcomes with quantitative specifics when available
- ASSUMPTION: key conditions the method/result depends on
- NEGATIVE: rejected alternatives or explicit limitations

Rules:
- Prefer argument-level claims over local procedure details.
- Use one proposition per claim.
- Preserve concrete numbers/comparisons in RESULT claims.
- Name systems/methods when the paper names them.
"""

_VALID_TIERS: set[ModelTier] = {"small", "medium", "heavy"}
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


call_model = call_model_with_fallback


def _resolve_model_tier(config: Any) -> ModelTier:
    if config is None:
        return "small"

    pipeline = getattr(config, "pipeline", None)
    if pipeline is not None:
        seed_cfg = getattr(pipeline, "seed", None)
        if isinstance(seed_cfg, Mapping):
            tier = seed_cfg.get("model_tier")
            if isinstance(tier, str) and tier in _VALID_TIERS:
                return cast(ModelTier, tier)

    if isinstance(config, Mapping):
        pipeline_cfg = config.get("pipeline")
        if isinstance(pipeline_cfg, Mapping):
            seed_cfg = pipeline_cfg.get("seed")
            if isinstance(seed_cfg, Mapping):
                tier = seed_cfg.get("model_tier")
                if isinstance(tier, str) and tier in _VALID_TIERS:
                    return cast(ModelTier, tier)

    return "small"


def _seed_cfg(config: Any) -> Mapping[str, Any]:
    pipeline = getattr(config, "pipeline", None)
    if pipeline is not None:
        seed_cfg = getattr(pipeline, "seed", None)
        if isinstance(seed_cfg, Mapping):
            return seed_cfg

    if isinstance(config, Mapping):
        pipeline_cfg = config.get("pipeline")
        if isinstance(pipeline_cfg, Mapping):
            seed_cfg = pipeline_cfg.get("seed")
            if isinstance(seed_cfg, Mapping):
                return seed_cfg

    return {}


def _cap(config: Any, key: str, default: int) -> int:
    seed_cfg = _seed_cfg(config)
    raw = seed_cfg.get(key, default)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _claim_tokens(text: str) -> set[str]:
    lowered = " ".join(text.lower().split())
    return {
        token
        for token in _TOKEN_SPLIT_RE.split(lowered)
        if token and len(token) > 2 and token not in _STOPWORDS
    }


def _strength(claim: RawClaim) -> float:
    tokens = len(_claim_tokens(claim.statement))
    evidence = len({pointer.artifact_id.strip().lower() for pointer in claim.evidence if pointer.artifact_id.strip()})
    entities = len({entity.strip().lower() for entity in claim.entity_names if entity.strip()})
    number_bonus = 1.2 if re.search(r"\d", claim.statement) else 0.0

    type_bonus = 0.0
    if claim.claim_type.value == "result":
        type_bonus = 0.8
    elif claim.claim_type.value == "method":
        type_bonus = 0.6
    elif claim.claim_type.value == "context":
        type_bonus = 0.4

    return float(tokens * 0.25 + evidence * 0.8 + min(entities, 3) * 0.3 + number_bonus + type_bonus)


def _select_top(claims: list[RawClaim], cap: int) -> list[RawClaim]:
    if cap <= 0:
        return []
    if len(claims) <= cap:
        return list(claims)

    ranked = sorted(
        claims,
        key=lambda claim: (_strength(claim), len(claim.statement), claim.claim_id),
        reverse=True,
    )
    return ranked[:cap]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _is_duplicate(left: RawClaim, right: RawClaim) -> bool:
    if left.claim_type != right.claim_type:
        return False
    left_tokens = _claim_tokens(left.statement)
    right_tokens = _claim_tokens(right.statement)
    if not left_tokens or not right_tokens:
        return False
    return _jaccard(left_tokens, right_tokens) >= 0.85


def _dedupe_claims(claims: list[RawClaim]) -> list[RawClaim]:
    kept: list[RawClaim] = []
    for claim in claims:
        if any(_is_duplicate(claim, existing) for existing in kept):
            continue
        kept.append(claim)
    return kept


def _build_skeleton(claims: list[RawClaim], config: Any) -> PaperSkeletonCandidate:
    deduped = _dedupe_claims(claims)

    context = [claim for claim in deduped if claim.claim_type.value == "context"]
    methods = [claim for claim in deduped if claim.claim_type.value == "method"]
    results = [claim for claim in deduped if claim.claim_type.value == "result"]
    assumptions = [claim for claim in deduped if claim.claim_type.value == "assumption"]
    negatives = [claim for claim in deduped if claim.claim_type.value == "negative"]

    context_cap = _cap(config, "context_cap", 2)
    method_cap = _cap(config, "core_method_cap", 3)
    result_cap = _cap(config, "topline_result_cap", 3)
    assumption_cap = _cap(config, "assumption_cap", 2)
    negative_cap = _cap(config, "negative_cap", 2)

    return PaperSkeletonCandidate(
        context_roots=_select_top(context, context_cap),
        core_methods=_select_top(methods, method_cap),
        topline_results=_select_top(results, result_cap),
        assumptions=_select_top(assumptions, assumption_cap),
        negatives=_select_top(negatives, negative_cap),
    )


def _augment_skeleton(
    skeleton: PaperSkeletonCandidate,
    candidates: list[RawClaim],
    *,
    context_add_cap: int,
    method_add_cap: int,
    result_add_cap: int,
) -> PaperSkeletonCandidate:
    context_add: list[RawClaim] = []
    method_add: list[RawClaim] = []
    result_add: list[RawClaim] = []

    existing = skeleton.claims()
    ranked = sorted(candidates, key=lambda claim: (_strength(claim), len(claim.statement)), reverse=True)

    for candidate in ranked:
        if any(_is_duplicate(candidate, existing_claim) for existing_claim in [*existing, *context_add, *method_add, *result_add]):
            continue

        if candidate.claim_type.value == "context" and len(context_add) < context_add_cap:
            context_add.append(candidate)
            continue
        if candidate.claim_type.value == "method" and len(method_add) < method_add_cap:
            method_add.append(candidate)
            continue
        if candidate.claim_type.value == "result" and len(result_add) < result_add_cap:
            result_add.append(candidate)
            continue

        if (
            len(context_add) >= context_add_cap
            and len(method_add) >= method_add_cap
            and len(result_add) >= result_add_cap
        ):
            break

    return PaperSkeletonCandidate(
        context_roots=[*skeleton.context_roots, *context_add],
        core_methods=[*skeleton.core_methods, *method_add],
        topline_results=[*skeleton.topline_results, *result_add],
        assumptions=list(skeleton.assumptions),
        negatives=list(skeleton.negatives),
    )


def build_seed_prompt(abstract_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SEED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract a compact paper skeleton from this text.\n\n"
                f"TEXT:\n{abstract_text}"
            ),
        },
    ]


async def extract_skeleton(abstract_text: str, config: Any) -> PaperSkeletonCandidate:
    messages = build_seed_prompt(abstract_text)
    result = await call_model(
        tier=_resolve_model_tier(config),
        messages=messages,
        response_schema=FlatSeedOutput,
        config=config,
    )
    if not isinstance(result, FlatSeedOutput):
        raise TypeError("Expected FlatSeedOutput from structured model call.")

    claims: list[RawClaim] = []
    for claim in result.claims:
        try:
            claims.append(flat_claim_to_raw(claim, fallback_section="abstract"))
        except Exception:
            continue

    return _build_skeleton(claims, config)


async def repair_skeleton(
    skeleton: PaperSkeletonCandidate,
    unmatched_high_strength_candidates: list[RawClaim],
    config: Any,
) -> PaperSkeletonCandidate:
    _ = config
    if not unmatched_high_strength_candidates:
        return skeleton

    return _augment_skeleton(
        skeleton,
        unmatched_high_strength_candidates,
        context_add_cap=1,
        method_add_cap=2,
        result_add_cap=2,
    )


async def extract_seed(abstract_text: str, config: Any) -> SeedOutput:
    skeleton = await extract_skeleton(abstract_text, config)
    return SeedOutput(claims=skeleton.claims())


__all__ = [
    "SEED_SYSTEM_PROMPT",
    "build_seed_prompt",
    "extract_seed",
    "extract_skeleton",
    "repair_skeleton",
]
