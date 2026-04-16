from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, cast

from ..schema import ClaimGroup, ClaimType, DedupOutput, ModelTier, RawClaim

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
_METHOD_VERB_RE = re.compile(r"\b(introduc|propos|present|design|build|develop|use|uses|enable|enables)\b", re.IGNORECASE)
_MECHANISM_ASSUMPTION_RE = re.compile(
    r"\b(allocat(?:e|es|ed|ing)|share(?:s|d|ing)?|broadcast(?:s|ed|ing)?|free(?:s|d|ing)?|"
    r"cop(?:y|ies|ied|ying)|schedul(?:e|es|ed|ing)|map(?:s|ped|ping)?|implement(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_PRIMITIVE_HINT_RE = re.compile(
    r"\b(pagedattention|attention algorithm|algorithm|paging|blockwise computation|non-contiguous memory)\b",
    re.IGNORECASE,
)
_SYSTEM_HINT_RE = re.compile(
    r"\b(vllm|serving system|runtime|server|framework|distributed execution|system overview)\b",
    re.IGNORECASE,
)
_SUBMECHANISM_HINT_RE = re.compile(
    r"\b(kernel|scheduler|allocator|cache manager|block table|eviction|preemption|copy-on-write|warp|cuda)\b",
    re.IGNORECASE,
)
_HEADLINE_RESULT_RE = re.compile(
    r"\b(throughput|request rate|request rates|outperform|higher throughput|same latency|speedup|latency)\b",
    re.IGNORECASE,
)
_MEMORY_RESULT_RE = re.compile(
    r"\b(memory|kv cache|fragmentation|batch size|memory waste|memory saving|cache utilization)\b",
    re.IGNORECASE,
)
_DECODING_RESULT_RE = re.compile(r"\b(beam|sampling|prefix|decoding|prompt sharing)\b", re.IGNORECASE)
_CONSTRAINT_RESULT_RE = re.compile(r"\b(overhead|penalty|constraint|cost|latency increase|oom|out of memory)\b", re.IGNORECASE)
_LABEL_BLOCKLIST = {
    "paper",
    "system",
    "method",
    "approach",
    "result",
    "results",
    "baseline",
    "baselines",
    "dataset",
    "sharegpt",
    "alpaca",
}
_ALLOWED_PRIMITIVE_SYSTEM_NAMES = {"pagedattention"}
DEDUP_PROMPT = "Group close paraphrases and keep one canonical claim per deterministic concept family."


@dataclass(frozen=True)
class CompressionResult:
    promoted_claims: list[RawClaim]
    residual_claims: list[RawClaim]
    claim_groups: list[ClaimGroup]
    diagnostics: dict[str, Any]


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


def _dedup_cfg(config: Any) -> Mapping[str, Any]:
    pipeline = getattr(config, "pipeline", None)
    if pipeline is not None:
        dedup_cfg = getattr(pipeline, "dedup", None)
        if isinstance(dedup_cfg, Mapping):
            return dedup_cfg

    if isinstance(config, Mapping):
        pipeline_cfg = config.get("pipeline")
        if isinstance(pipeline_cfg, Mapping):
            dedup_cfg = pipeline_cfg.get("dedup")
            if isinstance(dedup_cfg, Mapping):
                return dedup_cfg
    return {}


def _cap(config: Any, key: str, default: int) -> int:
    raw = _dedup_cfg(config).get(key, default)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _tokens(text: str) -> set[str]:
    lowered = _clean_text(text).lower()
    return {
        token
        for token in _TOKEN_SPLIT_RE.split(lowered)
        if token and len(token) > 2 and token not in _STOPWORDS
    }


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _evidence_ids(claim: RawClaim) -> set[str]:
    return {pointer.artifact_id.strip().lower() for pointer in claim.evidence if pointer.artifact_id.strip()}


def _source_score(source_section: str) -> int:
    lowered = source_section.strip().lower()
    score = 0
    if re.search(r"\b\d", lowered):
        score += 3
    if any(marker in lowered for marker in ("method", "evaluation", "results", "analysis", "discussion")):
        score += 2
    if lowered == "abstract":
        score -= 1
    return score


def _claim_score(claim: RawClaim) -> float:
    tokens = len(_tokens(claim.statement))
    evidence = len(_evidence_ids(claim))
    entities = len({entity.strip().lower() for entity in claim.entity_names if entity.strip()})
    number_bonus = 1.0 if re.search(r"\d", claim.statement) else 0.0
    return float(tokens * 0.2 + evidence * 1.1 + min(entities, 4) * 0.4 + _source_score(claim.source_section) + number_bonus)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _near_duplicate(left: RawClaim, right: RawClaim) -> bool:
    if left.claim_type != right.claim_type:
        return False
    left_tokens = _tokens(left.statement)
    right_tokens = _tokens(right.statement)
    if not left_tokens or not right_tokens:
        return False
    left_numbers = _numbers(left.statement)
    right_numbers = _numbers(right.statement)
    if left_numbers and right_numbers and left_numbers.isdisjoint(right_numbers):
        return False
    overlap = _jaccard(left_tokens, right_tokens)
    containment = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= 0.72 or containment >= 0.82


def _canonical_label_seed(claim: RawClaim) -> str:
    tokens: list[str] = []
    for token in _tokens(claim.statement):
        if token in _LABEL_BLOCKLIST:
            continue
        if token == "vllm" and classify_method_abstraction(claim) == "primitive":
            continue
        tokens.append(token)
    if claim.claim_type == ClaimType.method and "pagedattention" in _tokens(claim.statement):
        tokens = ["pagedattention", *[token for token in tokens if token != "pagedattention"]]
    return "_".join(tokens[:5]) or f"{claim.claim_type.value}_claim"


def classify_method_abstraction(claim: RawClaim) -> str:
    text = _clean_text(claim.statement).lower()
    if _SUBMECHANISM_HINT_RE.search(text):
        return "submechanism"
    primitive_hits = len(_PRIMITIVE_HINT_RE.findall(text))
    system_hits = len(_SYSTEM_HINT_RE.findall(text))
    if primitive_hits and primitive_hits >= system_hits:
        return "primitive"
    if system_hits:
        return "system_realization"
    if _METHOD_VERB_RE.search(text):
        return "system_realization"
    return "submechanism"


def classify_result_family(claim: RawClaim) -> str:
    text = _clean_text(claim.statement).lower()
    if _CONSTRAINT_RESULT_RE.search(text) or ("kernel" in text and "latency" in text):
        return "constraint_observation"
    if _DECODING_RESULT_RE.search(text):
        return "decoding_mode_improvement"
    if _MEMORY_RESULT_RE.search(text):
        return "memory_mechanism_validation"
    if _HEADLINE_RESULT_RE.search(text):
        return "headline_comparative_performance"
    return "headline_comparative_performance"


def _concept_family(claim: RawClaim) -> str:
    text = _clean_text(claim.statement).lower()
    if claim.claim_type == ClaimType.method:
        tier = classify_method_abstraction(claim)
        if "pagedattention" in text and tier == "primitive":
            return "pagedattention"
        if "vllm" in text and tier == "system_realization":
            return "vllm_system"
        seed = _canonical_label_seed(claim)
        return seed or f"method_{tier}"
    if claim.claim_type == ClaimType.result:
        return classify_result_family(claim)
    return _canonical_label_seed(claim)


def _dedupe_preserving_best(claims: list[RawClaim]) -> tuple[list[RawClaim], list[ClaimGroup]]:
    if not claims:
        return [], []

    used: set[str] = set()
    kept: list[RawClaim] = []
    groups: list[ClaimGroup] = []
    for claim in claims:
        if claim.claim_id in used:
            continue
        members = [other for other in claims if other.claim_id not in used and _near_duplicate(claim, other)]
        for member in members:
            used.add(member.claim_id)
        canonical = max(members, key=_claim_score)
        kept.append(canonical)
        groups.append(
            ClaimGroup(
                canonical_id=canonical.claim_id,
                member_ids=[member.claim_id for member in members],
                parent_id=None,
            )
        )
    return kept, groups


def _filter_assumptions(claims: list[RawClaim]) -> tuple[list[RawClaim], list[RawClaim], int]:
    kept: list[RawClaim] = []
    residual: list[RawClaim] = []
    rejected = 0
    for claim in claims:
        if _MECHANISM_ASSUMPTION_RE.search(claim.statement):
            residual.append(claim)
            rejected += 1
            continue
        kept.append(claim)
    return kept, residual, rejected


def _choose_contexts(claims: list[RawClaim], cap: int) -> tuple[list[RawClaim], list[RawClaim]]:
    ranked = sorted(claims, key=_claim_score, reverse=True)
    return ranked[:cap], ranked[cap:]


def _choose_methods(claims: list[RawClaim], cap: int) -> tuple[list[RawClaim], list[RawClaim], int]:
    if not claims or cap <= 0:
        return [], list(claims), 0

    ranked = sorted(claims, key=_claim_score, reverse=True)
    promoted: list[RawClaim] = []
    residual: list[RawClaim] = []
    seen_pairs: set[tuple[str, str]] = set()
    concept_collisions = 0

    for claim in ranked:
        tier = classify_method_abstraction(claim)
        family = _concept_family(claim)
        pair = (family, tier)
        if pair in seen_pairs:
            residual.append(claim)
            concept_collisions += 1
            continue
        if tier == "submechanism":
            residual.append(claim)
            continue
        if tier == "primitive" and "vllm" in _tokens(claim.statement):
            residual.append(claim)
            continue
        if len(promoted) >= cap:
            residual.append(claim)
            continue
        if tier == "system_realization" and any(classify_method_abstraction(existing) == "system_realization" for existing in promoted):
            residual.append(claim)
            concept_collisions += 1
            continue
        if tier == "primitive" and any(classify_method_abstraction(existing) == "primitive" for existing in promoted):
            residual.append(claim)
            concept_collisions += 1
            continue
        seen_pairs.add(pair)
        promoted.append(claim)

    if not promoted and ranked:
        promoted = [max(ranked, key=_claim_score)]
        residual = [claim for claim in ranked if claim.claim_id != promoted[0].claim_id]

    promoted.sort(key=lambda claim: (classify_method_abstraction(claim) != "primitive", -_claim_score(claim)))
    return promoted[:cap], residual + promoted[cap:], concept_collisions


def _choose_results(claims: list[RawClaim], cap: int) -> tuple[list[RawClaim], list[RawClaim], int]:
    ranked = sorted(claims, key=_claim_score, reverse=True)
    by_family: dict[str, RawClaim] = {}
    residual: list[RawClaim] = []
    collisions = 0
    for claim in ranked:
        family = classify_result_family(claim)
        existing = by_family.get(family)
        if existing is None:
            by_family[family] = claim
            continue
        collisions += 1
        residual.append(claim)
    promoted = list(by_family.values())
    promoted.sort(key=_claim_score, reverse=True)
    residual.extend(promoted[cap:])
    return promoted[:cap], residual, collisions


def _choose_capped(claims: list[RawClaim], cap: int) -> tuple[list[RawClaim], list[RawClaim]]:
    ranked = sorted(claims, key=_claim_score, reverse=True)
    return ranked[:cap], ranked[cap:]


def _renumber_claims(claims: list[RawClaim]) -> list[RawClaim]:
    counters: Counter[str] = Counter()
    prefix_by_type = {
        ClaimType.context: "C",
        ClaimType.method: "M",
        ClaimType.result: "R",
        ClaimType.assumption: "A",
        ClaimType.negative: "N",
    }
    renumbered: list[RawClaim] = []
    for claim in claims:
        prefix = prefix_by_type[claim.claim_type]
        counters[prefix] += 1
        renumbered.append(claim.model_copy(update={"claim_id": f"{prefix}{counters[prefix]}"}))
    return renumbered


def compress_claims_to_skeleton(claims: list[RawClaim], config: Any | None = None) -> CompressionResult:
    contexts, methods, results, assumptions, negatives = [], [], [], [], []
    normalized: list[RawClaim] = []
    for claim in claims:
        statement = _clean_text(claim.statement)
        if not statement:
            continue
        normalized_claim = claim.model_copy(
            update={
                "statement": statement,
                "source_section": _clean_text(claim.source_section) or claim.source_section,
                "entity_names": [_clean_text(entity) for entity in claim.entity_names if _clean_text(entity)],
            }
        )
        normalized.append(normalized_claim)

    deduped, groups = _dedupe_preserving_best(normalized)
    for claim in deduped:
        if claim.claim_type == ClaimType.context:
            contexts.append(claim)
        elif claim.claim_type == ClaimType.method:
            methods.append(claim)
        elif claim.claim_type == ClaimType.result:
            results.append(claim)
        elif claim.claim_type == ClaimType.assumption:
            assumptions.append(claim)
        elif claim.claim_type == ClaimType.negative:
            negatives.append(claim)

    kept_assumptions, assumption_residual, assumption_rejected = _filter_assumptions(assumptions)
    kept_contexts, residual_contexts = _choose_contexts(contexts, _cap(config, "context_cap", 1))
    kept_methods, residual_methods, method_collisions = _choose_methods(methods, _cap(config, "method_cap", 2))
    kept_results, residual_results, result_collisions = _choose_results(results, _cap(config, "result_family_cap", 4))
    kept_assumptions, residual_assumptions = _choose_capped(kept_assumptions, _cap(config, "assumption_cap", 2))
    kept_negatives, residual_negatives = _choose_capped(negatives, _cap(config, "negative_cap", 2))

    promoted = [*kept_contexts, *kept_methods, *kept_results, *kept_assumptions, *kept_negatives]
    promoted = _renumber_claims(promoted)

    diagnostics: dict[str, Any] = {
        "promoted_by_type": dict(Counter(claim.claim_type.value for claim in promoted)),
        "concept_family_collisions": method_collisions + result_collisions,
        "assumption_rejections_as_mechanism": assumption_rejected,
        "assumption_candidates": len(assumptions),
    }
    residual = [
        *residual_contexts,
        *residual_methods,
        *residual_results,
        *assumption_residual,
        *residual_assumptions,
        *residual_negatives,
    ]
    return CompressionResult(
        promoted_claims=promoted,
        residual_claims=residual,
        claim_groups=groups,
        diagnostics=diagnostics,
    )


def build_dedup_prompt(claims: list[RawClaim]) -> list[dict[str, str]]:
    if claims:
        lines = [
            f'{idx}. [{claim.claim_id}] {claim.claim_type.value.upper()}: "{_clean_text(claim.statement)}"'
            for idx, claim in enumerate(claims, start=1)
        ]
        body = "\n".join(lines)
    else:
        body = "(no claims provided)"
    return [
        {
            "role": "system",
            "content": (
                "Group close paraphrases that assert the same proposition. Ignore claim_id naming style and keep one canonical claim per group."
            ),
        },
        {"role": "user", "content": body},
    ]


def apply_dedup(claims: list[RawClaim], dedup_output: Any) -> list[RawClaim]:
    groups = list(getattr(dedup_output, "groups", []))
    if not groups:
        return list(claims)

    claim_by_id = {claim.claim_id: claim for claim in claims}
    canonical_by_member: dict[str, str] = {}
    for group in groups:
        if group.canonical_id not in claim_by_id and group.canonical_id not in canonical_by_member:
            members = [member_id for member_id in group.member_ids if member_id in claim_by_id]
            if not members:
                raise ValueError(f"canonical_id {group.canonical_id} is missing from claims")
            best_member = max((claim_by_id[member_id] for member_id in members), key=_claim_score)
            group.canonical_id = best_member.claim_id
        for member_id in group.member_ids:
            canonical_by_member[member_id] = group.canonical_id

    canonicals: list[RawClaim] = []
    seen: set[str] = set()
    for claim in claims:
        canonical_id = canonical_by_member.get(claim.claim_id, claim.claim_id)
        if canonical_id in seen:
            continue
        if canonical_id not in claim_by_id:
            raise ValueError(f"canonical_id {canonical_id} is missing from claims")
        seen.add(canonical_id)
        canonicals.append(claim_by_id[canonical_id])
    return canonicals


async def deduplicate_claims(claims: list[RawClaim], config: Any | None = None) -> DedupOutput:
    _ = _resolve_model_tier(config)
    _, groups = _dedupe_preserving_best(claims)
    return DedupOutput(groups=groups)


async def chunked_dedup(claims: list[RawClaim], config: Any | None = None) -> tuple[list[RawClaim], list[ClaimGroup]]:
    result = compress_claims_to_skeleton(claims, config)
    return result.promoted_claims, result.claim_groups


async def hybrid_dedup_promoted(claims: list[RawClaim], config: Any | None = None) -> tuple[list[RawClaim], list[ClaimGroup]]:
    result = compress_claims_to_skeleton(claims, config)
    return result.promoted_claims, result.claim_groups


__all__ = [
    "CompressionResult",
    "DEDUP_PROMPT",
    "apply_dedup",
    "build_dedup_prompt",
    "chunked_dedup",
    "classify_method_abstraction",
    "classify_result_family",
    "compress_claims_to_skeleton",
    "deduplicate_claims",
    "hybrid_dedup_promoted",
]
