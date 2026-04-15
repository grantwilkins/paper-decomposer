from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, cast

from ..models import call_model_with_fallback, flat_claim_to_raw
from ..schema import (
    ClaimLocalRole,
    ClaimStructuralHints,
    ClaimType,
    EvidenceArtifact,
    FlatClaim,
    FlatSectionOutput,
    ModelTier,
    ParentPreference,
    RhetoricalRole,
    RawClaim,
    Section,
    SectionExtractionOutput,
)

call_model = call_model_with_fallback

METHOD_INSTRUCTIONS = """You are reading the METHOD section of a
research paper. The paper's top-level claims are provided below.

Extract every sub-claim that describes a specific design decision,
mechanism, or formalization. A method sub-claim answers:
"What specifically was built, and what property does it have?"

Granularity discipline (strict):
- Keep claims argument-level, not operation-level.
- Emit a standalone METHOD claim only for:
  1) a core contribution, or
  2) a submechanism with its own design role/invariant.
- Do NOT split a single capability into many sibling claims just because
  the text lists lifecycle verbs (allocate/fetch/store/schedule/update).
- Fold helper operations and decode-loop step lists into the nearest
  parent mechanism unless the paper argues them as distinct contributions.
- Merge near-duplicate restatements in the same section.

For each claim:
1. State it as a single proposition (1-2 sentences).
2. Set claim_type to METHOD.
3. Note which seed claim it elaborates (by seed claim_id), or
   "NEW" if it introduces something not in the seed skeleton.
4. In evidence, list any Figures/Tables/Equations mentioned nearby.
5. In entity_names, list any named methods, systems, or components
   mentioned (e.g., "PagedAttention", "KV cache manager").

ALSO extract NEGATIVE claims - anything the authors say DID NOT WORK
or was REJECTED. For negative claims:
- Set claim_type to NEGATIVE
- Fill rejected_what: what was tried or considered
- Fill rejected_why: why it failed or was rejected
- Mark NEGATIVE only when the statement itself asserts failure,
  rejection, limitation, or non-applicability.
- Do NOT mark positive method statements as NEGATIVE just because they
  appear near limitations.

Ignore: notation definitions, background explanations, related work
digressions, forward references to evaluation sections.
"""

EVALUATION_INSTRUCTIONS = """You are reading the EVALUATION section
of a research paper. The paper's top-level claims are provided below.

Extract every specific empirical or formal finding. A result
sub-claim answers: "What was measured, under what conditions, and
what was the outcome?"

For each claim:
1. State the finding with quantitative specifics preserved.
2. Set claim_type to RESULT.
3. In evidence, list the Figure/Table that presents this data.
4. In entity_names, list the baselines, models, and datasets mentioned.

For NEGATIVE results (ablations that show removal doesn't hurt,
experiments that underperformed):
- Set claim_type to NEGATIVE
- Fill rejected_what and rejected_why
- Mark NEGATIVE only when the result statement itself reports a failure,
  limitation, regression, or rejected variant.
- Positive gains reported in limitation-heavy paragraphs are still RESULT.

Ignore: methodological details restated from earlier sections,
qualitative hand-waving without specific findings.
"""

INTRODUCTION_INSTRUCTIONS = """You are reading the INTRODUCTION of a
research paper. The paper's top-level claims are provided below.

Extract claims in three categories:

1. CONTEXT claims: What specific problem or gap does the paper
   identify? Look for signal phrases: "however", "but", "challenge",
   "limitation", "gap", "existing approaches suffer from".

2. CONTEXT claims about existing approaches: What does the paper
   critique about current methods? These are claims about the
   state of the art being insufficient.

3. Any METHOD or RESULT claims NOT already in the seed skeleton.
   Sometimes the intro previews contributions the abstract omitted.

Do NOT extract:
- Generic background ("LLMs are important") - only specific gaps.
- Restatements of seed skeleton claims at the same resolution.
  Only extract if the intro adds NEW specifics.
"""

DISCUSSION_INSTRUCTIONS = """You are reading the DISCUSSION /
LIMITATIONS / CONCLUSION section.

Extract:
1. ASSUMPTION claims: conditions under which results do/don't hold.
   These are often the most important claims in the paper.
2. NEGATIVE claims: limitations the authors acknowledge.
3. Any novel implications or connections drawn from the results.

Signal phrases for assumptions: "we assume", "this requires",
"in practice", "when X holds", "this may not generalize to".

Signal phrases for negative/limitations: "however", "one limitation",
"future work could address", "we do not consider", "this fails when".

NEGATIVE CALIBRATION RULE (strict):
- A claim is NEGATIVE only if the statement itself asserts a failure,
  limitation, rejection, or non-applicability.
- A positive claim (e.g., "achieves 2-4x throughput", "is high-throughput")
  is NOT NEGATIVE even if it appears in the same paragraph as limitations.
- If unsure, prefer RESULT/ASSUMPTION over NEGATIVE unless failure wording
  is explicit in the statement.
"""

BACKGROUND_INSTRUCTIONS = """You are reading BACKGROUND / RELATED WORK.

Extract ONLY contrastive positioning claims - statements where the
authors distinguish their work from prior work:
- "Unlike X, our approach..."
- "While X assumes..., we do not require..."
- "X addresses a different setting where..."

These become CONTEXT-type claims about the paper's positioning.

Group sentence-level contrasts into a small set of higher-level context
claims. Prefer 1-3 grouped positioning claims over many near-duplicate
sentence-level claims.

Do NOT extract: summaries of prior work that don't contrast with
this paper. Related work sections are mostly scaffolding.
"""

UNKNOWN_SECTION_INSTRUCTIONS = """You are reading a section with mixed or
unclear rhetorical role.

Extract only explicit, paper-specific claims supported by this section.
Allowed claim types:
- CONTEXT: concrete problem/gap statements
- METHOD: concrete mechanisms/design decisions
- RESULT: concrete measured findings
- ASSUMPTION: explicit conditions/requirements
- NEGATIVE: rejected alternatives or limitations

Ignore generic background and vague narrative text.
Prefer contribution-level claims over procedural step lists.
Avoid duplicate restatements of the same proposition.
If no valid claims are present, return {"claims": []}.
"""

_VALID_TIERS: set[ModelTier] = {"small", "medium", "heavy"}
_CLAIM_PREFIX: dict[str, str] = {
    "context": "C",
    "method": "M",
    "result": "R",
    "assumption": "A",
    "negative": "N",
}
MAX_SECTION_CHARS_FOR_PROMPT = 8000  # ~2000 tokens

_RESULT_CUE_RE = re.compile(
    r"(throughput|latenc|memory|speedup|improv|outperform|achiev|reduc|increase|gain|percent|%|\bx\b)",
    re.IGNORECASE,
)
_MECHANISM_CUE_RE = re.compile(
    r"(introduce|propose|design|build|implement|partition|allocate|schedule|map|kernel|runtime|manager|table)",
    re.IGNORECASE,
)
_ASSUMPTION_CUE_RE = re.compile(
    r"(assum|require|depends on|holds when|only when|in practice)",
    re.IGNORECASE,
)
_CONTEXT_CUE_RE = re.compile(
    r"(challenge|gap|limitation|bottleneck|existing|prior|however|insufficient|fragmentation)",
    re.IGNORECASE,
)
_POSITIONING_CUE_RE = re.compile(r"(unlike|while .* we|prior work|related work|in contrast)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d")
_HIGH_LEVEL_METHOD_RE = re.compile(
    r"(we (introduce|propose|present)|our (system|approach|framework)|this paper|novel)",
    re.IGNORECASE,
)
_NEGATIVE_CUE_RE = re.compile(
    r"(fail|failed|fails|rejected|reject|limitation|cannot|can't|does not work|too expensive|non-applicable)",
    re.IGNORECASE,
)
_PROCEDURAL_VERB_RE = re.compile(
    r"(select|allocate|fetch|gather|store|write|read|concatenate|append|schedule|update|dispatch|iterate|lookup)",
    re.IGNORECASE,
)
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


def _resolve_model_tier(config: Any) -> ModelTier:
    if config is None:
        return "small"

    pipeline = getattr(config, "pipeline", None)
    if pipeline is not None:
        section_cfg = getattr(pipeline, "section_extraction", None)
        if isinstance(section_cfg, Mapping):
            tier = section_cfg.get("model_tier")
            if isinstance(tier, str) and tier in _VALID_TIERS:
                return cast(ModelTier, tier)

    if isinstance(config, Mapping):
        pipeline_cfg = config.get("pipeline")
        if isinstance(pipeline_cfg, Mapping):
            section_cfg = pipeline_cfg.get("section_extraction")
            if isinstance(section_cfg, Mapping):
                tier = section_cfg.get("model_tier")
                if isinstance(tier, str) and tier in _VALID_TIERS:
                    return cast(ModelTier, tier)

    return "small"


def _section_label(section: Section) -> str:
    if section.section_number:
        return f"{section.section_number} {section.title}"
    return section.title


def _seed_claim_lines(seed_claims: list[RawClaim]) -> str:
    if not seed_claims:
        return "(none)"

    counters: defaultdict[str, int] = defaultdict(int)
    lines: list[str] = []
    for claim in seed_claims:
        claim_key = claim.claim_type.value
        prefix = _CLAIM_PREFIX.get(claim_key, "X")
        counters[prefix] += 1
        statement = claim.statement.strip().replace("\n", " ")
        lines.append(f"[{prefix}{counters[prefix]}] {claim_key.upper()}: {statement}")
    return "\n".join(lines)


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


def _artifact_lines(artifacts: list[EvidenceArtifact]) -> str:
    if not artifacts:
        return "(none)"
    return "\n".join(f"{_artifact_label(a)}: {a.caption.strip()}" for a in artifacts)


def get_instructions_for_role(role: RhetoricalRole) -> str:
    mapping = {
        RhetoricalRole.method: METHOD_INSTRUCTIONS,
        RhetoricalRole.theory: METHOD_INSTRUCTIONS,
        RhetoricalRole.evaluation: EVALUATION_INSTRUCTIONS,
        RhetoricalRole.introduction: INTRODUCTION_INSTRUCTIONS,
        RhetoricalRole.discussion: DISCUSSION_INSTRUCTIONS,
        RhetoricalRole.background: BACKGROUND_INSTRUCTIONS,
        RhetoricalRole.abstract: "",
        RhetoricalRole.appendix: EVALUATION_INSTRUCTIONS,
        RhetoricalRole.other: UNKNOWN_SECTION_INSTRUCTIONS,
    }
    return mapping.get(role, UNKNOWN_SECTION_INSTRUCTIONS)


def _truncate_section_body(body_text: str) -> str:
    if len(body_text) <= MAX_SECTION_CHARS_FOR_PROMPT:
        return body_text

    half = MAX_SECTION_CHARS_FOR_PROMPT // 2
    return (
        body_text[:half]
        + "\n\n[... middle truncated for length ...]\n\n"
        + body_text[-half:]
    )


def build_section_prompt(
    section: Section,
    seed_claims: list[RawClaim],
    artifacts: list[EvidenceArtifact],
) -> list[dict]:
    instructions = get_instructions_for_role(section.role)
    body_text = _truncate_section_body(section.body_text)
    user_content = (
        "SEED SKELETON:\n"
        f"{_seed_claim_lines(seed_claims)}\n\n"
        "EVIDENCE ARTIFACTS IN THIS PAPER:\n"
        f"{_artifact_lines(artifacts)}\n\n"
        f"SECTION ({section.role.value}): {_section_label(section)}\n"
        f"{body_text}\n\n"
        "Output requirements:\n"
        '- Return JSON object with key "claims" only.\n'
        '- Each claim item fields: claim_id, claim_type, statement, source_section, '
        "evidence_ids, entity_names, rejected_what, rejected_why, "
        "elaborates_seed_id, local_role, preferred_parent_type.\n"
        '- evidence_ids must be a list of artifact_id strings.\n'
        '- elaborates_seed_id: seed claim_id it elaborates, or "NEW" when not tied to seed claims.\n'
        "- local_role: one of top_level/mechanism/implementation_detail/empirical_finding/"
        "assumption/limitation/context_gap/positioning/other.\n"
        '- preferred_parent_type: "method", "context", or "none".\n'
        "- claim_type must be determined from the statement content itself, not nearby sentences.\n"
        "- Avoid sibling explosions: operation-level steps should stay inside a parent mechanism claim.\n"
        '- If no valid claims exist, return {"claims": []}.\n\n'
        "Extract all claims from this section following the instructions."
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content},
    ]


def _normalize_flat_claim(
    item: FlatClaim,
    *,
    source_section: str,
    index: int,
    valid_artifact_ids: set[str],
) -> RawClaim:
    normalized = flat_claim_to_raw(item, fallback_section=source_section)
    claim_id = normalized.claim_id.strip() or f"sec_{index}"
    evidence = [
        pointer
        for pointer in normalized.evidence
        if pointer.artifact_id.strip() and pointer.artifact_id.strip() in valid_artifact_ids
    ]

    updates: dict[str, Any] = {}
    if claim_id != normalized.claim_id:
        updates["claim_id"] = claim_id
    if len(evidence) != len(normalized.evidence):
        updates["evidence"] = evidence

    if not updates:
        return normalized
    return normalized.model_copy(update=updates)


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _claim_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_SPLIT_RE.split(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _best_seed_match(claim: RawClaim, seed_claims: list[RawClaim]) -> str | None:
    if not seed_claims:
        return None

    claim_tokens = _claim_tokens(claim.statement)
    if not claim_tokens:
        return None

    best_seed_id: str | None = None
    best_score = 0.0
    for seed_claim in seed_claims:
        seed_tokens = _claim_tokens(seed_claim.statement)
        if not seed_tokens:
            continue
        overlap = len(claim_tokens & seed_tokens)
        if overlap == 0:
            continue
        coverage = overlap / min(len(claim_tokens), len(seed_tokens))
        jaccard = overlap / len(claim_tokens | seed_tokens)
        score = coverage + jaccard
        if seed_claim.claim_type == claim.claim_type:
            score += 0.3
        if score > best_score:
            best_score = score
            best_seed_id = seed_claim.claim_id

    if best_score < 0.6:
        return None
    return best_seed_id


def _retag_claim_type(claim_type: ClaimType, statement: str, role: RhetoricalRole) -> ClaimType:
    lowered = statement.lower()
    has_negative = bool(_NEGATIVE_CUE_RE.search(lowered))
    has_result = bool(_RESULT_CUE_RE.search(lowered)) and (
        bool(_NUMBER_RE.search(lowered))
        or "baseline" in lowered
        or "throughput" in lowered
        or "latency" in lowered
    )
    has_method = bool(_MECHANISM_CUE_RE.search(lowered))
    has_assumption = bool(_ASSUMPTION_CUE_RE.search(lowered))
    has_context = bool(_CONTEXT_CUE_RE.search(lowered))

    if claim_type == ClaimType.negative or has_negative:
        return ClaimType.negative

    if has_assumption and claim_type != ClaimType.assumption and role in {
        RhetoricalRole.discussion,
        RhetoricalRole.introduction,
        RhetoricalRole.method,
        RhetoricalRole.theory,
        RhetoricalRole.evaluation,
    }:
        return ClaimType.assumption

    if role in {RhetoricalRole.evaluation, RhetoricalRole.appendix}:
        if claim_type in {ClaimType.method, ClaimType.context, ClaimType.assumption} and has_result:
            return ClaimType.result

    if role in {RhetoricalRole.method, RhetoricalRole.theory}:
        if claim_type in {ClaimType.context, ClaimType.method, ClaimType.assumption} and has_result and not has_method:
            return ClaimType.result
        if claim_type in {ClaimType.context, ClaimType.result} and has_method and not has_result:
            return ClaimType.method

    if role == RhetoricalRole.background:
        if claim_type in {ClaimType.method, ClaimType.result, ClaimType.assumption} and has_context and not has_method:
            return ClaimType.context

    if claim_type == ClaimType.context and has_result and role == RhetoricalRole.evaluation:
        return ClaimType.result

    if claim_type == ClaimType.context and has_method and not has_context:
        return ClaimType.method

    if claim_type == ClaimType.result and has_method and not has_result:
        return ClaimType.method

    return claim_type


def _infer_local_role(claim_type: ClaimType, statement: str) -> ClaimLocalRole:
    lowered = statement.lower()
    if claim_type == ClaimType.result:
        return ClaimLocalRole.empirical_finding
    if claim_type == ClaimType.assumption:
        return ClaimLocalRole.assumption
    if claim_type == ClaimType.negative:
        return ClaimLocalRole.limitation
    if claim_type == ClaimType.context:
        if _POSITIONING_CUE_RE.search(lowered):
            return ClaimLocalRole.positioning
        return ClaimLocalRole.context_gap
    if claim_type == ClaimType.method:
        if _HIGH_LEVEL_METHOD_RE.search(lowered):
            return ClaimLocalRole.top_level
        if re.search(r"\b(kernel|allocator|lookup|metadata|scheduler|buffer|table|cache manager)\b", lowered):
            return ClaimLocalRole.implementation_detail
        if _MECHANISM_CUE_RE.search(lowered):
            return ClaimLocalRole.mechanism
        return ClaimLocalRole.top_level
    return ClaimLocalRole.other


def _infer_parent_preference(claim_type: ClaimType, local_role: ClaimLocalRole) -> ParentPreference:
    if claim_type == ClaimType.context:
        return ParentPreference.context
    if claim_type == ClaimType.method:
        if local_role == ClaimLocalRole.top_level:
            return ParentPreference.context
        return ParentPreference.method
    if claim_type in {ClaimType.result, ClaimType.assumption, ClaimType.negative}:
        return ParentPreference.method
    return ParentPreference.none


def _postprocess_claim(
    claim: RawClaim,
    *,
    section: Section,
    seed_claims: list[RawClaim],
) -> RawClaim:
    cleaned_statement = _clean_text(claim.statement)
    if not cleaned_statement:
        return claim

    normalized_type = _retag_claim_type(claim.claim_type, cleaned_statement, section.role)
    existing_hints = claim.structural_hints or ClaimStructuralHints()
    seed_ids = {seed.claim_id for seed in seed_claims}

    elaborates_seed_id = existing_hints.elaborates_seed_id
    if elaborates_seed_id not in seed_ids:
        elaborates_seed_id = _best_seed_match(
            claim.model_copy(update={"claim_type": normalized_type, "statement": cleaned_statement}),
            seed_claims,
        )

    local_role = existing_hints.local_role or _infer_local_role(normalized_type, cleaned_statement)
    preferred_parent = existing_hints.preferred_parent_type or _infer_parent_preference(normalized_type, local_role)

    rejected_what = claim.rejected_what
    rejected_why = claim.rejected_why
    if normalized_type != ClaimType.negative:
        rejected_what = None
        rejected_why = None

    return claim.model_copy(
        update={
            "claim_type": normalized_type,
            "statement": cleaned_statement,
            "rejected_what": rejected_what,
            "rejected_why": rejected_why,
            "structural_hints": ClaimStructuralHints(
                elaborates_seed_id=elaborates_seed_id,
                local_role=local_role,
                preferred_parent_type=preferred_parent,
            ),
        }
    )


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _claim_specificity_score(claim: RawClaim) -> float:
    token_count = len(_claim_tokens(claim.statement))
    evidence_count = len({pointer.artifact_id.strip() for pointer in claim.evidence if pointer.artifact_id.strip()})
    entity_count = len({entity.strip().lower() for entity in claim.entity_names if entity.strip()})
    local_role = claim.structural_hints.local_role if claim.structural_hints is not None else None
    role_bonus = 0.0
    if local_role == ClaimLocalRole.top_level:
        role_bonus += 1.5
    if local_role == ClaimLocalRole.mechanism:
        role_bonus += 1.0
    return float(token_count + evidence_count * 3 + entity_count * 2 + role_bonus)


def _is_near_duplicate_claim(left: RawClaim, right: RawClaim) -> bool:
    if left.claim_type != right.claim_type:
        return False

    left_statement = _clean_text(left.statement).lower()
    right_statement = _clean_text(right.statement).lower()
    if not left_statement or not right_statement:
        return False
    if left_statement == right_statement:
        return True

    left_tokens = _claim_tokens(left_statement)
    right_tokens = _claim_tokens(right_statement)
    if not left_tokens or not right_tokens:
        return False

    smaller = min(len(left_tokens), len(right_tokens))
    overlap = len(left_tokens & right_tokens)
    if smaller >= 5 and overlap >= smaller and smaller / max(len(left_tokens), len(right_tokens)) >= 0.8:
        return True

    if smaller >= 6 and _token_jaccard(left_tokens, right_tokens) >= 0.88:
        return True

    return False


def _is_procedural_method_claim(claim: RawClaim) -> bool:
    if claim.claim_type != ClaimType.method:
        return False

    lowered = claim.statement.lower()
    verb_hits = len(_PROCEDURAL_VERB_RE.findall(lowered))
    list_like = lowered.count(",") >= 2 or ";" in lowered or " then " in lowered or "for each" in lowered
    long_statement = len(lowered) >= 180

    if verb_hits >= 5:
        return True
    if verb_hits >= 4 and (list_like or long_statement):
        return True
    if verb_hits >= 3 and list_like and claim.structural_hints and claim.structural_hints.local_role == ClaimLocalRole.implementation_detail:
        return True
    return False


def _has_nonprocedural_method_anchor(claim: RawClaim, candidates: list[RawClaim]) -> bool:
    claim_tokens = _claim_tokens(claim.statement)
    for candidate in candidates:
        if candidate.claim_type != ClaimType.method:
            continue
        if _is_procedural_method_claim(candidate):
            continue
        overlap = len(claim_tokens & _claim_tokens(candidate.statement))
        if overlap >= 4:
            return True
    return False


def _compact_section_claims(claims: list[RawClaim], seed_claims: list[RawClaim]) -> list[RawClaim]:
    compacted: list[RawClaim] = []

    for claim in claims:
        if any(_is_near_duplicate_claim(claim, seed_claim) for seed_claim in seed_claims):
            continue

        if _is_procedural_method_claim(claim) and _has_nonprocedural_method_anchor(claim, compacted):
            continue

        duplicate_index: int | None = None
        for idx, existing in enumerate(compacted):
            if _is_near_duplicate_claim(claim, existing):
                duplicate_index = idx
                break

        if duplicate_index is None:
            compacted.append(claim)
            continue

        existing = compacted[duplicate_index]
        if _claim_specificity_score(claim) > _claim_specificity_score(existing):
            compacted[duplicate_index] = claim

    return compacted


async def extract_section_claims(
    section: Section,
    seed_claims: list[RawClaim],
    artifacts: list[EvidenceArtifact],
    config: Any,
) -> SectionExtractionOutput:
    messages = build_section_prompt(section, seed_claims, artifacts)
    result = await call_model(
        tier=_resolve_model_tier(config),
        messages=messages,
        response_schema=FlatSectionOutput,
        config=config,
    )
    if not isinstance(result, FlatSectionOutput):
        raise TypeError("Expected FlatSectionOutput from structured model call.")

    source_section = _section_label(section)
    valid_artifact_ids = {artifact.artifact_id.strip() for artifact in artifacts if artifact.artifact_id.strip()}
    normalized_claims: list[RawClaim] = []
    for index, item in enumerate(result.claims, start=1):
        try:
            normalized = _normalize_flat_claim(
                item,
                source_section=source_section,
                index=index,
                valid_artifact_ids=valid_artifact_ids,
            )
        except Exception:
            continue
        normalized_claims.append(normalized)

    postprocessed_claims = [
        _postprocess_claim(claim, section=section, seed_claims=seed_claims)
        for claim in normalized_claims
    ]
    return SectionExtractionOutput(claims=_compact_section_claims(postprocessed_claims, seed_claims))


__all__ = [
    "METHOD_INSTRUCTIONS",
    "EVALUATION_INSTRUCTIONS",
    "INTRODUCTION_INSTRUCTIONS",
    "DISCUSSION_INSTRUCTIONS",
    "BACKGROUND_INSTRUCTIONS",
    "UNKNOWN_SECTION_INSTRUCTIONS",
    "get_instructions_for_role",
    "build_section_prompt",
    "extract_section_claims",
]
