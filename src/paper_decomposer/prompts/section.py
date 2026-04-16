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
    PaperSkeletonCandidate,
    ParentPreference,
    RhetoricalRole,
    RawClaim,
    Section,
    SectionArgumentCandidate,
    SectionDigestOutput,
    SectionExtractionOutput,
    SupportDetail,
    SupportDetailType,
    SupportRelationshipType,
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
_ASSUMPTION_CONDITIONAL_RE = re.compile(
    r"\b(if|when|unless|requires?|depends on|under\b|subject to|only when)\b",
    re.IGNORECASE,
)
_FUTURE_WORK_RE = re.compile(
    r"\b(future work|we plan to|could be explored|left for future)\b",
    re.IGNORECASE,
)
_NEGATIVE_TARGET_RE = re.compile(
    r"\b(reject(?:ed)?|limitation|fails?|cannot|can't|too expensive|non-applicable)\b.{0,60}\b(of|for|because|due to|when|under)\b",
    re.IGNORECASE,
)
# Observation/constraint patterns: statements characterising properties, proportions,
# or capabilities without the paper actively doing anything.  These should default to
# context or result, never method.
_OBSERVATION_CONSTRAINT_RE = re.compile(
    r"""
    \b(is|are)\s+(constrained|limited|bounded|dominated)\b
    |
    \bis\s+\w+[-\s]bound\b              # "is memory-bound", "is I/O-bound"
    | \bare\s+\w+[-\s]bound\b           # plural: "are bandwidth-bound"
    | \bgrows?\b.{0,50}\bwith\b         # "grows quickly with N"
    | \bgrow\s+(linearly|quadratically|quickly|rapidly|significantly|exponentially)\b
    | \bcan\s+(be\s+)?shared?\b         # "can share", "can be shared"
    | \baccounts?\s+for\b               # "accounts for X%"
    | \brepresents?\s+\S+\s+of\b       # "represents 30% of"
    | \btends?\s+to\s+(grow|increase|decrease|dominate|bottleneck)\b
    | \b(is|are)\s+(often|typically|generally|usually|mostly|primarily)\s+
      (dominated|limited|bounded|constrained|determined|bottlenecked)\b
    | \b(often|typically|generally|usually)\s+(dominated|limited|bounded|constrained)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Active-agency verbs: the paper / system is constructing or proposing something.
# Presence of this pattern confirms a claim is genuinely method-like even when the
# statement also contains observational language.
_ACTIVE_METHOD_AGENCY_RE = re.compile(
    r"\bwe\s+(introduce|propose|present|design|build|implement|allocate|partition|schedule|develop|create|extend|leverage|employ)\b"
    r"|\bour\s+(system|approach|framework|method|design|scheme|algorithm|technique|work)\b"
    r"|\bthis\s+(paper|work)\s+(introduce|propose|present|describe|design|develop)\b",
    re.IGNORECASE,
)
_IMPLICIT_METHOD_AGENCY_RE = re.compile(
    r"\b(vllm|pagedattention|system|runtime|approach|method|framework|scheduler|allocator|manager)\b.{0,40}\b"
    r"(uses?|maps?|implements?|allocates?|schedules?|manages?|partitions?|leverages?|employs?)\b",
    re.IGNORECASE,
)
_PROCEDURAL_VERB_RE = re.compile(
    r"(select|allocate|fetch|gather|store|write|read|concatenate|append|schedule|update|dispatch|iterate|lookup)",
    re.IGNORECASE,
)
_TRADEOFF_CUE_RE = re.compile(
    r"(trade-?off|at the cost of|without sacrificing|while .* (reducing|improving)|invariant|guarantee|bounded)",
    re.IGNORECASE,
)
_IMPLEMENTATION_SECTION_RE = re.compile(
    r"(implementation|engineering|system details|runtime details|serving stack|deployment)",
    re.IGNORECASE,
)
_IMPLEMENTS_USING_RE = re.compile(
    r"\b(implement(?:s|ed|ing)?|build(?:s|ing|t)?)\b.+\b(using|with|via)\b",
    re.IGNORECASE,
)
_CODEBASE_FACT_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*[km]?\s*(lines?|loc)\s*(of code)?|written in\s+\d+(?:\.\d+)?\s*[km]?\s*lines?)",
    re.IGNORECASE,
)
_FRAMEWORK_USAGE_RE = re.compile(
    r"\b(nccl|fastapi|pytorch|transformers|tensorflow|jax|flask|triton|cuda)\b",
    re.IGNORECASE,
)
_API_SURFACE_RE = re.compile(
    r"\b(api|interface|endpoint|frontend|ui|web app|cli|sdk|openai compatible|allows users to|allows user to)\b",
    re.IGNORECASE,
)
_STACK_INVENTORY_RE = re.compile(
    r"""
    \b(control[-\s]?related|runtime|scheduler|manager|components?)\b.{0,80}\b(developed|implemented|written)\s+in\b
    | \b(written|implemented|developed)\s+in\s+(python|cuda|c\+\+|rust|go|java)\b
    | \b(custom\s+cuda\s+kernels?|python\s+control\s+components?)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_TAUTOLOGICAL_OPERATION_RE = re.compile(
    r"(append(?:s|ed|ing)?\s+(?:a|the)?\s*token|free(?:s|d|ing)?\s+(?:a|the)?\s*sequence|delete(?:s|d|ing)?\s+(?:a|the)?\s*sequence)",
    re.IGNORECASE,
)
_MECHANISM_ANCHOR_RE = re.compile(
    r"(mechanism|module|component|manager|scheduler|allocator|kernel|block table|policy|cache)",
    re.IGNORECASE,
)
_EVAL_FINDING_CUE_RE = re.compile(
    r"(ablation|benchmark|baseline|dataset|experiment|measured|measures|throughput|latency|accuracy|f1|bleu|rouge)",
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
_NON_PROMOTABLE_SUPPORT_TYPES = {
    SupportDetailType.procedural_step,
    SupportDetailType.api_surface,
    SupportDetailType.framework_dependency,
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
    has_agency = bool(_ACTIVE_METHOD_AGENCY_RE.search(lowered))
    is_observation = bool(_OBSERVATION_CONSTRAINT_RE.search(lowered))

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

    # Demote method → context for purely observational/characterisation claims before
    # any section-role promotion runs.  A genuine method claim must describe the paper
    # actively doing something.  Statements like "X is memory-bound", "Y grows with N",
    # "Z can share …", "W accounts for …%" carry no method semantics — they are
    # contextual observations or bottleneck facts regardless of what section they appear in.
    # Guard: only fires when no mechanism-action vocabulary is present AND no
    # paper-agency construction is detected, so legitimate method claims are unaffected.
    if claim_type == ClaimType.method and is_observation and not has_agency:
        return ClaimType.context

    if role in {RhetoricalRole.evaluation, RhetoricalRole.appendix}:
        if claim_type in {ClaimType.method, ClaimType.context, ClaimType.assumption} and has_result:
            return ClaimType.result

    if role in {RhetoricalRole.method, RhetoricalRole.theory}:
        if claim_type in {ClaimType.context, ClaimType.method, ClaimType.assumption} and has_result and not has_method:
            return ClaimType.result
        if claim_type in {ClaimType.context, ClaimType.result} and has_method and has_agency and not has_result:
            return ClaimType.method

    if role == RhetoricalRole.background:
        if claim_type in {ClaimType.method, ClaimType.result, ClaimType.assumption} and has_context and not has_method:
            return ClaimType.context

    if claim_type == ClaimType.context and has_result and role == RhetoricalRole.evaluation:
        return ClaimType.result

    if claim_type == ClaimType.context and has_method and has_agency and not has_context:
        return ClaimType.method

    if claim_type == ClaimType.result and has_method and has_agency and not has_result:
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


def _is_implementation_section(section: Section) -> bool:
    return bool(_IMPLEMENTATION_SECTION_RE.search(section.title))


def _has_evaluation_finding_signal(claim: RawClaim) -> bool:
    lowered = claim.statement.lower()
    if claim.claim_type == ClaimType.result and _RESULT_CUE_RE.search(lowered):
        if _NUMBER_RE.search(lowered) or claim.evidence:
            return True
    if _EVAL_FINDING_CUE_RE.search(lowered) and (_NUMBER_RE.search(lowered) or claim.evidence):
        return True
    return False


def _has_mechanism_signal(claim: RawClaim) -> bool:
    lowered = claim.statement.lower()
    if _HIGH_LEVEL_METHOD_RE.search(lowered) or _MECHANISM_CUE_RE.search(lowered):
        return True
    if _MECHANISM_ANCHOR_RE.search(lowered):
        return True
    if claim.entity_names and re.search(r"\b(uses|maps|routes|partitions|manages|schedules|allocates)\b", lowered):
        return True
    return False


def _has_agency_signal(claim: RawClaim) -> bool:
    statement = claim.statement.lower()
    return bool(_ACTIVE_METHOD_AGENCY_RE.search(statement) or _IMPLICIT_METHOD_AGENCY_RE.search(statement))


def _is_core_context_claim(claim: RawClaim) -> bool:
    lowered = claim.statement.lower()
    if not _CONTEXT_CUE_RE.search(lowered):
        return False
    # Prefer bottleneck/problem statements over generic framing.
    return bool(
        re.search(
            r"\b(bottleneck|challenge|gap|limitation|insufficient|waste|fragmentation|memory-bound|constraint)\b",
            lowered,
        )
    )


def _is_real_assumption_claim(claim: RawClaim) -> bool:
    lowered = claim.statement.lower()
    if not _ASSUMPTION_CUE_RE.search(lowered):
        return False
    if _FUTURE_WORK_RE.search(lowered):
        return False
    return bool(_ASSUMPTION_CONDITIONAL_RE.search(lowered))


def _is_real_negative_claim(claim: RawClaim) -> bool:
    lowered = claim.statement.lower()
    if not _NEGATIVE_CUE_RE.search(lowered):
        return False
    if claim.rejected_what or claim.rejected_why:
        return True
    return bool(_NEGATIVE_TARGET_RE.search(lowered))


def _has_invariant_or_tradeoff_signal(claim: RawClaim) -> bool:
    lowered = claim.statement.lower()
    if _TRADEOFF_CUE_RE.search(lowered):
        return True
    return bool(re.search(r"\b(if|when|unless|requires|subject to|constraint)\b", lowered))


def _has_key_systems_choice_signal(claim: RawClaim) -> bool:
    lowered = claim.statement.lower()
    if not _MECHANISM_ANCHOR_RE.search(lowered):
        return False
    return bool(_has_invariant_or_tradeoff_signal(claim) or _has_evaluation_finding_signal(claim) or _MECHANISM_CUE_RE.search(lowered))


def _has_argumentative_force(claim: RawClaim) -> bool:
    if claim.claim_type in {ClaimType.result, ClaimType.assumption, ClaimType.negative}:
        return True
    if claim.evidence and _NUMBER_RE.search(claim.statement):
        return True
    if _has_evaluation_finding_signal(claim):
        return True
    if _has_invariant_or_tradeoff_signal(claim):
        return True
    return _has_mechanism_signal(claim) and bool(claim.entity_names)


def _matches_hard_suppression_pattern(claim: RawClaim) -> bool:
    lowered = claim.statement.lower()
    if _CODEBASE_FACT_RE.search(lowered):
        return True
    if _TAUTOLOGICAL_OPERATION_RE.search(lowered):
        return True
    if _STACK_INVENTORY_RE.search(lowered):
        return True

    if not _has_argumentative_force(claim):
        if _IMPLEMENTS_USING_RE.search(lowered):
            return True
        if _FRAMEWORK_USAGE_RE.search(lowered):
            return True
        if _API_SURFACE_RE.search(lowered):
            return True

    return False


def _passes_section_role_gate(claim: RawClaim, section: Section) -> bool:
    role = section.role

    if _is_implementation_section(section):
        if claim.claim_type == ClaimType.method:
            return (
                _has_mechanism_signal(claim)
                and _has_agency_signal(claim)
                and (_has_key_systems_choice_signal(claim) or _has_invariant_or_tradeoff_signal(claim) or _has_evaluation_finding_signal(claim))
            )
        return claim.claim_type in {ClaimType.result, ClaimType.assumption, ClaimType.negative}

    if role in {RhetoricalRole.evaluation, RhetoricalRole.appendix}:
        if claim.claim_type == ClaimType.result:
            return _has_evaluation_finding_signal(claim)
        if claim.claim_type == ClaimType.negative:
            return _is_real_negative_claim(claim) and (
                _has_evaluation_finding_signal(claim) or bool(claim.evidence)
            )
        if claim.claim_type == ClaimType.assumption:
            return _is_real_assumption_claim(claim) and bool(claim.evidence)
        return False

    if role in {RhetoricalRole.method, RhetoricalRole.theory}:
        if claim.claim_type == ClaimType.method:
            return (
                _has_agency_signal(claim)
                and (
                    _has_mechanism_signal(claim)
                    or _has_invariant_or_tradeoff_signal(claim)
                    or _has_key_systems_choice_signal(claim)
                    or _has_evaluation_finding_signal(claim)
                )
            )
        if claim.claim_type == ClaimType.result:
            return _has_evaluation_finding_signal(claim)
        if claim.claim_type == ClaimType.assumption:
            return _is_real_assumption_claim(claim)
        if claim.claim_type == ClaimType.negative:
            return _is_real_negative_claim(claim)
        return False

    if role == RhetoricalRole.discussion:
        if claim.claim_type == ClaimType.assumption:
            return _is_real_assumption_claim(claim)
        if claim.claim_type == ClaimType.negative:
            return _is_real_negative_claim(claim)
        if claim.claim_type == ClaimType.context:
            return _is_core_context_claim(claim)
        return claim.claim_type == ClaimType.result

    return True


def _claim_worthiness_score(claim: RawClaim, section: Section) -> float:
    token_count = len(_claim_tokens(claim.statement))
    evidence_count = len({pointer.artifact_id.strip() for pointer in claim.evidence if pointer.artifact_id.strip()})
    entity_count = len({entity.strip().lower() for entity in claim.entity_names if entity.strip()})
    lowered = claim.statement.lower()

    score = float(token_count) / 4.0
    score += float(evidence_count) * 1.0
    score += float(min(entity_count, 2)) * 0.6

    if claim.claim_type in {ClaimType.result, ClaimType.assumption, ClaimType.negative}:
        score += 1.0
    if _has_mechanism_signal(claim):
        score += 1.0
    if _has_invariant_or_tradeoff_signal(claim):
        score += 1.0
    if _has_evaluation_finding_signal(claim):
        score += 1.2
    if (
        claim.structural_hints
        and claim.structural_hints.local_role == ClaimLocalRole.implementation_detail
        and not _has_key_systems_choice_signal(claim)
    ):
        score -= 0.8
    if _FRAMEWORK_USAGE_RE.search(lowered):
        score -= 0.6
    if _API_SURFACE_RE.search(lowered):
        score -= 0.8
    if _CODEBASE_FACT_RE.search(lowered):
        score -= 1.4
    if _TAUTOLOGICAL_OPERATION_RE.search(lowered):
        score -= 1.4
    if _is_implementation_section(section):
        score -= 0.4

    return score


def _with_claim_strength(claim: RawClaim, section: Section) -> RawClaim:
    return claim.model_copy(update={"claim_strength": _claim_worthiness_score(claim, section)})


def _worth_score_threshold(section: Section, stage: str) -> float:
    if _is_implementation_section(section):
        return 1.8 if stage == "pre" else 2.8
    if section.role in {RhetoricalRole.evaluation, RhetoricalRole.appendix}:
        return 2.0 if stage == "pre" else 2.8
    if section.role in {RhetoricalRole.method, RhetoricalRole.theory}:
        return 1.6 if stage == "pre" else 2.4
    return 1.2 if stage == "pre" else 1.8


def _should_keep_claim_for_stage(claim: RawClaim, section: Section, stage: str) -> bool:
    if _matches_hard_suppression_pattern(claim):
        return False
    strength = claim.claim_strength
    if strength is None:
        strength = _claim_worthiness_score(claim, section)
    if strength < _worth_score_threshold(section, stage):
        return False
    if stage == "post" and not _passes_section_role_gate(claim, section):
        return False
    return True


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


def _to_section_argument_candidate(claim: RawClaim) -> SectionArgumentCandidate:
    hints = claim.structural_hints
    return SectionArgumentCandidate(
        claim_id=claim.claim_id,
        claim_type=claim.claim_type,
        statement=claim.statement,
        source_section=claim.source_section,
        evidence_ids=[pointer.artifact_id for pointer in claim.evidence if pointer.artifact_id.strip()],
        entity_names=list(claim.entity_names),
        rejected_what=claim.rejected_what,
        rejected_why=claim.rejected_why,
        elaborates_seed_id=hints.elaborates_seed_id if hints is not None else None,
        local_role=hints.local_role if hints is not None else None,
        preferred_parent_type=hints.preferred_parent_type if hints is not None else None,
        strength=claim.claim_strength,
    )


def _infer_support_detail_type(claim: RawClaim, section: Section) -> SupportDetailType:
    lowered = claim.statement.lower()
    if _API_SURFACE_RE.search(lowered):
        return SupportDetailType.api_surface
    if re.search(r"\b(kernel|warp|cuda|coalesced|latency overhead)\b", lowered):
        return SupportDetailType.local_kernel_optimization
    if _FRAMEWORK_USAGE_RE.search(lowered):
        return SupportDetailType.framework_dependency
    if _is_procedural_method_claim(claim):
        return SupportDetailType.procedural_step
    if _has_evaluation_finding_signal(claim):
        return SupportDetailType.numeric_support
    if _is_implementation_section(section) or claim.claim_type == ClaimType.method:
        return SupportDetailType.implementation_fact
    return SupportDetailType.numeric_support


def _relationship_for_support_type(detail_type: SupportDetailType) -> SupportRelationshipType:
    mapping = {
        SupportDetailType.implementation_fact: SupportRelationshipType.implements,
        SupportDetailType.procedural_step: SupportRelationshipType.operational_context,
        SupportDetailType.api_surface: SupportRelationshipType.instantiates,
        SupportDetailType.framework_dependency: SupportRelationshipType.uses_framework,
        SupportDetailType.local_kernel_optimization: SupportRelationshipType.local_optimization_of,
        SupportDetailType.numeric_support: SupportRelationshipType.measures,
    }
    return mapping[detail_type]


def _support_confidence(claim: RawClaim) -> float:
    strength = claim.claim_strength if claim.claim_strength is not None else 0.0
    normalized = max(0.0, min(1.0, strength / 6.0))
    return round(normalized, 3)


def _claim_to_support_detail(claim: RawClaim, section: Section, index: int) -> SupportDetail:
    detail_type = _infer_support_detail_type(claim, section)
    hints = claim.structural_hints
    anchor_id = hints.elaborates_seed_id if hints is not None else None
    candidate_anchor_ids = [anchor_id] if anchor_id else []
    return SupportDetail(
        support_detail_id=f"SD_{section.section_number or section.title}_{index}".replace(" ", "_"),
        detail_type=detail_type,
        text=claim.statement,
        source_section=claim.source_section,
        anchor_claim_id=anchor_id,
        candidate_anchor_ids=candidate_anchor_ids,
        relationship_type=_relationship_for_support_type(detail_type),
        confidence=_support_confidence(claim),
        evidence_ids=[pointer.artifact_id for pointer in claim.evidence if pointer.artifact_id.strip()],
    )


def _should_route_to_support_detail(claim: RawClaim, section: Section) -> bool:
    if _matches_hard_suppression_pattern(claim):
        return True

    detail_type = _infer_support_detail_type(claim, section)
    if detail_type in {
        SupportDetailType.api_surface,
        SupportDetailType.framework_dependency,
        SupportDetailType.procedural_step,
    }:
        return True

    if claim.claim_type != ClaimType.method:
        return False

    hints = claim.structural_hints
    if hints is not None and hints.local_role == ClaimLocalRole.implementation_detail:
        return True

    lowered = claim.statement.lower()
    if detail_type == SupportDetailType.local_kernel_optimization:
        return True
    return bool(re.search(r"\b(kernel|scheduler|allocator|cache manager|block table|warp|cuda)\b", lowered))


def _is_argument_candidate(claim: RawClaim, section: Section) -> bool:
    if _matches_hard_suppression_pattern(claim):
        return False
    if not _passes_section_role_gate(claim, section):
        return False

    if claim.claim_type == ClaimType.context:
        return _is_core_context_claim(claim)
    if claim.claim_type == ClaimType.result:
        return _has_evaluation_finding_signal(claim) or bool(claim.evidence)
    if claim.claim_type == ClaimType.assumption:
        return _is_real_assumption_claim(claim)
    if claim.claim_type == ClaimType.negative:
        return _is_real_negative_claim(claim)

    # For METHOD claims, require argumentative force beyond local operation details.
    if claim.claim_type == ClaimType.method:
        if not _has_agency_signal(claim):
            return False
        if _has_key_systems_choice_signal(claim):
            return True
        if _has_invariant_or_tradeoff_signal(claim):
            return True
        if _has_evaluation_finding_signal(claim):
            return True
        return bool(claim.structural_hints and claim.structural_hints.local_role == ClaimLocalRole.top_level)

    return False


async def extract_section_digest(
    section: Section,
    skeleton: PaperSkeletonCandidate,
    artifacts: list[EvidenceArtifact],
    config: Any,
) -> SectionDigestOutput:
    extraction = await extract_section_claims(section, skeleton.claims(), artifacts, config)
    argument_candidates: list[SectionArgumentCandidate] = []
    support_details: list[SupportDetail] = []

    for idx, claim in enumerate(extraction.claims, start=1):
        if _should_route_to_support_detail(claim, section):
            support_details.append(_claim_to_support_detail(claim, section, idx))
        else:
            argument_candidates.append(_to_section_argument_candidate(claim))

    return SectionDigestOutput(
        argument_candidates=argument_candidates,
        support_details=support_details,
    )


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
        normalized = _with_claim_strength(normalized, section)
        if not _should_keep_claim_for_stage(normalized, section, stage="pre"):
            continue
        normalized_claims.append(normalized)

    postprocessed_claims: list[RawClaim] = []
    for claim in normalized_claims:
        processed = _with_claim_strength(
            _postprocess_claim(claim, section=section, seed_claims=seed_claims),
            section,
        )
        if not _should_keep_claim_for_stage(processed, section, stage="post"):
            continue
        postprocessed_claims.append(processed)
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
    "extract_section_digest",
    "extract_section_claims",
]
