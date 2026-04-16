from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, cast

from ..models import call_model
from ..schema import (
    AbstractionLevel,
    AmbiguityResolutionOutput,
    ClaimLocalRole,
    ClaimGroup,
    ClaimNode,
    ClaimType,
    EvidenceArtifact,
    EvidencePointer,
    FacetedClaim,
    ModelTier,
    OneLiner,
    PaperDecomposition,
    PaperMetadata,
    ParentPreference,
    RawClaim,
    ResultSubtype,
    SemanticRole,
    SupportDetail,
    SupportDetailType,
    SupportRelationshipType,
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
DEFAULT_PARENT_ACCEPT_THRESHOLD = 0.68
DEFAULT_PARENT_MARGIN_THRESHOLD = 0.12


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

_IMPLEMENTATION_PROVENANCE_HINTS = (
    "implementation details",
    "implementation detail",
    "engineering details",
    "appendix",
    "reproducibility",
)

_TOP_LINE_RESULT_TOKENS = {
    "baseline",
    "baselines",
    "end",
    "latency",
    "memory",
    "outperform",
    "outperforms",
    "overall",
    "saving",
    "savings",
    "speedup",
    "throughput",
}

_SUBMECHANISM_RESULT_TOKENS = {
    "ablation",
    "allocator",
    "block",
    "cache",
    "kernel",
    "microbenchmark",
    "overhead",
    "recomputation",
    "scheduler",
    "swap",
    "swapping",
    "table",
}

_TOP_LINE_RESULT_PHRASES = (
    "across baselines",
    "across workloads",
    "end-to-end",
    "higher throughput",
    "lower latency",
    "memory saving",
    "memory savings",
    "overall latency",
    "overall memory",
    "overall throughput",
    "outperforms",
    "same latency",
)

_SUBMECHANISM_RESULT_PHRASES = (
    "ablation",
    "block size",
    "block-table indirection",
    "cache manager",
    "kernel latency",
    "kernel overhead",
    "microbenchmark",
    "scheduler comparison",
    "scheduler overhead",
    "swap vs recomputation",
    "swapping vs recomputation",
)

_MULTIPLIER_RESULT_RE = re.compile(r"\b\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?\s*x\b", re.IGNORECASE)

_SOFTWARE_STACK_TOKENS = {
    "api",
    "cuda",
    "flask",
    "grpc",
    "http",
    "jax",
    "kubernetes",
    "library",
    "numpy",
    "openai",
    "pytorch",
    "python",
    "rest",
    "service",
    "tensorflow",
    "torch",
    "triton",
}

_API_VERB_TOKENS = {
    "accepts",
    "api",
    "argument",
    "arguments",
    "call",
    "calls",
    "endpoint",
    "function",
    "invokes",
    "parameter",
    "parameters",
    "request",
    "requests",
    "response",
    "returns",
}

_PRIMITIVE_METHOD_TOKENS = {
    "algorithm",
    "attention",
    "formal",
    "formalization",
    "mechanism",
    "paging",
    "pagedattention",
    "primitive",
}

_SYSTEM_REALIZATION_TOKENS = {
    "engine",
    "framework",
    "runtime",
    "serving",
    "service",
    "system",
    "vllm",
}

_SUBMECHANISM_METHOD_TOKENS = _LOW_LEVEL_METHOD_TOKENS | {
    "blocktable",
    "cachemanager",
    "eviction",
    "preemption",
}

_TOP_LINE_BASELINE_RE = re.compile(
    r"\b(vs|versus|baseline|baselines|fastertransformer|orca|request rates?|qps)\b",
    re.IGNORECASE,
)

_TOP_LINE_METRIC_RE = re.compile(
    r"\b(end[- ]to[- ]end|throughput|latency|speedup|request rates?|same latency|higher throughput)\b",
    re.IGNORECASE,
)

_METHOD_INSTANTIATION_RE = re.compile(
    r"\b(leverages?|builds? on|built on|uses?|implements?|instantiates?|powered by|based on)\b",
    re.IGNORECASE,
)

_OBSERVATIONAL_METHOD_RE = re.compile(
    r"""
    \b(is|are)\s+(constrained|limited|bounded|dominated)\b
    | \bis\s+\w+[-\s]bound\b
    | \bare\s+\w+[-\s]bound\b
    | \baccounts?\s+for\b
    | \brepresents?\s+\S+\s+of\b
    | \bgrows?\b.{0,40}\bwith\b
    | \b(is|are)\s+(often|typically|generally|usually)\s+(limited|bounded|constrained|dominated)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ACTIVE_METHOD_AGENCY_RE = re.compile(
    r"\bwe\s+(introduce|propose|present|design|build|implement|develop|leverage|employ)\b"
    r"|\bour\s+(system|approach|framework|method|design|algorithm|technique)\b"
    r"|\bthis\s+(paper|work)\s+(introduce|propose|present|describe|design|develop)\b",
    re.IGNORECASE,
)

_ABLATED_RESULT_RE = re.compile(
    r"\b(ablation|remove[sd]?|without\b|disabled?|turn(?:ed)? off|w/o)\b",
    re.IGNORECASE,
)
_WORKLOAD_RESULT_RE = re.compile(
    r"\b(workload|trace|distribution|dataset|sharegpt|alpaca|input length|output length|prompt length|request profile|request mix)\b",
    re.IGNORECASE,
)
_CONSTRAINT_RESULT_RE = re.compile(
    r"\b(overhead|bottleneck|bound|memory[- ]bound|latency overhead|constraint|limited by|exhausted|oom|out of memory|pcie)\b",
    re.IGNORECASE,
)
_HEADLINE_RESULT_RE = re.compile(
    r"\b(throughput|request rates?|latency|speedup|outperform|higher throughput|same latency|end[- ]to[- ]end)\b",
    re.IGNORECASE,
)
_NORMALIZE_STATEMENT_PREFIX_RE = re.compile(
    r"^\s*(we|our system|our approach|this paper|this work)\s+"
    r"(introduce|introduces|propose|proposes|present|presents|show|shows|demonstrate|demonstrates|implement|implements|design|designs)\s+",
    re.IGNORECASE,
)
_NORMALIZE_RHETORIC_RE = re.compile(
    r"\b(in this paper|we show that|we find that|we demonstrate that|it is important to note that)\b",
    re.IGNORECASE,
)
_CANONICAL_LABEL_BLOCKLIST = {
    "vllm",
    "paper",
    "orca",
    "fastertransformer",
    "alpaca",
    "sharegpt",
    "dataset",
    "baseline",
}
_CANONICAL_LABEL_VERB_WEIGHTS = {
    "allocate",
    "allocation",
    "map",
    "mapping",
    "share",
    "sharing",
    "evict",
    "eviction",
    "schedule",
    "scheduler",
    "recompute",
    "swapping",
    "swap",
    "partition",
    "paged",
    "attention",
    "throughput",
    "latency",
    "memory",
}


def _claim_tokens(text: str) -> set[str]:
    lowered = _clean_text(text).lower()
    return {
        token
        for token in _TOKEN_SPLIT_RE.split(lowered)
        if token and len(token) > 2 and token not in _STOPWORDS
    }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _token_overlap_score(left_text: str, right_text: str) -> float:
    left = _claim_tokens(left_text)
    right = _claim_tokens(right_text)
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _section_proximity_score(child: RawClaim, parent: RawClaim) -> float:
    child_source = _clean_text(child.source_section).lower()
    parent_source = _clean_text(parent.source_section).lower()
    if not child_source or not parent_source:
        return 0.0
    if child_source == parent_source:
        return 1.0
    if child_source in parent_source or parent_source in child_source:
        return 0.75
    child_tokens = {token for token in _TOKEN_SPLIT_RE.split(child_source) if token}
    parent_tokens = {token for token in _TOKEN_SPLIT_RE.split(parent_source) if token}
    if not child_tokens or not parent_tokens:
        return 0.0
    overlap = len(child_tokens & parent_tokens)
    return overlap / max(1, min(len(child_tokens), len(parent_tokens)))


def _entity_overlap_score(child: RawClaim, parent: RawClaim) -> float:
    child_entities = {entity.strip().lower() for entity in child.entity_names if entity.strip()}
    parent_entities = {entity.strip().lower() for entity in parent.entity_names if entity.strip()}
    if not child_entities or not parent_entities:
        return 0.0
    return len(child_entities & parent_entities) / max(1, len(child_entities))


def _evidence_overlap_score(child: RawClaim, parent: RawClaim) -> float:
    child_evidence = _evidence_ids(child)
    parent_evidence = _evidence_ids(parent)
    if not child_evidence or not parent_evidence:
        return 0.0
    return len(child_evidence & parent_evidence) / max(1, len(child_evidence))


def _anchor_match_strength(child: RawClaim, parent: RawClaim) -> float:
    hints = child.structural_hints
    if hints is None:
        return 0.0
    if hints.elaborates_seed_id and hints.elaborates_seed_id == parent.claim_id:
        return 1.0
    if hints.preferred_parent_type == ParentPreference.method and parent.claim_type == ClaimType.method:
        return 0.45
    if hints.preferred_parent_type == ParentPreference.context and parent.claim_type == ClaimType.context:
        return 0.45
    return 0.0


def _parent_level_prior(child: RawClaim, parent: RawClaim) -> float:
    if parent.claim_type != ClaimType.method:
        return 0.5
    if child.claim_type == ClaimType.result:
        scope = _classify_result_scope(child)
        if scope == "top_level":
            if _is_system_realization_method(parent):
                return 1.0
            if _is_primitive_method(parent):
                return 0.55
            return 0.25
        if scope == "submechanism":
            return 1.0 if _is_submechanism_method(parent) else 0.3
    if child.claim_type == ClaimType.method:
        child_level = _method_abstraction_level(child)
        parent_level = _method_abstraction_level(parent)
        if child_level == "submechanism":
            return 1.0 if parent_level in {"primitive", "system_realization", "submechanism"} else 0.4
        if child_level == "system_realization":
            return 1.0 if parent_level in {"primitive", "system_realization"} else 0.45
        if child_level == "primitive":
            return 1.0 if parent_level == "primitive" else 0.3
        return 0.6
    return 0.7


def parent_confidence_score(child: RawClaim, parent: RawClaim) -> float:
    # Candidate parent set is already filtered by allowed parent grammar.
    section_score = _section_proximity_score(child, parent)
    entity_score = _entity_overlap_score(child, parent)
    evidence_score = _evidence_overlap_score(child, parent)
    token_score = _token_overlap_score(child.statement, parent.statement)
    anchor_score = _anchor_match_strength(child, parent)
    prior_score = _parent_level_prior(child, parent)

    weighted = (
        0.18 * section_score
        + 0.18 * entity_score
        + 0.18 * evidence_score
        + 0.18 * token_score
        + 0.14 * anchor_score
        + 0.14 * prior_score
    )
    return _clamp(weighted)


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


def _method_local_role(claim: RawClaim) -> ClaimLocalRole | None:
    hints = claim.structural_hints
    if hints is None:
        return None
    return hints.local_role


def _section_rank(source_section: str) -> tuple[int, ...]:
    normalized = _clean_text(source_section).lower()
    match = re.search(r"\b(\d+(?:\.\d+)*)\b", normalized)
    if not match:
        if "abstract" in normalized:
            return (0,)
        if "introduction" in normalized:
            return (1,)
        if "conclusion" in normalized or "discussion" in normalized:
            return (99,)
        return (50,)
    return tuple(int(part) for part in match.group(1).split("."))


def _method_abstraction_level(claim: RawClaim) -> str:
    """Return one of: primitive | system_realization | submechanism."""
    local_role = _method_local_role(claim)
    if local_role == ClaimLocalRole.implementation_detail:
        return "submechanism"

    statement = _clean_text(claim.statement).lower()
    tokens = _claim_tokens(statement)
    entities = {entity.strip().lower() for entity in claim.entity_names if entity.strip()}

    primitive_hits = len(tokens & _PRIMITIVE_METHOD_TOKENS)
    system_hits = len(tokens & _SYSTEM_REALIZATION_TOKENS)
    sub_hits = len(tokens & _SUBMECHANISM_METHOD_TOKENS)

    if any("pagedattention" in entity or "algorithm" in entity for entity in entities):
        primitive_hits += 2
    if any("vllm" in entity or "system" in entity or "runtime" in entity for entity in entities):
        system_hits += 2
    if any(
        key in entity
        for entity in entities
        for key in ("scheduler", "allocator", "kernel", "cache manager", "block table")
    ):
        sub_hits += 2

    # Prompt/hint default first.
    if local_role == ClaimLocalRole.top_level:
        default_level = "system_realization"
    elif local_role == ClaimLocalRole.mechanism:
        default_level = "submechanism"
    else:
        default_level = "system_realization"

    # Deterministic override second.
    if sub_hits >= 3 and system_hits <= primitive_hits + 1:
        return "submechanism"
    if system_hits >= primitive_hits + 1 and system_hits >= 2:
        return "system_realization"
    if primitive_hits >= max(2, system_hits):
        return "primitive"
    if _is_low_level_method(claim):
        return "submechanism"
    return default_level


def _is_system_realization_method(claim: RawClaim) -> bool:
    return _method_abstraction_level(claim) == "system_realization"


def _is_primitive_method(claim: RawClaim) -> bool:
    return _method_abstraction_level(claim) == "primitive"


def _has_explicit_instantiation_evidence(system_claim: RawClaim, primitive_claim: RawClaim) -> bool:
    if system_claim.claim_id == primitive_claim.claim_id:
        return False

    system_text = _clean_text(system_claim.statement).lower()
    primitive_entities = {
        entity.strip().lower()
        for entity in primitive_claim.entity_names
        if entity.strip() and len(entity.strip()) > 2
    }
    primitive_tokens = _claim_tokens(primitive_claim.statement)
    shared_entities = {
        entity.strip().lower()
        for entity in system_claim.entity_names
        if entity.strip()
    } & primitive_entities

    mentions_primitive_name = any(entity in system_text for entity in primitive_entities)
    token_overlap = bool(_claim_tokens(system_claim.statement) & primitive_tokens)
    verb_signal = bool(_METHOD_INSTANTIATION_RE.search(system_text))
    seed_link = (
        (system_claim.structural_hints is not None and system_claim.structural_hints.elaborates_seed_id == primitive_claim.claim_id)
        or (
            primitive_claim.structural_hints is not None
            and primitive_claim.structural_hints.elaborates_seed_id == system_claim.claim_id
        )
    )
    section_progression = _section_rank(system_claim.source_section) >= _section_rank(primitive_claim.source_section)

    if seed_link:
        return True
    if verb_signal and (mentions_primitive_name or shared_entities):
        return True
    if mentions_primitive_name and token_overlap and section_progression:
        return True
    return False


def _is_observational_method_statement(claim: RawClaim) -> bool:
    statement = _clean_text(claim.statement).lower()
    if not _OBSERVATIONAL_METHOD_RE.search(statement):
        return False
    if _ACTIVE_METHOD_AGENCY_RE.search(statement):
        return False
    return True


def _context_problem_signal_count(claim: RawClaim) -> int:
    text = _clean_text(claim.statement).lower()
    cues = (
        "bottleneck",
        "challenge",
        "gap",
        "insufficient",
        "limitation",
        "problem",
        "waste",
    )
    return sum(1 for cue in cues if cue in text)


def _is_derivative_context_claim(claim: RawClaim, core_contexts: list[RawClaim]) -> bool:
    if not core_contexts:
        return False
    if _context_problem_signal_count(claim) > 0:
        return False

    overlaps = [
        _token_overlap_score(claim.statement, core_context.statement)
        for core_context in core_contexts
    ]
    best_overlap = max(overlaps, default=0.0)
    claim_tokens = _claim_tokens(claim.statement)
    best_overlap_tokens = max(
        (len(claim_tokens & _claim_tokens(core_context.statement)) for core_context in core_contexts),
        default=0,
    )
    statement = _clean_text(claim.statement).lower()
    quantifies_existing = bool(
        re.search(
            r"\b(\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?\s*(x|%|percent|gb|mb|tokens?|requests?)|for example|e\.g\.)\b",
            statement,
        )
    )
    return (best_overlap >= 0.18 or best_overlap_tokens >= 2) and quantifies_existing


def _depends_on_descendant(claim_id: str, dependency_id: str, parent_by_claim: dict[str, str]) -> bool:
    current = dependency_id
    visited: set[str] = set()
    while current and current not in visited:
        if current == claim_id:
            return True
        visited.add(current)
        current = parent_by_claim.get(current, "")
    return False


def _is_top_level_method(claim: RawClaim) -> bool:
    return _method_abstraction_level(claim) in {"primitive", "system_realization"}


def _is_submechanism_method(claim: RawClaim) -> bool:
    return _method_abstraction_level(claim) == "submechanism"


def _classify_result_scope(claim: RawClaim) -> str:
    if claim.claim_type != ClaimType.result:
        return "neutral"

    statement = _clean_text(claim.statement).lower()
    tokens = _claim_tokens(statement)
    top_level_score = len(tokens & _TOP_LINE_RESULT_TOKENS)
    submechanism_score = len(tokens & _SUBMECHANISM_RESULT_TOKENS)

    top_level_score += sum(2 for phrase in _TOP_LINE_RESULT_PHRASES if phrase in statement)
    submechanism_score += sum(2 for phrase in _SUBMECHANISM_RESULT_PHRASES if phrase in statement)

    if _MULTIPLIER_RESULT_RE.search(statement):
        top_level_score += 2
    has_baseline_comparison = bool(_TOP_LINE_BASELINE_RE.search(statement))
    if " vs " in f" {statement} " or " versus " in f" {statement} ":
        top_level_score += 1
    if has_baseline_comparison:
        top_level_score += 2
    if _TOP_LINE_METRIC_RE.search(statement) and (
        has_baseline_comparison
        or "end-to-end" in statement
        or "overall" in statement
        or "request rate" in statement
    ):
        top_level_score += 2

    if top_level_score >= submechanism_score + 2 and top_level_score > 0:
        return "top_level"
    if submechanism_score >= top_level_score + 1 and submechanism_score > 0:
        return "submechanism"
    return "neutral"


def _classify_result_subtype(claim: RawClaim) -> ResultSubtype | None:
    if claim.claim_type != ClaimType.result:
        return None

    statement = _clean_text(claim.statement).lower()
    if _ABLATED_RESULT_RE.search(statement):
        return ResultSubtype.ablation
    if _WORKLOAD_RESULT_RE.search(statement):
        # Workload/content characterization takes precedence over generic constraints.
        return ResultSubtype.workload_characterization
    if _CONSTRAINT_RESULT_RE.search(statement):
        return ResultSubtype.constraint_observation
    if _HEADLINE_RESULT_RE.search(statement) and _TOP_LINE_BASELINE_RE.search(statement):
        return ResultSubtype.headline_result
    return ResultSubtype.mechanism_validation


def _normalize_statement_conservative(statement: str) -> str:
    normalized = _clean_text(statement)
    normalized = _NORMALIZE_STATEMENT_PREFIX_RE.sub("", normalized)
    normalized = _NORMALIZE_RHETORIC_RE.sub("", normalized)
    normalized = _clean_text(normalized.strip(" ,.;:"))

    replacements = {
        "kv cache": "KV cache",
        "llm": "LLM",
        "gpu": "GPU",
        "cpu": "CPU",
    }
    lowered = normalized.lower()
    for source, target in replacements.items():
        lowered = re.sub(rf"\b{re.escape(source)}\b", target, lowered)
    normalized = _clean_text(lowered)
    if not normalized:
        normalized = _clean_text(statement)
    if normalized and normalized[-1] not in ".!?":
        normalized = f"{normalized}."
    return normalized


def _claim_rank_score(claim: RawClaim) -> float:
    evidence_score = len(_evidence_ids(claim))
    entity_score = len({entity.strip().lower() for entity in claim.entity_names if entity.strip()})
    token_score = len(_claim_tokens(claim.statement))
    strength = claim.claim_strength or 0.0
    return strength + evidence_score * 2.0 + entity_score * 1.5 + min(token_score, 16) / 8.0


def _is_restatement_pair(parent: RawClaim, child: RawClaim) -> bool:
    if parent.claim_type != ClaimType.method or child.claim_type != ClaimType.method:
        return False
    token_overlap = _token_overlap_score(parent.statement, child.statement)
    if token_overlap >= 0.72:
        return True
    parent_text = _clean_text(parent.statement).lower()
    child_text = _clean_text(child.statement).lower()
    if parent_text in child_text or child_text in parent_text:
        return True
    return False


def _abstraction_level_for_claim(claim: RawClaim) -> AbstractionLevel:
    if claim.claim_type == ClaimType.context:
        return AbstractionLevel.problem
    if claim.claim_type == ClaimType.method:
        level = _method_abstraction_level(claim)
        if level == "primitive":
            return AbstractionLevel.primitive
        if level == "system_realization":
            return AbstractionLevel.system_realization
        return AbstractionLevel.submechanism
    return AbstractionLevel.not_applicable


def _semantic_role_for_claim(
    claim: RawClaim,
    abstraction_level: AbstractionLevel,
    result_subtype: ResultSubtype | None,
) -> SemanticRole:
    if claim.claim_type == ClaimType.context:
        return SemanticRole.problem
    if claim.claim_type == ClaimType.method:
        if abstraction_level in {AbstractionLevel.primitive, AbstractionLevel.system_realization}:
            return SemanticRole.method_core
        return SemanticRole.method_support
    if claim.claim_type == ClaimType.result:
        if result_subtype == ResultSubtype.headline_result:
            return SemanticRole.headline_result
        return SemanticRole.scoped_result
    if claim.claim_type == ClaimType.assumption:
        return SemanticRole.assumption
    return SemanticRole.limitation


def _canonical_label_parts(claim: RawClaim, normalized_statement: str) -> list[str]:
    tokens = [
        token
        for token in _TOKEN_SPLIT_RE.split(normalized_statement.lower())
        if token
        and len(token) > 2
        and token not in _STOPWORDS
        and token not in _CANONICAL_LABEL_BLOCKLIST
        and not token.isdigit()
    ]

    entity_tokens: list[str] = []
    for entity in claim.entity_names:
        for token in _TOKEN_SPLIT_RE.split(entity.lower()):
            if (
                token
                and len(token) > 2
                and token not in _STOPWORDS
                and token not in _CANONICAL_LABEL_BLOCKLIST
                and not token.isdigit()
            ):
                entity_tokens.append(token)

    weighted = [token for token in tokens if token in _CANONICAL_LABEL_VERB_WEIGHTS]
    parts: list[str] = []
    for token in [*weighted, *entity_tokens, *tokens]:
        if token in parts:
            continue
        parts.append(token)
        if len(parts) >= 6:
            break
    if not parts:
        parts = [claim.claim_type.value, "claim"]
    return parts


def _best_method_parent(
    claim: RawClaim,
    method_claims: list[RawClaim],
) -> tuple[str | None, int]:
    candidate_methods = [method for method in method_claims if method.claim_id != claim.claim_id]
    if not candidate_methods:
        return None, -1

    result_scope = _classify_result_scope(claim)
    if result_scope == "top_level":
        system_methods = [method for method in candidate_methods if _is_system_realization_method(method)]
        if system_methods:
            candidate_methods = system_methods
        else:
            top_level_methods = [method for method in candidate_methods if _is_top_level_method(method)]
            if top_level_methods:
                candidate_methods = top_level_methods
    elif result_scope == "submechanism":
        submechanism_methods = [method for method in candidate_methods if _is_submechanism_method(method)]
        if submechanism_methods:
            candidate_methods = submechanism_methods

    best_id: str | None = None
    best_score = -1
    for method in candidate_methods:
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
    local_role = _method_local_role(claim)
    if local_role is not None:
        if local_role in {ClaimLocalRole.mechanism, ClaimLocalRole.implementation_detail}:
            return True
        if local_role == ClaimLocalRole.top_level:
            return False
    statement = _clean_text(claim.statement).lower()
    if any(hint in statement for hint in _HIGH_LEVEL_METHOD_HINTS):
        return False
    return _method_specificity_score(claim) >= 2


def _method_entity_set(claim: RawClaim) -> set[str]:
    return {entity.strip().lower() for entity in claim.entity_names if entity.strip()}


def _has_strong_core_method_overlap(claim: RawClaim, core_method_claims: list[RawClaim]) -> bool:
    claim_entities = _method_entity_set(claim)
    if claim_entities:
        for core_method in core_method_claims:
            if core_method.claim_id == claim.claim_id:
                continue
            if claim_entities & _method_entity_set(core_method):
                return True

    best_method_id, affinity = _best_method_parent(claim, core_method_claims)
    return best_method_id is not None and affinity >= 8


def _looks_like_software_stack_description(claim: RawClaim) -> bool:
    tokens = _claim_tokens(claim.statement)
    return len(tokens & _SOFTWARE_STACK_TOKENS) >= 2 and len(tokens) <= 10


def _looks_like_local_api_definition(claim: RawClaim) -> bool:
    tokens = _claim_tokens(claim.statement)
    return len(tokens & _API_VERB_TOKENS) >= 2 and len(tokens) <= 12


def _is_too_short_or_tautological_method(claim: RawClaim) -> bool:
    statement = _clean_text(claim.statement)
    if len(statement) < 35:
        return True
    tokens = _claim_tokens(statement)
    if len(tokens) < 5:
        return True

    lowered = statement.lower()
    tautological_prefixes = ("we use ", "we implement ", "this method is ", "our method is ")
    if lowered.startswith(tautological_prefixes):
        return True
    if "is a method" in lowered or "our approach works" in lowered:
        return True
    return False


def _is_claim_worthy_node(
    claim: RawClaim,
    faceted_by_id: dict[str, FacetedClaim],
    core_method_claims: list[RawClaim],
    source_section: str,
) -> bool:
    if claim.claim_type != ClaimType.method:
        return True

    hints = claim.structural_hints
    if hints is not None and hints.local_role == ClaimLocalRole.top_level:
        return True
    if claim.claim_strength is not None:
        threshold = 2.4
        if hints is not None and hints.local_role is not None:
            if hints.local_role == ClaimLocalRole.mechanism:
                threshold -= 0.2
            elif hints.local_role == ClaimLocalRole.implementation_detail:
                threshold += 0.6
        if any(hint in source_section.strip().lower() for hint in _IMPLEMENTATION_PROVENANCE_HINTS):
            threshold += 0.4
        return claim.claim_strength >= threshold
    if (
        hints is not None
        and hints.elaborates_seed_id
        and hints.preferred_parent_type == ParentPreference.method
    ):
        return True

    has_evidence = bool(_evidence_ids(claim))
    has_facets = claim.claim_id in faceted_by_id
    has_strong_overlap = _has_strong_core_method_overlap(claim, core_method_claims)

    source_lower = source_section.strip().lower()
    implementation_provenance = any(hint in source_lower for hint in _IMPLEMENTATION_PROVENANCE_HINTS)
    stack_description = _looks_like_software_stack_description(claim)
    local_api_definition = _looks_like_local_api_definition(claim)
    too_short_or_tautological = _is_too_short_or_tautological_method(claim)

    if too_short_or_tautological:
        return False
    if implementation_provenance and not has_evidence and not has_strong_overlap:
        return False
    if stack_description and not has_evidence and not has_strong_overlap:
        return False
    if local_api_definition and not has_evidence and not has_strong_overlap:
        return False
    if not has_evidence and not has_strong_overlap and _is_low_level_method(claim):
        return False

    return has_evidence or has_facets or has_strong_overlap or not _is_low_level_method(claim)


def _nearest_higher_level_method_id(
    claim: RawClaim,
    method_claims: list[RawClaim],
) -> str | None:
    if not method_claims:
        return None

    higher_level_candidates = [method for method in method_claims if method.claim_id != claim.claim_id]
    higher_level_candidates = [method for method in higher_level_candidates if not _is_low_level_method(method)]
    if not higher_level_candidates:
        higher_level_candidates = [method for method in method_claims if method.claim_id != claim.claim_id]
    if not higher_level_candidates:
        return None

    best_parent_id, score = _best_method_parent(claim, higher_level_candidates)
    if best_parent_id is None:
        return None
    if score < 0:
        return higher_level_candidates[0].claim_id
    return best_parent_id


def _merge_evidence_pointers(
    primary: list[EvidencePointer],
    supplemental: list[EvidencePointer],
) -> list[EvidencePointer]:
    merged: list[EvidencePointer] = list(primary)
    seen = {
        (pointer.artifact_id.strip().lower(), pointer.role.strip().lower())
        for pointer in primary
        if pointer.artifact_id.strip()
    }
    for pointer in supplemental:
        artifact_id = pointer.artifact_id.strip()
        if not artifact_id:
            continue
        key = (artifact_id.lower(), pointer.role.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(pointer)
    return merged


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


def _compaction_support_detail(
    demoted_claim: RawClaim,
    *,
    anchor_claim_id: str,
    index: int,
) -> SupportDetail:
    strength = demoted_claim.claim_strength or 0.0
    confidence = max(0.0, min(1.0, round(0.45 + (strength / 10.0), 3)))
    return SupportDetail(
        support_detail_id=f"SD_compact_{demoted_claim.claim_id}_{index}",
        detail_type=SupportDetailType.implementation_fact,
        text=demoted_claim.statement,
        source_section=demoted_claim.source_section,
        anchor_claim_id=anchor_claim_id,
        candidate_anchor_ids=[anchor_claim_id],
        relationship_type=SupportRelationshipType.implements,
        confidence=confidence,
        evidence_ids=[pointer.artifact_id for pointer in demoted_claim.evidence if pointer.artifact_id.strip()],
        promotable=True,
    )


def _build_tree_nodes(
    claims: list[RawClaim],
    faceted: list[FacetedClaim],
    assignments: list[TreeNodeAssignment],
    parent_hints: dict[str, str] | None = None,
) -> tuple[list[ClaimNode], list[SupportDetail], dict[str, float | int]]:
    claim_by_id: dict[str, RawClaim] = {}
    claim_order: list[str] = []
    for claim in claims:
        if claim.claim_id in claim_by_id:
            continue
        claim_by_id[claim.claim_id] = claim
        claim_order.append(claim.claim_id)

    facets_by_id = {faceted_claim.claim.claim_id: faceted_claim for faceted_claim in faceted}
    all_method_claims = [claim_by_id[claim_id] for claim_id in claim_order if claim_by_id[claim_id].claim_type == ClaimType.method]
    core_method_claims = [claim for claim in all_method_claims if not _is_low_level_method(claim)] or list(all_method_claims)
    generated_support_details: list[SupportDetail] = []
    initial_node_count = len(claim_order)

    suppressed_method_ids: set[str] = set()
    for method_claim in all_method_claims:
        if _is_claim_worthy_node(
            method_claim,
            faceted_by_id=facets_by_id,
            core_method_claims=core_method_claims,
            source_section=method_claim.source_section,
        ):
            continue
        suppressed_method_ids.add(method_claim.claim_id)

    # Always keep at least one method node if methods are present.
    if len(suppressed_method_ids) == len(all_method_claims) and all_method_claims:
        strongest_method = max(
            all_method_claims,
            key=lambda claim: (len(_evidence_ids(claim)), _method_specificity_score(claim)),
        )
        suppressed_method_ids.discard(strongest_method.claim_id)

    kept_method_claims = [
        method_claim for method_claim in all_method_claims if method_claim.claim_id not in suppressed_method_ids
    ]
    folded_support_evidence: dict[str, list[EvidencePointer]] = {}
    for suppressed_id in suppressed_method_ids:
        suppressed_claim = claim_by_id[suppressed_id]
        fold_parent_id = _nearest_higher_level_method_id(suppressed_claim, kept_method_claims)
        if fold_parent_id is None:
            continue
        folded_support_evidence.setdefault(fold_parent_id, []).extend(suppressed_claim.evidence)

    claim_order = [
        claim_id
        for claim_id in claim_order
        if claim_id not in suppressed_method_ids
    ]
    claim_by_id = {claim_id: claim_by_id[claim_id] for claim_id in claim_order}

    mistyped_method_ids = {
        claim_id
        for claim_id in claim_order
        if claim_by_id[claim_id].claim_type == ClaimType.method
        and _is_observational_method_statement(claim_by_id[claim_id])
    }
    safe_demote_method_ids = {
        claim_id
        for claim_id in mistyped_method_ids
        if claim_id not in facets_by_id
        and (
            claim_by_id[claim_id].structural_hints is None
            or claim_by_id[claim_id].structural_hints.local_role != ClaimLocalRole.top_level
        )
    }
    for claim_id in safe_demote_method_ids:
        claim_by_id[claim_id] = claim_by_id[claim_id].model_copy(update={"claim_type": ClaimType.context})
    non_parentable_method_ids = mistyped_method_ids - safe_demote_method_ids

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
    parentable_method_claims = [
        method_claim for method_claim in method_claims if method_claim.claim_id not in non_parentable_method_ids
    ]
    parentable_method_ids = [method_claim.claim_id for method_claim in parentable_method_claims]
    result_ids = [claim_id for claim_id in claim_order if claim_by_id[claim_id].claim_type == ClaimType.result]

    # Enforce structural grammar on provided parent assignments.
    for child_id, parent_id in list(parent_by_claim.items()):
        child_claim = claim_by_id[child_id]
        parent_claim = claim_by_id.get(parent_id)
        if parent_claim is None:
            del parent_by_claim[child_id]
            continue

        if parent_id in non_parentable_method_ids:
            del parent_by_claim[child_id]
            if depends_by_claim.get(child_id) == [parent_id]:
                depends_by_claim[child_id] = []
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
            method_claims=parentable_method_claims,
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
            if parentable_method_ids:
                best_method_id, score = _best_method_parent(claim, parentable_method_claims)
                fallback_parent = best_method_id if best_method_id is not None and score > 0 else parentable_method_ids[0]
            elif context_ids:
                fallback_parent = context_ids[0]
        elif claim.claim_type == ClaimType.assumption:
            if parentable_method_ids:
                best_method_id, score = _best_method_parent(claim, parentable_method_claims)
                fallback_parent = best_method_id if best_method_id is not None and score > 0 else parentable_method_ids[0]
            elif result_ids:
                fallback_parent = result_ids[0]
            elif context_ids:
                fallback_parent = context_ids[0]
        elif claim.claim_type == ClaimType.negative:
            if parentable_method_ids:
                best_method_id, score = _best_method_parent(claim, parentable_method_claims)
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

    # Reattach clearly derivative context roots under core context claims.
    root_context_ids = [claim_id for claim_id in context_ids if claim_id not in parent_by_claim]
    core_context_ids = [
        claim_id
        for claim_id in root_context_ids
        if _context_problem_signal_count(claim_by_id[claim_id]) > 0
    ]
    if not core_context_ids and root_context_ids:
        core_context_ids = [root_context_ids[0]]
    core_context_claims = [claim_by_id[claim_id] for claim_id in core_context_ids]
    for claim_id in root_context_ids:
        if claim_id in core_context_ids:
            continue
        claim = claim_by_id[claim_id]
        if not _is_derivative_context_claim(claim, core_context_claims):
            continue
        best_core_id, score = _best_context_parent(claim, core_context_claims)
        if best_core_id is None:
            continue
        if score <= 0:
            continue
        if _would_create_parent_cycle(claim_id, best_core_id, parent_by_claim):
            continue
        parent_by_claim[claim_id] = best_core_id
        depends_by_claim[claim_id] = [best_core_id]

    child_count: dict[str, int] = {}
    for parent_id in parent_by_claim.values():
        child_count[parent_id] = child_count.get(parent_id, 0) + 1

    # Mistyped observational method nodes should not parent other claims.
    for claim_id in non_parentable_method_ids:
        if claim_id not in claim_by_id:
            continue
        current_parent_id = parent_by_claim.get(claim_id)
        current_parent_claim = claim_by_id.get(current_parent_id) if current_parent_id else None
        if current_parent_claim is not None and current_parent_claim.claim_type == ClaimType.context:
            continue
        if not context_claims:
            continue
        best_context_id, context_score = _best_context_parent(claim_by_id[claim_id], context_claims)
        fallback_context = best_context_id if best_context_id is not None and context_score > 0 else context_ids[0]
        if _would_create_parent_cycle(claim_id, fallback_context, parent_by_claim):
            continue
        if current_parent_id:
            child_count[current_parent_id] = max(0, child_count.get(current_parent_id, 1) - 1)
        parent_by_claim[claim_id] = fallback_context
        child_count[fallback_context] = child_count.get(fallback_context, 0) + 1
        depends_by_claim[claim_id] = [fallback_context]

    # Low-level implementation claims should prefer method parents over broad context roots.
    for claim_id in method_ids:
        if claim_id in non_parentable_method_ids:
            continue
        if len(parentable_method_claims) <= 1:
            break

        claim = claim_by_id[claim_id]
        current_parent = parent_by_claim.get(claim_id)
        best_method_id, best_score = _best_method_parent(claim, parentable_method_claims)
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

    # Enforce method abstraction ordering constraints.
    for claim_id in method_ids:
        if claim_id in non_parentable_method_ids:
            continue
        if claim_id not in claim_by_id:
            continue

        claim = claim_by_id[claim_id]
        child_level = _method_abstraction_level(claim)
        current_parent_id = parent_by_claim.get(claim_id)
        current_parent_claim = claim_by_id.get(current_parent_id) if current_parent_id else None

        # Submechanisms should attach beneath method parents whenever possible.
        if (
            child_level == "submechanism"
            and (
                current_parent_claim is None
                or current_parent_claim.claim_type != ClaimType.method
            )
            and len(parentable_method_claims) > 1
        ):
            preferred_parents = [
                candidate
                for candidate in parentable_method_claims
                if candidate.claim_id != claim_id and _method_abstraction_level(candidate) in {"primitive", "system_realization"}
            ]
            if not preferred_parents:
                preferred_parents = [
                    candidate
                    for candidate in parentable_method_claims
                    if candidate.claim_id != claim_id
                ]
            if preferred_parents:
                best_parent_id, _ = _best_method_parent(claim, preferred_parents)
                if best_parent_id is not None and not _would_create_parent_cycle(claim_id, best_parent_id, parent_by_claim):
                    if current_parent_id:
                        child_count[current_parent_id] = max(0, child_count.get(current_parent_id, 1) - 1)
                    parent_by_claim[claim_id] = best_parent_id
                    child_count[best_parent_id] = child_count.get(best_parent_id, 0) + 1
                    current_parent_id = best_parent_id
                    current_parent_claim = claim_by_id.get(best_parent_id)

        if current_parent_claim is None or current_parent_claim.claim_type != ClaimType.method:
            continue

        parent_level = _method_abstraction_level(current_parent_claim)
        violates = False
        if child_level == "primitive" and parent_level in {"system_realization", "submechanism"}:
            violates = True
        elif child_level == "system_realization":
            if parent_level == "submechanism":
                violates = True
            elif parent_level == "primitive" and not _has_explicit_instantiation_evidence(claim, current_parent_claim):
                violates = True
        elif child_level == "submechanism":
            # Allowed under primitive/system/submechanism; no special-case.
            violates = False
        if parent_level == "submechanism" and child_level in {"primitive", "system_realization"}:
            violates = True

        if not violates:
            continue

        replacement_parent: str | None = None
        if child_level == "submechanism" and parentable_method_claims:
            preferred_parents = [
                candidate
                for candidate in parentable_method_claims
                if candidate.claim_id != claim_id and _method_abstraction_level(candidate) in {"primitive", "system_realization"}
            ]
            if preferred_parents:
                replacement_parent, _ = _best_method_parent(claim, preferred_parents)
        if replacement_parent is None and context_claims:
            best_context_id, context_score = _best_context_parent(claim, context_claims)
            if best_context_id is not None and context_score > 0:
                replacement_parent = best_context_id
            elif context_ids:
                replacement_parent = context_ids[0]
        if replacement_parent is None:
            continue
        if _would_create_parent_cycle(claim_id, replacement_parent, parent_by_claim):
            continue

        child_count[current_parent_id] = max(0, child_count.get(current_parent_id, 1) - 1)
        parent_by_claim[claim_id] = replacement_parent
        child_count[replacement_parent] = child_count.get(replacement_parent, 0) + 1

    # Reattach results/assumptions/negatives to the most specific method when warranted.
    for claim_id in claim_order:
        claim = claim_by_id[claim_id]
        if claim.claim_type not in {ClaimType.result, ClaimType.assumption, ClaimType.negative}:
            continue
        if not parentable_method_claims:
            continue

        current_parent = parent_by_claim.get(claim_id)
        best_method_id, best_score = _best_method_parent(claim, parentable_method_claims)
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
        overloaded_parent = child_count.get(current_parent, 0) >= 4
        better_specific_match = best_score >= current_score + 3
        result_scope = _classify_result_scope(claim)
        scope_prefers_best = (
            result_scope == "top_level"
            and _is_system_realization_method(claim_by_id[best_method_id])
            and not _is_system_realization_method(current_parent_claim)
        )
        should_reattach = best_method_id != current_parent and (
            scope_prefers_best or
            better_specific_match or (overloaded_parent and best_score > current_score)
        )
        if not should_reattach:
            continue
        if _would_create_parent_cycle(claim_id, best_method_id, parent_by_claim):
            continue
        child_count[current_parent] = max(0, child_count.get(current_parent, 1) - 1)
        parent_by_claim[claim_id] = best_method_id
        child_count[best_method_id] = child_count.get(best_method_id, 0) + 1

    # Subtree-local restatement compaction for method parent-child pairs.
    compaction_index = 1
    changed = True
    while changed:
        changed = False
        children_by_parent: dict[str, list[str]] = {}
        for child_id, parent_id in parent_by_claim.items():
            children_by_parent.setdefault(parent_id, []).append(child_id)

        for parent_id in list(claim_order):
            parent_claim = claim_by_id.get(parent_id)
            if parent_claim is None or parent_claim.claim_type != ClaimType.method:
                continue
            for child_id in list(children_by_parent.get(parent_id, [])):
                child_claim = claim_by_id.get(child_id)
                if child_claim is None or child_claim.claim_type != ClaimType.method:
                    continue
                if not _is_restatement_pair(parent_claim, child_claim):
                    continue

                parent_score = _claim_rank_score(parent_claim)
                child_score = _claim_rank_score(child_claim)
                if child_score > parent_score + 0.15:
                    survivor_id = child_id
                    demoted_id = parent_id
                else:
                    survivor_id = parent_id
                    demoted_id = child_id

                demoted_claim = claim_by_id.get(demoted_id)
                if demoted_claim is None:
                    continue

                generated_support_details.append(
                    _compaction_support_detail(
                        demoted_claim,
                        anchor_claim_id=survivor_id,
                        index=compaction_index,
                    )
                )
                compaction_index += 1
                folded_support_evidence.setdefault(survivor_id, []).extend(demoted_claim.evidence)

                if demoted_id == child_id:
                    # Keep parent and lift child descendants.
                    for node_id, node_parent in list(parent_by_claim.items()):
                        if node_parent != child_id:
                            continue
                        if _would_create_parent_cycle(node_id, survivor_id, parent_by_claim):
                            continue
                        parent_by_claim[node_id] = survivor_id
                        deps = depends_by_claim.get(node_id, [])
                        if deps == [child_id]:
                            depends_by_claim[node_id] = [survivor_id]
                else:
                    # Keep child and collapse parent summary into support detail.
                    grandparent_id = parent_by_claim.get(parent_id)
                    if grandparent_id and not _would_create_parent_cycle(survivor_id, grandparent_id, parent_by_claim):
                        parent_by_claim[survivor_id] = grandparent_id
                    else:
                        parent_by_claim.pop(survivor_id, None)
                    for node_id, node_parent in list(parent_by_claim.items()):
                        if node_parent != parent_id or node_id == survivor_id:
                            continue
                        if _would_create_parent_cycle(node_id, survivor_id, parent_by_claim):
                            continue
                        parent_by_claim[node_id] = survivor_id
                        deps = depends_by_claim.get(node_id, [])
                        if deps == [parent_id]:
                            depends_by_claim[node_id] = [survivor_id]

                parent_by_claim.pop(demoted_id, None)
                depends_by_claim.pop(demoted_id, None)
                for node_id, deps in list(depends_by_claim.items()):
                    if demoted_id not in deps:
                        continue
                    repaired: list[str] = []
                    for dep in deps:
                        if dep == demoted_id:
                            if survivor_id != node_id and survivor_id not in repaired:
                                repaired.append(survivor_id)
                            continue
                        if dep != node_id and dep not in repaired:
                            repaired.append(dep)
                    depends_by_claim[node_id] = repaired

                if demoted_id in claim_order:
                    claim_order.remove(demoted_id)
                claim_by_id.pop(demoted_id, None)
                changed = True
                break
            if changed:
                break

    # Sanitize depends_on edges that contradict the structural tree.
    for claim_id in claim_order:
        dependencies = depends_by_claim.get(claim_id, [])
        if not dependencies:
            continue
        cleaned: list[str] = []
        for dependency_id in dependencies:
            if dependency_id == claim_id:
                continue
            if dependency_id not in claim_by_id:
                continue
            if _depends_on_descendant(claim_id, dependency_id, parent_by_claim):
                continue
            if dependency_id in cleaned:
                continue
            cleaned.append(dependency_id)
        depends_by_claim[claim_id] = cleaned

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

    abstraction_by_claim: dict[str, AbstractionLevel] = {}
    role_by_claim: dict[str, SemanticRole] = {}
    subtype_by_claim: dict[str, ResultSubtype | None] = {}
    normalized_by_claim: dict[str, str] = {}
    label_by_claim: dict[str, str] = {}
    used_labels: set[str] = set()

    for claim_id in claim_order:
        claim = claim_by_id[claim_id]
        abstraction = _abstraction_level_for_claim(claim)
        subtype = _classify_result_subtype(claim)
        role = _semantic_role_for_claim(claim, abstraction, subtype)
        normalized_statement = _normalize_statement_conservative(claim.statement)

        base_label = "_".join(_canonical_label_parts(claim, normalized_statement))
        if not base_label:
            base_label = f"{claim.claim_type.value}_claim"
        label = base_label
        suffix = 2
        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        used_labels.add(label)

        abstraction_by_claim[claim_id] = abstraction
        subtype_by_claim[claim_id] = subtype
        role_by_claim[claim_id] = role
        normalized_by_claim[claim_id] = normalized_statement
        label_by_claim[claim_id] = label

    nodes = {
        claim_id: ClaimNode(
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            abstraction_level=abstraction_by_claim[claim_id],
            semantic_role=role_by_claim[claim_id],
            canonical_label=label_by_claim[claim_id],
            normalized_statement=normalized_by_claim[claim_id],
            result_subtype=subtype_by_claim[claim_id],
            statement=claim.statement,
            evidence=_merge_evidence_pointers(
                list(claim.evidence),
                folded_support_evidence.get(claim_id, []),
            ),
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

    roots = [nodes[claim_id] for claim_id in claim_order if claim_id not in parent_by_claim]
    metrics: dict[str, float | int] = {
        "initial_node_count": initial_node_count,
        "final_node_count": len(claim_order),
        "compaction_demotions": len(generated_support_details),
    }
    return roots, generated_support_details, metrics


def _read_tree_float(config: Any, key: str, default: float) -> float:
    pipeline = getattr(config, "pipeline", None)
    tree_cfg = getattr(pipeline, "tree", None) if pipeline is not None else None
    value: Any = None
    if isinstance(tree_cfg, Mapping):
        value = tree_cfg.get(key)
    elif isinstance(config, Mapping):
        pipeline_cfg = config.get("pipeline")
        if isinstance(pipeline_cfg, Mapping):
            raw_tree = pipeline_cfg.get("tree")
            if isinstance(raw_tree, Mapping):
                value = raw_tree.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_tree_int(config: Any, key: str, default: int) -> int:
    pipeline = getattr(config, "pipeline", None)
    tree_cfg = getattr(pipeline, "tree", None) if pipeline is not None else None
    value: Any = None
    if isinstance(tree_cfg, Mapping):
        value = tree_cfg.get(key)
    elif isinstance(config, Mapping):
        pipeline_cfg = config.get("pipeline")
        if isinstance(pipeline_cfg, Mapping):
            raw_tree = pipeline_cfg.get("tree")
            if isinstance(raw_tree, Mapping):
                value = raw_tree.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ambiguity_budget(non_context_count: int, config: Any) -> int:
    configured = _read_tree_int(config, "ambiguity_budget", -1)
    if configured >= 0:
        return max(0, configured)
    return min(8, int(math.ceil(0.15 * max(0, non_context_count))))


def _deterministic_one_liner(claims: list[RawClaim]) -> OneLiner:
    context_claim = next((claim for claim in claims if claim.claim_type == ClaimType.context), None)
    method_claim = next((claim for claim in claims if claim.claim_type == ClaimType.method), None)
    result_claim = next((claim for claim in claims if claim.claim_type == ClaimType.result), None)
    fallback = claims[0].statement if claims else "UNSPECIFIED"
    return OneLiner(
        achieved=_clean_text(result_claim.statement if result_claim is not None else fallback) or "UNSPECIFIED",
        via=_clean_text(method_claim.statement if method_claim is not None else fallback) or "UNSPECIFIED",
        because=_clean_text(context_claim.statement if context_claim is not None else fallback) or "UNSPECIFIED",
    )


def _candidate_parent_scores(
    claim: RawClaim,
    claims: list[RawClaim],
) -> list[tuple[RawClaim, float]]:
    allowed_parent_types = _ALLOWED_PARENT_TYPES.get(claim.claim_type, set())
    candidates = [
        parent
        for parent in claims
        if parent.claim_id != claim.claim_id and parent.claim_type in allowed_parent_types
    ]
    scored = [(parent, parent_confidence_score(claim, parent)) for parent in candidates]
    scored.sort(key=lambda item: (item[1], item[0].claim_id), reverse=True)
    return scored


def _build_deterministic_assignments(
    claims: list[RawClaim],
    *,
    parent_hints: dict[str, str],
    accept_threshold: float,
    margin_threshold: float,
    ambiguity_budget: int,
) -> tuple[list[TreeNodeAssignment], list[dict[str, Any]]]:
    claim_by_id = {claim.claim_id: claim for claim in claims}
    assignments: dict[str, TreeNodeAssignment] = {}
    ambiguities: list[dict[str, Any]] = []

    for claim in claims:
        if claim.claim_type == ClaimType.context:
            assignments[claim.claim_id] = TreeNodeAssignment(
                claim_id=claim.claim_id,
                parent_id=None,
                depends_on=[],
            )
            continue

        hinted_parent = parent_hints.get(claim.claim_id)
        if hinted_parent in claim_by_id and hinted_parent != claim.claim_id:
            assignments[claim.claim_id] = TreeNodeAssignment(
                claim_id=claim.claim_id,
                parent_id=hinted_parent,
                depends_on=[hinted_parent],
            )
            continue

        scored = _candidate_parent_scores(claim, claims)
        if not scored:
            assignments[claim.claim_id] = TreeNodeAssignment(
                claim_id=claim.claim_id,
                parent_id=None,
                depends_on=[],
            )
            continue

        top_parent, top_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        margin = top_score - second_score
        assignments[claim.claim_id] = TreeNodeAssignment(
            claim_id=claim.claim_id,
            parent_id=top_parent.claim_id,
            depends_on=[top_parent.claim_id],
        )

        if top_score >= accept_threshold and margin >= margin_threshold:
            continue

        ambiguities.append(
            {
                "claim": claim,
                "top_score": top_score,
                "margin": margin,
                "candidates": scored[:4],
            }
        )

    if ambiguity_budget <= 0 or not ambiguities:
        return list(assignments.values()), []

    ambiguities.sort(key=lambda item: (item["top_score"], item["margin"]))
    return list(assignments.values()), ambiguities[:ambiguity_budget]


def _build_ambiguity_prompt(
    ambiguities: list[dict[str, Any]],
) -> list[dict[str, str]]:
    lines: list[str] = []
    for idx, entry in enumerate(ambiguities, start=1):
        claim = cast(RawClaim, entry["claim"])
        lines.append(f"{idx}. CLAIM [{claim.claim_id}] ({claim.claim_type.value}): {claim.statement}")
        for rank, (candidate, score) in enumerate(cast(list[tuple[RawClaim, float]], entry["candidates"]), start=1):
            lines.append(
                f"   - C{rank} [{candidate.claim_id}] ({candidate.claim_type.value}) score={score:.3f}: {candidate.statement}"
            )

    return [
        {
            "role": "system",
            "content": (
                "Resolve ambiguous parent assignments. Choose one parent_id per claim from listed candidates only. "
                "Respect type grammar and prefer the most specific semantically aligned parent."
            ),
        },
        {
            "role": "user",
            "content": (
                "Ambiguous claims and candidate parents:\\n"
                + "\\n".join(lines)
                + "\\n\\nReturn JSON with key `nodes`: [{claim_id, parent_id, depends_on}]."
            ),
        },
    ]


async def _resolve_ambiguities(
    ambiguities: list[dict[str, Any]],
    config: Any,
) -> dict[str, TreeNodeAssignment]:
    if not ambiguities:
        return {}

    messages = _build_ambiguity_prompt(ambiguities)
    result = await call_model(
        tier=_resolve_model_tier(config),
        messages=messages,
        response_schema=AmbiguityResolutionOutput,
        config=config,
    )
    if not isinstance(result, AmbiguityResolutionOutput):
        return {}

    allowed_by_claim: dict[str, set[str]] = {}
    for entry in ambiguities:
        claim = cast(RawClaim, entry["claim"])
        candidate_ids = {candidate.claim_id for candidate, _ in cast(list[tuple[RawClaim, float]], entry["candidates"])}
        allowed_by_claim[claim.claim_id] = candidate_ids

    resolved: dict[str, TreeNodeAssignment] = {}
    for node in result.nodes:
        allowed = allowed_by_claim.get(node.claim_id)
        if not allowed:
            continue
        parent_id = (node.parent_id or "").strip()
        if parent_id not in allowed:
            continue
        resolved[node.claim_id] = TreeNodeAssignment(
            claim_id=node.claim_id,
            parent_id=parent_id,
            depends_on=[parent_id],
        )
    return resolved


async def assemble_tree_deterministic(
    metadata: PaperMetadata,
    claims: list[RawClaim],
    faceted: list[FacetedClaim],
    negatives: list[RawClaim],
    artifacts: list[EvidenceArtifact],
    config: Any,
    claim_groups: list[ClaimGroup] | None = None,
    support_details: list[Any] | None = None,
) -> PaperDecomposition:
    normalized_negatives = list(negatives) or [claim for claim in claims if claim.claim_type == ClaimType.negative]
    combined_claims = list(claims)
    known_ids = {claim.claim_id for claim in combined_claims}
    for claim in normalized_negatives:
        if claim.claim_id in known_ids:
            continue
        known_ids.add(claim.claim_id)
        combined_claims.append(claim)

    parent_hints = _canonical_parent_hints(combined_claims, claim_groups)
    accept_threshold = _read_tree_float(config, "parent_accept_threshold", DEFAULT_PARENT_ACCEPT_THRESHOLD)
    margin_threshold = _read_tree_float(config, "parent_margin_threshold", DEFAULT_PARENT_MARGIN_THRESHOLD)
    non_context_count = sum(1 for claim in combined_claims if claim.claim_type != ClaimType.context)
    budget = _ambiguity_budget(non_context_count, config)

    assignments, ambiguities = _build_deterministic_assignments(
        combined_claims,
        parent_hints=parent_hints,
        accept_threshold=accept_threshold,
        margin_threshold=margin_threshold,
        ambiguity_budget=budget,
    )
    assignment_by_id = {assignment.claim_id: assignment for assignment in assignments}

    if ambiguities:
        try:
            resolved = await _resolve_ambiguities(ambiguities, config)
        except Exception:
            resolved = {}
        assignment_by_id.update(resolved)

    final_assignments = [assignment_by_id[claim.claim_id] for claim in combined_claims if claim.claim_id in assignment_by_id]
    claim_tree, compacted_support_details, _ = _build_tree_nodes(
        combined_claims,
        faceted,
        final_assignments,
        parent_hints=parent_hints,
    )
    merged_support_details = [*list(support_details or []), *compacted_support_details]

    return PaperDecomposition(
        metadata=metadata,
        one_liner=_deterministic_one_liner(combined_claims),
        claim_tree=claim_tree,
        negative_claims=normalized_negatives,
        support_details=merged_support_details,
        all_artifacts=list(artifacts),
    )


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
    claim_tree, compacted_support_details, _ = _build_tree_nodes(
        combined_claims,
        faceted,
        result.nodes,
        parent_hints=parent_hints,
    )
    return PaperDecomposition(
        metadata=metadata,
        one_liner=_one_liner_or_unspecified(result.one_liner),
        claim_tree=claim_tree,
        negative_claims=normalized_negatives,
        support_details=compacted_support_details,
        all_artifacts=list(artifacts),
    )


__all__ = [
    "TREE_SYSTEM_PROMPT",
    "build_tree_prompt",
    "parent_confidence_score",
    "assemble_tree_deterministic",
    "assemble_tree",
]
