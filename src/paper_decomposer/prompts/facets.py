from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from ..models import call_model_with_fallback
from ..schema import (
    AlgorithmFacets,
    ArchitectureFacets,
    ClaimType,
    EvaluationFacets,
    FacetedClaim,
    FlatInterventionClassification,
    FlatUniversalFacets,
    GroundingType,
    InterventionType,
    ModelTier,
    ObjectiveFacets,
    PipelineFacets,
    RawClaim,
    RepresentationFacets,
    ScopeOfChange,
    Section,
    StackLayer,
    SystemsFacets,
    TheoryFacets,
    UniversalFacets,
)

call_model = call_model_with_fallback

SourceSection = Section | str
PromptBuilder = Callable[[RawClaim, SourceSection], list[dict[str, str]]]
FacetSchema = type[BaseModel]
TFacetModel = TypeVar("TFacetModel", bound=BaseModel)

_VALID_TIERS: set[ModelTier] = {"small", "medium", "heavy"}
_STACK_LAYER_CHOICES = ", ".join(layer.value for layer in StackLayer)

CLASSIFY_SYSTEM_PROMPT = """You classify method claims from research papers.
Choose exactly one primary intervention type based on the claim and section context.

Intervention types:
- architecture: new model structure or module arrangement
- objective: new loss function or training target
- algorithm: new optimization, inference, search, or sampling procedure
- representation: new encoding, feature space, or embedding formulation
- data: new dataset, curation method, or labeling strategy
- systems: infrastructure, kernel, runtime, scheduling, or resource-management mechanism
- theory: theorem, bound, proof, or formal analysis framework
- evaluation: benchmark, protocol, or measurement methodology
- pipeline: novel composition of existing components/stages
"""

_DOMAIN_SYSTEM_PROMPT = """You extract domain facets for method claims.
Answer each field in <=15 words.
If a field cannot be inferred from the provided text, return "UNSPECIFIED".
Do not guess.
Never copy empirical outcomes (throughput/memory/latency improvements) into
mechanism fields. Facets must describe the method, not its results.
Never introduce analogies unless explicitly stated in the source text.
Never invent hardware assumptions; if not explicitly stated, return "UNSPECIFIED".
Use vocabulary grounded in the claim/section text, not plausible filler.
For enum-constrained fields, return only an allowed enum token."""

_UNIVERSAL_SYSTEM_PROMPT = """You extract universal method facets.
Use only evidence from the claim and section context.
For free-text fields: answer in <=15 words.
If unknown, return "UNSPECIFIED".
Do not guess.
Do not invent analogies; if no explicit analogy appears, return "UNSPECIFIED".
Use vocabulary grounded in the claim/section text, not plausible filler.
For enum fields, output only exact enum values."""

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
_UNSPECIFIED_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "not specified",
    "not stated",
    "unknown",
    "unspecified",
}
_GENERIC_FACET_TOKENS = {
    "approach",
    "baseline",
    "component",
    "constraint",
    "cost",
    "design",
    "implementation",
    "method",
    "model",
    "module",
    "policy",
    "process",
    "resource",
    "result",
    "runtime",
    "system",
    "tradeoff",
}
_ANALOGY_LANGUAGE_RE = re.compile(
    r"\b(analogy|analogous|analog|like|akin|metaphor|similar to|as if)\b",
    re.IGNORECASE,
)
_ANALOGY_CONTEXT_RE = re.compile(r"\b(analogy|analog|inspired by|similar to|like)\b", re.IGNORECASE)
_HARDWARE_CUE_RE = re.compile(
    r"\b(gpu|cpu|tpu|hbm|dram|vram|nvlink|pcie|cuda|memory|bandwidth|device|accelerator|hardware|interconnect)\b",
    re.IGNORECASE,
)
_VALID_STACK_LAYER_VALUES = {layer.value for layer in StackLayer}


def _resolve_model_tier(config: Any) -> ModelTier:
    if config is None:
        return "small"

    pipeline = getattr(config, "pipeline", None)
    if pipeline is not None:
        for field in ("facets", "facet_extraction", "section_extraction"):
            section_cfg = getattr(pipeline, field, None)
            if isinstance(section_cfg, Mapping):
                tier = section_cfg.get("model_tier")
                if isinstance(tier, str) and tier in _VALID_TIERS:
                    return cast(ModelTier, tier)

    if isinstance(config, Mapping):
        pipeline_cfg = config.get("pipeline")
        if isinstance(pipeline_cfg, Mapping):
            for field in ("facets", "facet_extraction", "section_extraction"):
                section_cfg = pipeline_cfg.get(field)
                if isinstance(section_cfg, Mapping):
                    tier = section_cfg.get("model_tier")
                    if isinstance(tier, str) and tier in _VALID_TIERS:
                        return cast(ModelTier, tier)

    return "small"


def _section_label(section: SourceSection) -> str:
    if isinstance(section, Section):
        if section.section_number:
            return f"{section.section_number} {section.title}"
        return section.title
    return "source excerpt"


def _section_text(section: SourceSection) -> str:
    if isinstance(section, Section):
        return section.body_text
    return section


def _domain_prompt(
    claim: RawClaim,
    section: SourceSection,
    *,
    domain_name: str,
    questions: list[str],
) -> list[dict[str, str]]:
    section_label = _section_label(section)
    section_text = _section_text(section)
    question_block = "\n".join(questions)
    user_content = (
        f"Claim:\n{claim.statement}\n\n"
        f"Section ({section_label}):\n{section_text}\n\n"
        f"{domain_name.upper()} FACET QUESTIONS:\n"
        f"{question_block}\n\n"
        'Return valid JSON matching the schema. Use "UNSPECIFIED" when unsupported.'
    )
    return [
        {"role": "system", "content": _DOMAIN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _to_intervention_key(intervention_type: InterventionType | str) -> str:
    if isinstance(intervention_type, InterventionType):
        return intervention_type.value
    return str(intervention_type).strip().lower()


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _is_unspecified(value: str | None) -> bool:
    if value is None:
        return True
    return _clean_text(value).lower() in _UNSPECIFIED_VALUES


def _tokenize(text: str) -> set[str]:
    lowered = _clean_text(text).lower()
    return {
        token
        for token in _TOKEN_SPLIT_RE.split(lowered)
        if token and len(token) > 2 and token not in _STOPWORDS
    }


def _token_variants(token: str) -> set[str]:
    variants = {token}
    if token.endswith("ies") and len(token) > 5:
        variants.add(f"{token[:-3]}y")
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            variants.add(token[: -len(suffix)])
    return variants


def _token_supported(token: str, source_tokens: set[str]) -> bool:
    return any(variant in source_tokens for variant in _token_variants(token))


def _is_supported_by_source(value: str, source_tokens: set[str]) -> bool:
    candidate_tokens = {
        token
        for token in _tokenize(value)
        if token not in _GENERIC_FACET_TOKENS and token not in _UNSPECIFIED_VALUES
    }
    if not candidate_tokens:
        return False

    supported = {token for token in candidate_tokens if _token_supported(token, source_tokens)}
    if not supported:
        return False
    return (len(supported) / len(candidate_tokens)) >= 0.6


def _sanitize_facet_text(
    value: str | None,
    *,
    source_text: str,
    source_tokens: set[str],
    allow_analogy: bool = False,
    require_hardware: bool = False,
) -> str:
    if _is_unspecified(value):
        return "UNSPECIFIED"

    assert value is not None  # helps type-checkers
    cleaned = _clean_text(value)
    if not cleaned:
        return "UNSPECIFIED"

    has_analogy_language = bool(_ANALOGY_LANGUAGE_RE.search(cleaned))
    if allow_analogy and not _ANALOGY_CONTEXT_RE.search(source_text):
        return "UNSPECIFIED"
    if has_analogy_language and not allow_analogy:
        return "UNSPECIFIED"

    if require_hardware:
        if not _HARDWARE_CUE_RE.search(cleaned):
            return "UNSPECIFIED"
        if not _HARDWARE_CUE_RE.search(source_text):
            return "UNSPECIFIED"

    if not _is_supported_by_source(cleaned, source_tokens):
        return "UNSPECIFIED"
    return cleaned


def _normalize_stack_layer_value(value: str) -> str:
    normalized = _normalize_token(value)
    aliases = {
        "hardware": StackLayer.hardware.value,
        "hw": StackLayer.hardware.value,
        "gpu": StackLayer.hardware.value,
        "cpu": StackLayer.hardware.value,
        "accelerator": StackLayer.hardware.value,
        "device": StackLayer.hardware.value,
        "os_kernel": StackLayer.os_kernel.value,
        "kernel": StackLayer.os_kernel.value,
        "os": StackLayer.os_kernel.value,
        "operating_system": StackLayer.os_kernel.value,
        "runtime": StackLayer.runtime.value,
        "inference_runtime": StackLayer.runtime.value,
        "application_level": StackLayer.application_level.value,
        "application": StackLayer.application_level.value,
        "service_layer": StackLayer.application_level.value,
    }
    return aliases.get(normalized, normalized)


def _source_context(claim: RawClaim, section: SourceSection) -> tuple[str, set[str]]:
    combined = _clean_text(f"{claim.statement}\n{_section_text(section)}")
    source_tokens = _tokenize(combined)
    return combined.lower(), source_tokens


def _normalize_intervention_type(value: str) -> InterventionType:
    normalized = _normalize_token(value)
    aliases = {
        "architectural": "architecture",
        "system": "systems",
        "infra": "systems",
        "alg": "algorithm",
    }
    try:
        return InterventionType(aliases.get(normalized, normalized))
    except ValueError:
        return InterventionType.systems


def _normalize_scope(value: str) -> ScopeOfChange:
    normalized = _normalize_token(value)
    aliases = {
        "dropin": "drop_in",
        "drop_in_replacement": "drop_in",
        "component": "module",
        "full_system": "system",
    }
    try:
        return ScopeOfChange(aliases.get(normalized, normalized))
    except ValueError:
        return ScopeOfChange.module


def _normalize_grounding(value: str) -> GroundingType:
    normalized = _normalize_token(value)
    aliases = {
        "formal_empirical": "formal_and_empirical",
        "formal_plus_empirical": "formal_and_empirical",
        "empirical": "empirical_demo",
        "controlled_empirical": "empirical_controlled",
    }
    try:
        return GroundingType(aliases.get(normalized, normalized))
    except ValueError:
        return GroundingType.qualitative


def _to_universal_facets(
    flat: FlatUniversalFacets,
    classification: InterventionType,
) -> UniversalFacets:
    intervention_types: list[InterventionType] = []
    for raw in flat.intervention_types:
        normalized = _normalize_intervention_type(raw)
        if normalized not in intervention_types:
            intervention_types.append(normalized)

    if classification not in intervention_types:
        intervention_types = [classification, *intervention_types]

    analogy = _clean_text(flat.analogy_source) if flat.analogy_source else ""
    if _is_unspecified(analogy):
        analogy = ""
    return UniversalFacets(
        intervention_types=intervention_types[:2],
        scope=_normalize_scope(flat.scope),
        improves_or_replaces=_clean_text(flat.improves_or_replaces) or "UNSPECIFIED",
        core_tradeoff=_clean_text(flat.core_tradeoff) or "UNSPECIFIED",
        grounding=_normalize_grounding(flat.grounding),
        analogy_source=analogy or None,
    )


def build_classify_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    section_label = _section_label(section)
    section_text = _section_text(section)
    user_content = (
        f"Claim:\n{claim.statement}\n\n"
        f"Section ({section_label}):\n{section_text}\n\n"
        "Select exactly one intervention_type enum value."
    )
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_systems_facet_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    return _domain_prompt(
        claim,
        section,
        domain_name="systems",
        questions=[
            "S1 (s1_resource). What resource is managed (e.g., GPU memory, bandwidth, compute)?",
            "S2 (s2_alloc_unit). What concrete allocation/scheduling unit is used (e.g., KV block, page, token chunk)?",
            f"S3 (s3_stack_layer). Choose exactly one from [{_STACK_LAYER_CHOICES}].",
            "S4 (s4_mapping). State mapping as logical entity -> physical entity via mechanism.",
            "S5 (s5_policy). What policy governs dynamic decisions?",
            "S6 (s6_hw_assumption). What hardware/platform assumption is required?",
        ],
    )


def build_architecture_facet_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    return _domain_prompt(
        claim,
        section,
        domain_name="architecture",
        questions=[
            "A1 (a1). What primary structural element is introduced or changed?",
            "A2 (a2). What computational pattern is newly enabled?",
            "A3 (a3). What scaling behavior change is claimed?",
            "A4 (a4). What invariance or structural constraint is preserved?",
            "A5 (a5). What parameterization change is introduced?",
        ],
    )


def build_objective_facet_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    return _domain_prompt(
        claim,
        section,
        domain_name="objective",
        questions=[
            "O1 (o1). What optimization signal is added or modified?",
            "O2 (o2). How does that signal relate to the target task goal?",
            "O3 (o3). How is differentiability handled?",
            "O4 (o4). Which failure mode or bias is addressed?",
        ],
    )


def build_algorithm_facet_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    return _domain_prompt(
        claim,
        section,
        domain_name="algorithm",
        questions=[
            "G1 (g1). What algorithm class is this (search, sampling, optimizer, planner)?",
            "G2 (g2). Which primitive operation is modified?",
            "G3 (g3). What guarantee/target property is claimed (e.g., convergence, correctness, bounded error, complexity)? If only empirical gains are given, return UNSPECIFIED.",
            "G4 (g4). What iteration/control-flow structure is used?",
            "G5 (g5). What state or cache is maintained between steps?",
        ],
    )


def build_theory_facet_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    return _domain_prompt(
        claim,
        section,
        domain_name="theory",
        questions=[
            "T1 (t1). What formal object is analyzed?",
            "T2 (t2). What result type is provided (bound/theorem/equivalence)?",
            "T3 (t3). What assumptions are required?",
            "T4 (t4). What proof or analysis technique is used?",
            "T5 (t5). What scaling/asymptotic claim is made?",
        ],
    )


def build_representation_facet_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    return _domain_prompt(
        claim,
        section,
        domain_name="representation",
        questions=[
            "P1 (p1). What input domain is represented?",
            "P2 (p2). What structure or encoding is introduced?",
            "P3 (p3). What property is preserved by this representation?",
            "P4 (p4). What dimensionality or compression behavior is claimed?",
        ],
    )


def build_evaluation_facet_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    return _domain_prompt(
        claim,
        section,
        domain_name="evaluation",
        questions=[
            "E1 (e1). What capability or behavior is evaluated?",
            "E2 (e2). What dataset/benchmark/source provides evaluation data?",
            "E3 (e3). What scoring metric or protocol is used?",
        ],
    )


def build_pipeline_facet_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    return _domain_prompt(
        claim,
        section,
        domain_name="pipeline",
        questions=[
            "L1 (l1). What pipeline components are composed?",
            "L2 (l2). What control flow/order connects components?",
            "L3 (l3). What composition novelty gives the improvement?",
        ],
    )


def build_data_facet_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    return _domain_prompt(
        claim,
        section,
        domain_name="data",
        questions=[
            "E1 (e1). What dataset asset or data capability is introduced?",
            "E2 (e2). What data source/curation/filtering method is used?",
            "E3 (e3). What data quality or evaluation criterion is reported?",
        ],
    )


def build_universal_prompt(claim: RawClaim, section: SourceSection) -> list[dict[str, str]]:
    section_label = _section_label(section)
    section_text = _section_text(section)
    user_content = (
        f"Claim:\n{claim.statement}\n\n"
        f"Section ({section_label}):\n{section_text}\n\n"
        "Fill universal facets:\n"
        "- intervention_types: 1-2 values from InterventionType enum.\n"
        "- scope: one of drop_in/module/system/paradigm.\n"
        "- improves_or_replaces: what baseline thing is improved/replaced.\n"
        "- core_tradeoff: compactly state the main tradeoff.\n"
        "- grounding: one of formal_only/formal_and_empirical/empirical_controlled/"
        "empirical_demo/qualitative.\n"
        '- analogy_source: short analogy, or "UNSPECIFIED".\n\n'
        'Return valid JSON matching the schema. Use "UNSPECIFIED" when unsupported.'
    )
    return [
        {"role": "system", "content": _UNIVERSAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


_FACET_PROMPT_BUILDERS: dict[str, PromptBuilder] = {
    "architecture": build_architecture_facet_prompt,
    "objective": build_objective_facet_prompt,
    "algorithm": build_algorithm_facet_prompt,
    "representation": build_representation_facet_prompt,
    "data": build_data_facet_prompt,
    "systems": build_systems_facet_prompt,
    "theory": build_theory_facet_prompt,
    "evaluation": build_evaluation_facet_prompt,
    "pipeline": build_pipeline_facet_prompt,
}

_FACET_SCHEMA_BY_TYPE: dict[str, FacetSchema] = {
    "architecture": ArchitectureFacets,
    "objective": ObjectiveFacets,
    "algorithm": AlgorithmFacets,
    "representation": RepresentationFacets,
    "data": EvaluationFacets,
    "systems": SystemsFacets,
    "theory": TheoryFacets,
    "evaluation": EvaluationFacets,
    "pipeline": PipelineFacets,
}

_FACET_FIELD_BY_TYPE: dict[str, str] = {
    "architecture": "architecture_facets",
    "objective": "objective_facets",
    "algorithm": "algorithm_facets",
    "representation": "representation_facets",
    "data": "evaluation_facets",
    "systems": "systems_facets",
    "theory": "theory_facets",
    "evaluation": "evaluation_facets",
    "pipeline": "pipeline_facets",
}


def get_facet_prompt(intervention_type: InterventionType | str) -> PromptBuilder:
    key = _to_intervention_key(intervention_type)
    return _FACET_PROMPT_BUILDERS[key]


def get_facet_schema(intervention_type: InterventionType | str) -> FacetSchema:
    key = _to_intervention_key(intervention_type)
    return _FACET_SCHEMA_BY_TYPE[key]


async def _call_typed_model(
    *,
    tier: ModelTier,
    messages: list[dict[str, str]],
    schema: type[TFacetModel],
    config: Any,
) -> TFacetModel:
    result = await call_model(
        tier=tier,
        messages=messages,
        response_schema=schema,
        config=config,
    )
    if isinstance(result, schema):
        return result
    raise TypeError(f"Expected {schema.__name__} from structured model call.")


_GUARANTEE_HINT = re.compile(
    r"(converg|bound|guarante|correct|consisten|stable|monotonic|complexit|o\()",
    re.IGNORECASE,
)
_EMPIRICAL_RESULT_HINT = re.compile(
    r"(throughput|latenc|memory|faster|slower|speedup|improv|worse|better|percent|%|\bx\b|\d)",
    re.IGNORECASE,
)


def _normalize_algorithm_facets(facets: AlgorithmFacets) -> AlgorithmFacets:
    g3 = facets.g3.strip()
    if not g3:
        return facets.model_copy(update={"g3": "UNSPECIFIED"})
    if _GUARANTEE_HINT.search(g3):
        return facets
    if _EMPIRICAL_RESULT_HINT.search(g3):
        return facets.model_copy(update={"g3": "UNSPECIFIED"})
    return facets


def _sanitize_domain_facets(
    domain_facets: TFacetModel,
    *,
    source_text: str,
    source_tokens: set[str],
) -> TFacetModel:
    if isinstance(domain_facets, SystemsFacets):
        normalized_stack_layer = _normalize_stack_layer_value(domain_facets.s3_stack_layer)
        if normalized_stack_layer not in _VALID_STACK_LAYER_VALUES:
            normalized_stack_layer = "UNSPECIFIED"
        sanitized = domain_facets.model_copy(
            update={
                "s1_resource": _sanitize_facet_text(
                    domain_facets.s1_resource,
                    source_text=source_text,
                    source_tokens=source_tokens,
                ),
                "s2_alloc_unit": _sanitize_facet_text(
                    domain_facets.s2_alloc_unit,
                    source_text=source_text,
                    source_tokens=source_tokens,
                ),
                "s3_stack_layer": normalized_stack_layer,
                "s4_mapping": _sanitize_facet_text(
                    domain_facets.s4_mapping,
                    source_text=source_text,
                    source_tokens=source_tokens,
                ),
                "s5_policy": _sanitize_facet_text(
                    domain_facets.s5_policy,
                    source_text=source_text,
                    source_tokens=source_tokens,
                ),
                "s6_hw_assumption": _sanitize_facet_text(
                    domain_facets.s6_hw_assumption,
                    source_text=source_text,
                    source_tokens=source_tokens,
                    require_hardware=True,
                ),
            }
        )
        return cast(TFacetModel, sanitized)

    sanitized_updates: dict[str, str] = {}
    for key, value in domain_facets.model_dump().items():
        sanitized_updates[key] = _sanitize_facet_text(
            str(value),
            source_text=source_text,
            source_tokens=source_tokens,
        )
    return cast(TFacetModel, domain_facets.model_copy(update=sanitized_updates))


def _sanitize_universal_facets(
    universal: UniversalFacets,
    *,
    classification: InterventionType,
    source_text: str,
    source_tokens: set[str],
) -> UniversalFacets:
    intervention_types = [classification]
    for intervention_type in universal.intervention_types:
        if intervention_type not in intervention_types:
            intervention_types.append(intervention_type)
        if len(intervention_types) == 2:
            break

    analogy = _sanitize_facet_text(
        universal.analogy_source,
        source_text=source_text,
        source_tokens=source_tokens,
        allow_analogy=True,
    )
    sanitized_analogy = None if analogy == "UNSPECIFIED" else analogy
    return universal.model_copy(
        update={
            "intervention_types": intervention_types,
            "improves_or_replaces": _sanitize_facet_text(
                universal.improves_or_replaces,
                source_text=source_text,
                source_tokens=source_tokens,
            ),
            "core_tradeoff": _sanitize_facet_text(
                universal.core_tradeoff,
                source_text=source_text,
                source_tokens=source_tokens,
            ),
            "analogy_source": sanitized_analogy,
        }
    )


async def extract_facets(claim: RawClaim, source_section: SourceSection, config: Any) -> FacetedClaim:
    if claim.claim_type != ClaimType.method:
        raise ValueError("extract_facets only supports METHOD claims.")

    tier = _resolve_model_tier(config)
    flat_classification = await _call_typed_model(
        tier=tier,
        messages=build_classify_prompt(claim, source_section),
        schema=FlatInterventionClassification,
        config=config,
    )
    intervention_type = _normalize_intervention_type(flat_classification.intervention_type)

    intervention_key = intervention_type.value
    facet_prompt_builder = get_facet_prompt(intervention_key)
    facet_schema = cast(type[TFacetModel], get_facet_schema(intervention_key))
    domain_facets = await _call_typed_model(
        tier=tier,
        messages=facet_prompt_builder(claim, source_section),
        schema=facet_schema,
        config=config,
    )
    if isinstance(domain_facets, AlgorithmFacets):
        domain_facets = cast(TFacetModel, _normalize_algorithm_facets(domain_facets))

    source_text, source_tokens = _source_context(claim, source_section)
    domain_facets = _sanitize_domain_facets(
        domain_facets,
        source_text=source_text,
        source_tokens=source_tokens,
    )

    flat_universal = await _call_typed_model(
        tier=tier,
        messages=build_universal_prompt(claim, source_section),
        schema=FlatUniversalFacets,
        config=config,
    )
    universal = _sanitize_universal_facets(
        _to_universal_facets(flat_universal, intervention_type),
        classification=intervention_type,
        source_text=source_text,
        source_tokens=source_tokens,
    )

    facet_field = _FACET_FIELD_BY_TYPE[intervention_key]
    return FacetedClaim(
        claim=claim,
        universal_facets=universal,
        **{facet_field: domain_facets},
    )


__all__ = [
    "CLASSIFY_SYSTEM_PROMPT",
    "build_classify_prompt",
    "build_systems_facet_prompt",
    "build_architecture_facet_prompt",
    "build_objective_facet_prompt",
    "build_algorithm_facet_prompt",
    "build_theory_facet_prompt",
    "build_representation_facet_prompt",
    "build_evaluation_facet_prompt",
    "build_pipeline_facet_prompt",
    "build_data_facet_prompt",
    "get_facet_prompt",
    "get_facet_schema",
    "build_universal_prompt",
    "extract_facets",
]
