from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
import re
from typing import Any, Awaitable, Callable, TypeVar

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .config import load_config
from .models import get_cost_tracker, preflight_model_tiers, reset_cost_tracker
from .pdf_parser import parse_pdf
from .prompts.dedup import (
    classify_method_abstraction,
    classify_result_family,
    compress_claims_to_skeleton,
)
from .prompts.facets import extract_facets
from .prompts.section import classify_support_detail_type, extract_section_digest, support_relationship_for_type
from .prompts.seed import extract_seed
from .prompts.tree import assemble_tree_deterministic
from .schema import (
    ClaimNode,
    ClaimStructuralHints,
    ClaimType,
    FacetedClaim,
    PaperDecomposition,
    PaperSkeletonCandidate,
    RawClaim,
    Section,
    SectionArgumentCandidate,
    SupportDetail,
    SupportDetailType,
    SupportRelationshipType,
)

console = Console()
T = TypeVar("T")
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


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _tokens(text: str) -> set[str]:
    lowered = _clean_text(text).lower()
    return {
        token
        for token in _TOKEN_SPLIT_RE.split(lowered)
        if token and len(token) > 2 and token not in _STOPWORDS
    }


def _cost_delta(before: dict[str, float | int], after: dict[str, float | int]) -> float:
    return float(after.get("total_cost_usd", 0.0) or 0.0) - float(before.get("total_cost_usd", 0.0) or 0.0)


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
    return candidate.replace(" ", "_") or fallback


def _resolve_output_dir(config: Any) -> Path:
    pipeline = getattr(config, "pipeline", None)
    output_cfg = getattr(pipeline, "output", None) if pipeline is not None else None
    if isinstance(output_cfg, dict):
        return Path(str(output_cfg.get("output_dir", "output")))
    return Path("output")


def _find_seed_text(document: Any) -> str:
    abstract_sections = [section for section in document.sections if section.role.value == "abstract"]
    if abstract_sections:
        return "\n\n".join(section.body_text for section in abstract_sections)
    intro_sections = [section for section in document.sections if section.role.value == "introduction"]
    if intro_sections:
        return "\n\n".join(section.body_text for section in intro_sections)
    if not document.sections:
        return ""
    return document.sections[0].body_text


def _candidate_to_raw(candidate: SectionArgumentCandidate) -> RawClaim:
    hints = None
    if candidate.elaborates_seed_id or candidate.local_role or candidate.preferred_parent_type:
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
        evidence=[{"artifact_id": evidence_id, "role": "supports"} for evidence_id in candidate.evidence_ids],
        entity_names=list(candidate.entity_names),
        rejected_what=candidate.rejected_what,
        rejected_why=candidate.rejected_why,
        structural_hints=hints,
        claim_strength=candidate.strength,
    )


def _seed_skeleton(seed_claims: list[RawClaim]) -> PaperSkeletonCandidate:
    return PaperSkeletonCandidate(
        context_roots=[claim for claim in seed_claims if claim.claim_type == ClaimType.context],
        core_methods=[claim for claim in seed_claims if claim.claim_type == ClaimType.method],
        topline_results=[claim for claim in seed_claims if claim.claim_type == ClaimType.result],
        assumptions=[claim for claim in seed_claims if claim.claim_type == ClaimType.assumption],
        negatives=[claim for claim in seed_claims if claim.claim_type == ClaimType.negative],
    )


def _claim_similarity(left: RawClaim | SupportDetail, right: RawClaim) -> float:
    left_text = left.statement if isinstance(left, RawClaim) else left.text
    left_tokens = _tokens(left_text)
    right_tokens = _tokens(right.statement)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    overlap = left_tokens & right_tokens
    score = len(overlap) / len(union)
    if isinstance(left, RawClaim):
        left_evidence = {pointer.artifact_id.strip().lower() for pointer in left.evidence if pointer.artifact_id.strip()}
    else:
        left_evidence = {item.strip().lower() for item in left.evidence_ids if item.strip()}
    right_evidence = {pointer.artifact_id.strip().lower() for pointer in right.evidence if pointer.artifact_id.strip()}
    if left_evidence and right_evidence and (left_evidence & right_evidence):
        score += 0.2
    if _clean_text(left.source_section).lower() == _clean_text(right.source_section).lower():
        score += 0.1
    return min(score, 1.0)


def _claim_to_support_detail(claim: RawClaim, index: int) -> SupportDetail:
    detail_type = classify_support_detail_type(claim)
    return SupportDetail(
        support_detail_id=f"SD_residual_{index}",
        detail_type=detail_type,
        text=claim.statement,
        source_section=claim.source_section,
        anchor_claim_id=None,
        candidate_anchor_ids=[],
        relationship_type=support_relationship_for_type(detail_type),
        confidence=min(1.0, max(0.2, (claim.claim_strength or 0.0) / 6.0 if claim.claim_strength is not None else 0.4)),
        evidence_ids=[pointer.artifact_id for pointer in claim.evidence if pointer.artifact_id.strip()],
    )


def _is_legal_anchor(detail: SupportDetail, claim: RawClaim) -> bool:
    if detail.detail_type == SupportDetailType.numeric_support:
        return claim.claim_type == ClaimType.result
    if detail.detail_type in {
        SupportDetailType.implementation_fact,
        SupportDetailType.procedural_step,
        SupportDetailType.local_kernel_optimization,
        SupportDetailType.api_surface,
        SupportDetailType.framework_dependency,
    }:
        return claim.claim_type == ClaimType.method
    return False


def _anchor_preference(detail: SupportDetail, claim: RawClaim) -> float:
    bonus = 0.0
    if detail.detail_type in {SupportDetailType.api_surface, SupportDetailType.framework_dependency}:
        if classify_method_abstraction(claim) == "system_realization":
            bonus += 0.15
    return bonus


def _attach_support_details(support_details: list[SupportDetail], claims: list[RawClaim]) -> list[SupportDetail]:
    if not support_details or not claims:
        return list(support_details)

    claim_by_id = {claim.claim_id: claim for claim in claims}
    remapped: list[SupportDetail] = []
    for detail in support_details:
        current_anchor = claim_by_id.get(detail.anchor_claim_id) if detail.anchor_claim_id else None
        if current_anchor is not None and _is_legal_anchor(detail, current_anchor):
            remapped.append(detail)
            continue
        legal_claims = [claim for claim in claims if _is_legal_anchor(detail, claim)]
        if not legal_claims:
            remapped.append(detail.model_copy(update={"anchor_claim_id": None, "candidate_anchor_ids": []}))
            continue
        scored = sorted(
            ((claim, _claim_similarity(detail, claim) + _anchor_preference(detail, claim)) for claim in legal_claims),
            key=lambda item: (item[1], _claim_similarity(detail, item[0])),
            reverse=True,
        )
        candidate_anchor_ids = [claim.claim_id for claim, score in scored[:3] if score >= 0.05]
        anchor_claim_id = candidate_anchor_ids[0] if candidate_anchor_ids else None
        confidence = detail.confidence
        if scored and anchor_claim_id is not None:
            confidence = round(max(detail.confidence, min(1.0, _claim_similarity(detail, scored[0][0]))), 3)
        remapped.append(
            detail.model_copy(
                update={
                    "anchor_claim_id": anchor_claim_id,
                    "candidate_anchor_ids": candidate_anchor_ids,
                    "confidence": confidence,
                }
            )
        )
    return remapped


def _concept_family_collision_count(claims: list[RawClaim]) -> int:
    seen: Counter[tuple[str, str]] = Counter()
    collisions = 0
    for claim in claims:
        if claim.claim_type == ClaimType.method:
            key = (claim.claim_type.value, classify_method_abstraction(claim))
        elif claim.claim_type == ClaimType.result:
            key = (claim.claim_type.value, classify_result_family(claim))
        else:
            key = (claim.claim_type.value, next(iter(sorted(_tokens(claim.statement))), claim.claim_type.value))
        seen[key] += 1
        if seen[key] > 1:
            collisions += 1
    return collisions


def _abstraction_tier_violations(claims: list[RawClaim]) -> int:
    methods = [claim for claim in claims if claim.claim_type == ClaimType.method]
    counts = Counter(classify_method_abstraction(claim) for claim in methods)
    violations = 0
    if counts.get("primitive", 0) > 1:
        violations += counts["primitive"] - 1
    if counts.get("system_realization", 0) > 1:
        violations += counts["system_realization"] - 1
    return violations


def _illegal_dependency_directions(nodes: list[ClaimNode]) -> int:
    node_by_id: dict[str, ClaimNode] = {}

    def walk(node: ClaimNode) -> None:
        node_by_id[node.claim_id] = node
        for child in node.children:
            walk(child)

    for node in nodes:
        walk(node)

    violations = 0
    for node in node_by_id.values():
        for dependency_id in node.depends_on:
            dependency = node_by_id.get(dependency_id)
            if dependency is None:
                violations += 1
                continue
            if node.claim_type == ClaimType.result and dependency.claim_type != ClaimType.method:
                violations += 1
            if node.claim_type == ClaimType.method and dependency.claim_type not in {ClaimType.context, ClaimType.method}:
                violations += 1
    return violations


def _scorecard(
    promoted_claims: list[RawClaim],
    support_details: list[SupportDetail],
    claim_tree: list[ClaimNode],
    compression_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    implementation_like = [
        detail for detail in support_details if detail.detail_type in {SupportDetailType.implementation_fact, SupportDetailType.local_kernel_optimization}
    ]
    promoted_by_type = dict(Counter(claim.claim_type.value for claim in promoted_claims))
    assumption_candidates = int(compression_diagnostics.get("assumption_candidates", 0) or 0)
    rejected_assumptions = int(compression_diagnostics.get("assumption_rejections_as_mechanism", 0) or 0)
    return {
        "promoted_nodes_by_type": promoted_by_type,
        "concept_family_collisions": _concept_family_collision_count(promoted_claims),
        "abstraction_tier_violations": _abstraction_tier_violations(promoted_claims),
        "illegal_dependency_directions": _illegal_dependency_directions(claim_tree),
        "implementation_detail_support_fraction": round(len(implementation_like) / max(1, len(support_details)), 3),
        "assumption_mechanism_rejection_fraction": round(rejected_assumptions / max(1, assumption_candidates), 3),
    }


def _iter_nodes(nodes: list[ClaimNode]):
    for node in nodes:
        yield node
        yield from _iter_nodes(node.children)


def _tree_stats(nodes: list[ClaimNode]) -> tuple[int, int]:
    def depth(node: ClaimNode) -> int:
        if not node.children:
            return 1
        return 1 + max(depth(child) for child in node.children)

    all_nodes = list(_iter_nodes(nodes))
    return len(all_nodes), max((depth(node) for node in nodes), default=0)


async def _run_parallel_with_progress(
    items: list[T],
    *,
    progress: Progress,
    description: str,
    limit: int,
    worker: Callable[[T], Awaitable[Any]],
) -> list[Any]:
    if not items:
        return []
    semaphore = asyncio.Semaphore(max(1, limit))
    task_id = progress.add_task(description, total=len(items))

    async def wrapped(item: T) -> Any:
        async with semaphore:
            try:
                return await worker(item)
            finally:
                progress.advance(task_id)

    return await asyncio.gather(*(wrapped(item) for item in items), return_exceptions=True)


async def decompose_paper(pdf_path: str, config_path: str = "config.yaml") -> PaperDecomposition:
    reset_cost_tracker()
    config = load_config(config_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        parse_task = progress.add_task("Phase 0/4: Parse PDF", total=1)
        document = parse_pdf(pdf_path, config)
        progress.advance(parse_task)

        preflight_task = progress.add_task("Phase 0b/4: Model preflight", total=1)
        await preflight_model_tiers(["small"], config=config)
        progress.advance(preflight_task)

        phase1_before = get_cost_tracker()
        seed_candidates: list[RawClaim] = []
        seed_text = _find_seed_text(document)
        if seed_text.strip():
            try:
                seed_output = await extract_seed(seed_text, config)
                seed_candidates = list(seed_output.claims)
            except Exception as exc:
                console.print(f"[yellow]Seed candidate extraction failed:[/yellow] {exc}")
        seed_skeleton = _seed_skeleton(seed_candidates)

        section_inputs = [
            section
            for section in document.sections
            if section.role.value != "abstract" and not _is_reference_section(section)
        ]

        async def section_worker(section: Section) -> Any:
            section_artifacts = list(section.artifacts) if section.artifacts else list(document.all_artifacts)
            return await extract_section_digest(
                section=section,
                skeleton=seed_skeleton,
                artifacts=section_artifacts,
                config=config,
            )

        section_results = await _run_parallel_with_progress(
            section_inputs,
            progress=progress,
            description="Phase 1/4: Candidate extraction",
            limit=4,
            worker=section_worker,
        )

        argument_candidates: list[RawClaim] = list(seed_candidates)
        support_details: list[SupportDetail] = []
        for section_obj, result in zip(section_inputs, section_results, strict=False):
            if isinstance(result, Exception):
                console.print(f"[red]Section extraction failed[/red] ({_section_label(section_obj)}): {result}")
                continue
            argument_candidates.extend(_candidate_to_raw(candidate) for candidate in result.argument_candidates)
            support_details.extend(result.support_details)

        if not argument_candidates:
            raise RuntimeError("Phase 1 produced zero candidates; aborting run.")

        phase1_after = get_cost_tracker()
        console.print(
            "Phase 1 candidates: "
            f"seed={len(seed_candidates)} section={len(argument_candidates) - len(seed_candidates)} support={len(support_details)} "
            f"| cost +${_cost_delta(phase1_before, phase1_after):.4f}"
        )

        phase2_before = get_cost_tracker()
        compression = compress_claims_to_skeleton(argument_candidates, config)
        promoted_claims = list(compression.promoted_claims)
        residual_support = [_claim_to_support_detail(claim, idx) for idx, claim in enumerate(compression.residual_claims, start=1)]
        phase2_after = get_cost_tracker()
        console.print(
            "Phase 2 skeleton: "
            f"promoted={len(promoted_claims)} residual={len(compression.residual_claims)} "
            f"| cost +${_cost_delta(phase2_before, phase2_after):.4f}"
        )

        phase3_before = get_cost_tracker()
        attached_support_details = _attach_support_details([*support_details, *residual_support], promoted_claims)
        phase3_after = get_cost_tracker()
        console.print(
            "Phase 3 support routing: "
            f"support_details={len(attached_support_details)} | cost +${_cost_delta(phase3_before, phase3_after):.4f}"
        )

        phase4_before = get_cost_tracker()
        method_claims = [claim for claim in promoted_claims if claim.claim_type == ClaimType.method]

        async def facet_worker(claim: RawClaim) -> FacetedClaim:
            matching_section = next(
                (section for section in document.sections if _clean_text(_section_label(section)).lower() == _clean_text(claim.source_section).lower()),
                claim.source_section,
            )
            return await extract_facets(claim, matching_section, config)

        facet_results = await _run_parallel_with_progress(
            method_claims,
            progress=progress,
            description="Phase 4/4: Facets + tree",
            limit=4,
            worker=facet_worker,
        )
        faceted_claims = [result for result in facet_results if not isinstance(result, Exception)]
        for claim, result in zip(method_claims, facet_results, strict=False):
            if isinstance(result, Exception):
                console.print(f"[yellow]Facet extraction failed[/yellow] ({claim.claim_id}): {result}")

        decomposition = await assemble_tree_deterministic(
            metadata=document.metadata,
            claims=promoted_claims,
            faceted=faceted_claims,
            negatives=[claim for claim in promoted_claims if claim.claim_type == ClaimType.negative],
            artifacts=document.all_artifacts,
            config=config,
            claim_groups=compression.claim_groups,
            support_details=attached_support_details,
        )
        total_cost = float(get_cost_tracker().get("total_cost_usd", 0.0))
        decomposition = decomposition.model_copy(update={"extraction_cost_usd": total_cost})
        phase4_after = get_cost_tracker()

        node_count, depth = _tree_stats(decomposition.claim_tree)
        scorecard = _scorecard(promoted_claims, decomposition.support_details, decomposition.claim_tree, compression.diagnostics)
        console.print(
            f"Phase 4 tree nodes={node_count} depth={depth} | cost +${_cost_delta(phase4_before, phase4_after):.4f}"
        )
        console.print(
            "Scorecard: "
            f"promoted={scorecard['promoted_nodes_by_type']} "
            f"family_collisions={scorecard['concept_family_collisions']} "
            f"tier_violations={scorecard['abstraction_tier_violations']} "
            f"illegal_deps={scorecard['illegal_dependency_directions']} "
            f"impl_support_frac={scorecard['implementation_detail_support_fraction']} "
            f"assumption_reject_frac={scorecard['assumption_mechanism_rejection_fraction']}"
        )

    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_filename(document.metadata.title or Path(pdf_path).stem, Path(pdf_path).stem)
    output_path = output_dir / f"{stem}.json"
    suffix = 1
    while output_path.exists():
        suffix += 1
        output_path = output_dir / f"{stem}_{suffix}.json"
    output_path.write_text(decomposition.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"Wrote decomposition to {output_path}")
    return decomposition


__all__ = ["decompose_paper", "_attach_support_details", "_scorecard"]
