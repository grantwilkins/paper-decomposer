from __future__ import annotations

from .contracts import LocalEntityResolutionTask, PaperExtraction


def build_local_entity_resolution_tasks(extraction: PaperExtraction) -> list[LocalEntityResolutionTask]:
    """Build paper-local entity tasks for a later global resolver."""
    tasks: list[LocalEntityResolutionTask] = []

    for node in extraction.graph.systems:
        tasks.append(
            LocalEntityResolutionTask(
                local_entity_id=node.local_node_id,
                entity_kind="system",
                canonical_name=node.canonical_name,
                aliases=node.aliases,
                description=node.description,
                problem_ids=node.problem_ids,
                mechanism_signature=node.mechanism_signature,
                evidence_span_ids=node.evidence_span_ids,
            )
        )

    for node in extraction.graph.methods:
        if node.kind == "method_category":
            continue
        tasks.append(
            LocalEntityResolutionTask(
                local_entity_id=node.local_node_id,
                entity_kind="method",
                canonical_name=node.canonical_name,
                aliases=node.aliases,
                description=node.description,
                problem_ids=node.problem_ids,
                mechanism_signature=node.mechanism_signature,
                evidence_span_ids=node.evidence_span_ids,
            )
        )

    for setting in extraction.graph.settings:
        tasks.append(
            LocalEntityResolutionTask(
                local_entity_id=setting.local_setting_id,
                entity_kind="setting",
                canonical_name=setting.canonical_name,
                aliases=setting.aliases,
                description=setting.description,
                evidence_span_ids=setting.evidence_span_ids,
            )
        )

    for problem in extraction.problems:
        tasks.append(
            LocalEntityResolutionTask(
                local_entity_id=problem.problem_id,
                entity_kind="problem",
                canonical_name=problem.statement,
                description=problem.description,
                evidence_span_ids=problem.evidence_span_ids,
            )
        )

    return tasks


__all__ = ["build_local_entity_resolution_tasks"]
