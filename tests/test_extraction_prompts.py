"""
Claim:
Default extraction prompts are short, field-general contracts for paper-local
graph extraction with slugged IDs, explicit outcome rows, and attached claims.

Plausible wrong implementations:
- Reintroduce vLLM-specific calibration in the default prompt.
- Allow opaque numeric local IDs instead of slugged typed IDs.
- Keep metric/value/delta/baseline/comparator fields on claims.
- Let outcome-linked claims omit responsible methods.
- Allow noisy component/example/formula evidence as claim evidence.
- Ask for final candidates instead of promoted graph fields.
"""

from __future__ import annotations

from paper_decomposer.extraction.contracts import EvidenceSpan, ExtractionCaps
from paper_decomposer.extraction.prompts import (
    big_model_compact_prompt,
    cleanup_prompt,
    compression_prompt,
    method_graph_prompt,
)


def _span() -> EvidenceSpan:
    return EvidenceSpan(
        span_id="s1",
        paper_id="paper-1",
        section_title="PagedAttention",
        section_kind="method",
        text="PagedAttention stores KV cache in non-contiguous fixed-size blocks.",
    )


def test_default_method_graph_prompt_is_generic_open_world() -> None:
    content = method_graph_prompt([_span()], "{}")[0]["content"]
    user_content = method_graph_prompt([_span()], "{}")[1]["content"]

    assert "paper-local objects only" in content
    assert "do not assign global identity" in content
    assert "typed slug IDs" in content
    assert "local:method:*" in content
    assert "Baseline systems are reference system nodes" in content
    assert "Claims are compact propositions" in content
    assert "Put measurements in outcomes" in content
    assert "category_tags" in content
    assert "component labels, examples, formula fragments" in content
    assert "vLLM is the system" not in content
    assert "PagedAttention is the central primitive" not in content
    assert "Do not extract claims yet" in user_content


def test_compression_prompt_preserves_promoted_graph_without_candidates() -> None:
    messages = compression_prompt("{}", "{}")
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]

    assert "problems go in problems" in system_content
    assert "Outcomes are measurement rows" in system_content
    assert "graph.systems" in user_content
    assert "problems" in user_content
    assert "Do not include final candidates" in user_content
    assert "Every method needs mechanism_sentence" in user_content
    assert "metric/value/baseline/comparator fields" not in cleanup_prompt("{}", [], [])[-1]["content"]


def test_big_model_prompt_enforces_compact_claims_and_resolved_ids() -> None:
    messages = big_model_compact_prompt([_span()], ExtractionCaps(max_system_nodes=1, max_method_nodes=10, max_claims=8))
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]

    assert "One-pass extraction" in system_content
    assert "compact claims" in system_content
    assert "separate outcome rows" in system_content
    assert "local:method:*" in system_content
    assert "mechanism_sentence" in system_content
    assert "do not add validation_notes" in user_content
    assert "system nodes <= 1" in user_content
    assert "method nodes <= 10" in user_content
    assert "compact claims <= 8" in user_content
    assert "every referenced outcome_id exists exactly once" in user_content
    assert "claims with outcomes also cite responsible methods" in user_content
    assert "[s1]" in user_content
