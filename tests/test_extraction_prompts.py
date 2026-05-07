"""
Claim:
Default extraction prompts are field-general and enforce open-world paper-local
discovery without vLLM fixture calibration.

Plausible wrong implementations:
- Leave vLLM-specific calibration in the default prompt.
- Ask the model to assign global identity during extraction.
- Keep row-level metric/value fields on claims.
- Treat problems as settings instead of top-level problem objects.
- Ask for final candidates instead of promoted graph fields.
- Omit semantic signatures needed by later entity resolution.
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

    assert "open-world paper-local discovery" in content
    assert "do not assign global identity" in content
    assert "Baseline systems are reference system nodes" in content
    assert "Claims are propositions" in content
    assert "category_tags" in content
    assert "vLLM is the system" not in content
    assert "PagedAttention is the central primitive" not in content
    assert "Do not extract claims yet" in user_content


def test_compression_prompt_preserves_promoted_graph_without_candidates() -> None:
    messages = compression_prompt("{}", "{}")
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]

    assert "Problem/challenge context belongs in top-level problems" in system_content
    assert "Outcomes are measurements" in system_content
    assert "graph.systems" in user_content
    assert "problems" in user_content
    assert "Do not include final candidates" in user_content
    assert "metric/value/baseline/comparator fields" not in cleanup_prompt("{}", [], [])[-1]["content"]


def test_big_model_prompt_enforces_compact_claims_and_resolved_ids() -> None:
    messages = big_model_compact_prompt([_span()], ExtractionCaps(max_system_nodes=1, max_method_nodes=10, max_claims=8))
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]

    assert "Single-pass compact extraction" in system_content
    assert "Keep claims compact" in system_content
    assert "Outcomes carry numeric rows" in system_content
    assert "local:method:*" in system_content
    assert "mechanism_signature" in system_content
    assert "Do not emit validation notes" in system_content
    assert "system nodes <= 1" in user_content
    assert "method nodes <= 10" in user_content
    assert "compact claims <= 8" in user_content
    assert "every referenced outcome_id exists" in user_content
    assert "[s1]" in user_content
