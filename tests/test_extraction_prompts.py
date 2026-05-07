"""
Claim:
Extraction prompts calibrate models toward reusable mechanism DAGs instead of
paper outlines, implementation details, or application-scenario nodes.

Plausible wrong implementations:
- Omit the system-to-primitive graph shape, encouraging flat node bags.
- Promote decoding scenarios such as beam search as methods in vLLM-like papers.
- Promote implementation support such as fused kernels or APIs as first-class methods.
- Demote concrete KV-cache mechanisms when only the mechanism sentence is missing.
- Attach system-level throughput claims to the wrong supporting mechanism.
- Ask for final candidates instead of promoted graph fields.
"""

from __future__ import annotations

from paper_decomposer.extraction.contracts import EvidenceSpan
from paper_decomposer.extraction.prompts import compression_prompt, method_graph_prompt


def _span() -> EvidenceSpan:
    return EvidenceSpan(
        span_id="s1",
        paper_id="paper-1",
        section_title="PagedAttention",
        section_kind="method",
        text="PagedAttention stores KV cache in non-contiguous fixed-size blocks.",
    )


def test_method_graph_prompt_calibrates_vllm_mechanism_granularity() -> None:
    content = method_graph_prompt([_span()], "{}")[0]["content"]
    user_content = method_graph_prompt([_span()], "{}")[1]["content"]

    assert "graph-first" in content
    assert "vLLM is the system" in content
    assert "PagedAttention is the central primitive" in content
    assert "block-wise KV cache address translation" in content
    assert "on-demand KV block allocation" in content
    assert "KV block copy-on-write" in content
    assert "KV-cache recomputation" in content
    assert "Do not demote those concrete mechanisms" in content
    assert "Do not extract claims yet" in user_content


def test_prompts_demote_applications_and_implementation_support() -> None:
    messages = compression_prompt("{}", "{}")
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]

    assert "beam search" in system_content
    assert "applications/settings or claim contexts" in system_content
    assert "fused kernels" in system_content
    assert "fork/append/free APIs" in system_content
    assert "Attach composed throughput claims to the system" in system_content
    assert "fill metric, delta/value, baseline, and comparator" in system_content
    assert "graph.systems" in user_content
    assert "Do not include final candidates" in user_content
