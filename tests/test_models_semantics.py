"""
Claim:
`call_model` must use the selected model tier's parameters, send Together's
structured-output format when a schema is requested, retry with exponential
backoff according to API config, and accumulate token costs in USD at per-tier
per-million rates.

Plausible wrong implementations:
- Read model settings from the wrong tier, or from the wrong config object.
- Send OpenAI-style structured output fields instead of Together's
  `{"type": "json_object", "schema": ...}` contract.
- Retry the wrong number of times or with linear/no backoff.
- Compute costs with the wrong denominator (e.g., per-1k instead of per-1M).
- Use non-priced runtime tiers (AppSettings.model_tiers) for cost accounting
  instead of priced raw config (AppSettings.raw.models).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

import paper_decomposer.models as model_client
from paper_decomposer.schema import (
    AppSettings,
    PaperDecomposerConfig,
    RuntimeModelConfig,
    RuntimePipelineConfig,
)


class AnswerOutput(BaseModel):
    answer: int


class FlatClaim(BaseModel):
    claim_id: str
    claim_type: str
    statement: str
    evidence_artifact_ids: list[str] = []


class FlatClaimsOutput(BaseModel):
    claims: list[FlatClaim] = []


class _FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _fake_response(content: str, prompt_tokens: int, completion_tokens: int) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _test_config(*, max_retries: int = 2, retry_backoff_base: float = 2.0) -> dict[str, Any]:
    return {
        "api": {
            "max_retries": max_retries,
            "retry_backoff_base": retry_backoff_base,
        },
        "models": {
            "small": {
                "model": "small-model",
                "temperature": 0.1,
                "max_tokens": 128,
                "input_cost_per_m": 1.0,
                "output_cost_per_m": 2.0,
            },
            "medium": {
                "model": "medium-model",
                "temperature": 0.2,
                "max_tokens": 256,
                "input_cost_per_m": 10.0,
                "output_cost_per_m": 20.0,
            },
            "heavy": {
                "model": "heavy-model",
                "temperature": 0.3,
                "max_tokens": 512,
                "input_cost_per_m": 30.0,
                "output_cost_per_m": 40.0,
            },
        },
    }


def _app_settings_with_priced_raw_config() -> AppSettings:
    raw = PaperDecomposerConfig.model_validate(
        {
            "api": {
                "provider": "together",
                "base_url": "https://api.together.xyz/v1",
                "max_retries": 1,
                "retry_backoff_base": 2.0,
            },
            "models": {
                "small": {
                    "model": "small-priced",
                    "temperature": 0.11,
                    "max_tokens": 100,
                    "input_cost_per_m": 5.0,
                    "output_cost_per_m": 7.0,
                },
                "medium": {
                    "model": "medium-priced",
                    "temperature": 0.22,
                    "max_tokens": 200,
                    "input_cost_per_m": 9.0,
                    "output_cost_per_m": 11.0,
                },
                "heavy": {
                    "model": "heavy-priced",
                    "temperature": 0.33,
                    "max_tokens": 300,
                    "input_cost_per_m": 13.0,
                    "output_cost_per_m": 17.0,
                },
            },
            "pipeline": {
                "pdf": {
                    "min_section_chars": 1,
                    "max_section_chars": 10,
                },
                "extraction": {},
            },
        }
    )

    return AppSettings(
        config_path="in-memory.yaml",
        api_key="test-key",
        model_tiers={
            "small": RuntimeModelConfig(model="small-runtime", temperature=0.01, max_tokens=11),
            "medium": RuntimeModelConfig(model="medium-runtime", temperature=0.02, max_tokens=22),
            "heavy": RuntimeModelConfig(model="heavy-runtime", temperature=0.03, max_tokens=33),
        },
        pipeline=RuntimePipelineConfig(
            parser="pymupdf",
            extract_captions=False,
            extract_equations=False,
            min_section_chars=1,
            max_section_chars=10,
        ),
        raw=raw,
    )


@pytest.fixture(autouse=True)
def _reset_cost_tracker() -> None:
    model_client.reset_cost_tracker()
    yield
    model_client.reset_cost_tracker()


def test_call_model_uses_selected_tier_kwargs_and_returns_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config()
    completions = _FakeCompletions([_fake_response("plain-response", 10, 5)])
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    result = asyncio.run(
        model_client.call_model(
            "medium",
            [{"role": "user", "content": "hello"}],
            config=config,
        )
    )

    assert result == "plain-response"
    sent = completions.calls[0]
    assert sent["model"] == "medium-model"
    assert sent["temperature"] == pytest.approx(0.2)
    assert sent["max_tokens"] == 256
    assert "response_format" not in sent


def test_model_client_uses_configured_api_key_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(model_client, "AsyncOpenAI", FakeOpenAI)
    monkeypatch.setattr(model_client, "_client", None)
    monkeypatch.setattr(model_client, "_client_key", None)
    monkeypatch.setattr(model_client, "_requested_client_key", ("configured-key", "https://example.test/v1"))

    client = model_client._get_client()

    assert isinstance(client, FakeOpenAI)
    assert captured == {"api_key": "configured-key", "base_url": "https://example.test/v1"}


def test_call_model_structured_output_uses_together_schema_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config()
    completions = _FakeCompletions([_fake_response('{"answer": 4}', 12, 7)])
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    result = asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "2+2"}],
            response_schema=AnswerOutput,
            config=config,
        )
    )

    assert isinstance(result, AnswerOutput)
    assert result.answer == 4

    sent = completions.calls[0]
    assert "response_format" in sent
    assert sent["response_format"]["type"] == "json_object"
    assert "schema" in sent["response_format"]
    assert "answer" in sent["response_format"]["schema"]["properties"]


def test_call_model_retries_with_exponential_backoff_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(max_retries=3, retry_backoff_base=3.0)
    completions = _FakeCompletions(
        [
            RuntimeError("first transient failure"),
            RuntimeError("second transient failure"),
            _fake_response("ok-after-retries", 8, 6),
        ]
    )
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    waits: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(model_client.asyncio, "sleep", _fake_sleep)

    result = asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "retry"}],
            config=config,
        )
    )

    assert result == "ok-after-retries"
    assert len(completions.calls) == 3
    assert waits == [1.0, 3.0]


def test_call_model_stops_after_max_retries_and_re_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(max_retries=2, retry_backoff_base=2.0)
    completions = _FakeCompletions(
        [
            RuntimeError("temporary-1"),
            RuntimeError("temporary-2"),
            RuntimeError("final-error"),
        ]
    )
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    waits: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(model_client.asyncio, "sleep", _fake_sleep)

    with pytest.raises(RuntimeError, match="final-error"):
        asyncio.run(
            model_client.call_model(
                "small",
                [{"role": "user", "content": "will-fail"}],
                config=config,
            )
        )

    assert len(completions.calls) == 3
    assert waits == [1.0, 2.0]


def test_cost_tracker_accumulates_across_calls_and_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config()
    completions = _FakeCompletions(
        [
            _fake_response("first", prompt_tokens=1000, completion_tokens=500),
            _fake_response("second", prompt_tokens=100, completion_tokens=10),
        ]
    )
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "first"}],
            config=config,
        )
    )
    asyncio.run(
        model_client.call_model(
            "medium",
            [{"role": "user", "content": "second"}],
            config=config,
        )
    )

    tracker = model_client.get_cost_tracker()
    assert tracker["total_calls"] == 2
    assert tracker["prompt_tokens"] == 1100
    assert tracker["completion_tokens"] == 510
    assert tracker["input_cost_usd"] == pytest.approx(0.002)
    assert tracker["output_cost_usd"] == pytest.approx(0.0012)
    assert tracker["total_cost_usd"] == pytest.approx(0.0032)


def test_app_settings_input_uses_raw_priced_config_for_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_settings = _app_settings_with_priced_raw_config()
    completions = _FakeCompletions([_fake_response("priced", prompt_tokens=100, completion_tokens=200)])
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "price-check"}],
            config=app_settings,
        )
    )

    tracker = model_client.get_cost_tracker()
    expected_input = 100 * 5.0 / 1_000_000
    expected_output = 200 * 7.0 / 1_000_000
    assert tracker["input_cost_usd"] == pytest.approx(expected_input)
    assert tracker["output_cost_usd"] == pytest.approx(expected_output)
    assert tracker["total_cost_usd"] == pytest.approx(expected_input + expected_output)


def test_call_model_structured_output_accepts_fenced_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config()
    completions = _FakeCompletions([_fake_response("```json\n{\"answer\": 4}\n```", 12, 7)])
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    result = asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "2+2"}],
            response_schema=AnswerOutput,
            config=config,
        )
    )

    assert isinstance(result, AnswerOutput)
    assert result.answer == 4


def test_call_model_structured_output_accepts_embedded_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config()
    content = "Here is the structured answer:\n{\"answer\": 4}\nThanks."
    completions = _FakeCompletions([_fake_response(content, 12, 7)])
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    result = asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "2+2"}],
            response_schema=AnswerOutput,
            config=config,
        )
    )

    assert isinstance(result, AnswerOutput)
    assert result.answer == 4


def test_call_model_structured_output_uses_valid_json_candidate_after_invalid_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(max_retries=0)
    content = (
        'Scratch: {"not_answer": 1}\n'
        "```json\n"
        '{"answer": 4}\n'
        "```"
    )
    completions = _FakeCompletions([_fake_response(content, 12, 7)])
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    result = asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "2+2"}],
            response_schema=AnswerOutput,
            config=config,
        )
    )

    assert isinstance(result, AnswerOutput)
    assert result.answer == 4
    assert len(completions.calls) == 1


class ClaimsListOutput(BaseModel):
    claims: list[int] = []


def test_call_model_structured_output_wraps_top_level_list_for_single_field_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(max_retries=0)
    completions = _FakeCompletions([_fake_response("[1,2,3]", 12, 7)])
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    result = asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "return list"}],
            response_schema=ClaimsListOutput,
            config=config,
        )
    )

    assert isinstance(result, ClaimsListOutput)
    assert result.claims == [1, 2, 3]


def test_call_model_structured_output_raises_clear_error_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(max_retries=1, retry_backoff_base=2.0)
    completions = _FakeCompletions(
        [
            _fake_response("not-json", 12, 7),
            _fake_response("still-not-json", 12, 7),
        ]
    )
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    with pytest.raises(ValueError, match="Structured response was not valid JSON"):
        asyncio.run(
            model_client.call_model(
                "small",
                [{"role": "user", "content": "2+2"}],
                response_schema=AnswerOutput,
                config=config,
            )
        )


def test_call_model_retries_on_empty_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(max_retries=2, retry_backoff_base=2.0)
    completions = _FakeCompletions(
        [
            _fake_response("   ", 12, 7),
            _fake_response('{"answer": 4}', 12, 7),
        ]
    )
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    waits: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(model_client.asyncio, "sleep", _fake_sleep)

    result = asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "2+2"}],
            response_schema=AnswerOutput,
            config=config,
        )
    )

    assert isinstance(result, AnswerOutput)
    assert result.answer == 4
    assert len(completions.calls) == 2
    assert waits == [1.0]


def test_call_model_retries_on_invalid_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(max_retries=2, retry_backoff_base=2.0)
    completions = _FakeCompletions(
        [
            _fake_response("not-json", 12, 7),
            _fake_response('{"answer": 4}', 12, 7),
        ]
    )
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    waits: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(model_client.asyncio, "sleep", _fake_sleep)

    result = asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "2+2"}],
            response_schema=AnswerOutput,
            config=config,
        )
    )

    assert isinstance(result, AnswerOutput)
    assert result.answer == 4
    assert len(completions.calls) == 2
    assert waits == [1.0]


def test_call_model_structured_retry_appends_repair_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(max_retries=2, retry_backoff_base=2.0)
    completions = _FakeCompletions(
        [
            _fake_response("not-json", 12, 7),
            _fake_response('{"answer": 4}', 12, 7),
        ]
    )
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    result = asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "2+2"}],
            response_schema=AnswerOutput,
            config=config,
        )
    )

    assert isinstance(result, AnswerOutput)
    assert result.answer == 4
    assert len(completions.calls) == 2
    repair_prompt = (
        "Respond ONLY with valid JSON matching the schema. "
        "No preamble, no markdown fences."
    )
    second_call_messages = completions.calls[1]["messages"]
    assert isinstance(second_call_messages, list)
    assert repair_prompt in second_call_messages[-1]["content"]


def test_call_model_structured_output_parses_flat_claim_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config()
    completions = _FakeCompletions(
        [
            _fake_response(
                '{"claims":[{"claim_id":"s1","claim_type":"method","statement":"Uses block table.","evidence_artifact_ids":["Fig. 1"]}]}',
                20,
                10,
            )
        ]
    )
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    result = asyncio.run(
        model_client.call_model(
            "small",
            [{"role": "user", "content": "extract claims"}],
            response_schema=FlatClaimsOutput,
            config=config,
        )
    )

    assert isinstance(result, FlatClaimsOutput)
    assert len(result.claims) == 1
    assert result.claims[0].claim_id == "s1"
    assert result.claims[0].evidence_artifact_ids == ["Fig. 1"]


def test_preflight_model_tiers_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _test_config(max_retries=0, retry_backoff_base=2.0)
    completions = _FakeCompletions(
        [
            _fake_response('{"ok": true}', 4, 2),
            RuntimeError("medium tier down"),
        ]
    )
    monkeypatch.setattr(model_client, "_get_client", lambda: _FakeClient(completions))

    with pytest.raises(model_client.ModelPreflightError, match="medium"):
        asyncio.run(
            model_client.preflight_model_tiers(
                ["small", "medium"],
                config=config,
            )
        )
