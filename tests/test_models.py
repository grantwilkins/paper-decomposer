from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from paper_decomposer.config import load_config
from paper_decomposer.models import call_model, get_cost_tracker, preflight_model_tiers, reset_cost_tracker

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

requires_api_key = pytest.mark.skipif(
    not os.getenv("TOGETHER_API_KEY"),
    reason="TOGETHER_API_KEY is not set.",
)


class SimpleOutput(BaseModel):
    answer: int


@pytest.mark.api
@requires_api_key
def test_call_model_plain_text_small_tier() -> None:
    settings = load_config(CONFIG_PATH)
    reset_cost_tracker()

    response = asyncio.run(
        call_model(
            "small",
            [
                {
                    "role": "user",
                    "content": "What is 2+2? Respond with only the number.",
                }
            ],
            config=settings,
        )
    )

    assert isinstance(response, str)
    assert "4" in response


@pytest.mark.api
@requires_api_key
def test_call_model_structured_output() -> None:
    settings = load_config(CONFIG_PATH)

    response = asyncio.run(
        call_model(
            "small",
            [
                {
                    "role": "user",
                    "content": "Return a JSON object with one key `answer` for 2+2.",
                }
            ],
            response_schema=SimpleOutput,
            config=settings,
        )
    )

    assert isinstance(response, SimpleOutput)
    assert response.answer == 4


@pytest.mark.api
@requires_api_key
def test_cost_tracker_accumulates() -> None:
    settings = load_config(CONFIG_PATH)
    reset_cost_tracker()

    first = asyncio.run(
        call_model(
            "small",
            [{"role": "user", "content": "What is 2+2? Respond with only the number."}],
            config=settings,
        )
    )
    assert isinstance(first, str)

    first_costs = get_cost_tracker()
    assert first_costs["total_calls"] == 1
    assert first_costs["prompt_tokens"] > 0
    assert first_costs["completion_tokens"] > 0

    second = asyncio.run(
        call_model(
            "small",
            [{"role": "user", "content": "What is 3+3? Respond with only the number."}],
            config=settings,
        )
    )
    assert isinstance(second, str)

    second_costs = get_cost_tracker()
    assert second_costs["total_calls"] == 2
    assert second_costs["prompt_tokens"] >= first_costs["prompt_tokens"]
    assert second_costs["completion_tokens"] >= first_costs["completion_tokens"]
    assert second_costs["input_cost_usd"] >= first_costs["input_cost_usd"]
    assert second_costs["output_cost_usd"] >= first_costs["output_cost_usd"]
    assert second_costs["total_cost_usd"] >= first_costs["total_cost_usd"]


@pytest.mark.api
@requires_api_key
def test_preflight_model_tiers_smoke() -> None:
    settings = load_config(CONFIG_PATH)
    asyncio.run(preflight_model_tiers(["small"], config=settings))
