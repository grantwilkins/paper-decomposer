from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..models import call_model_with_fallback, flat_claim_to_raw
from ..schema import FlatSeedOutput, ModelTier, RawClaim, SeedOutput

SEED_SYSTEM_PROMPT = """You are a research paper analyst. You extract
the claim skeleton from a paper's abstract.

A claim is a single assertive proposition that the paper asks the
reader to accept. Claims are typed:
- CONTEXT: a problem, gap, or constraint in the current landscape
- METHOD: something the authors built, designed, or formalized
- RESULT: something the authors demonstrated or measured
- ASSUMPTION: a premise the argument requires

Rules:
- Extract 3-7 claims. Most abstracts have exactly this many.
- Each claim is ONE proposition. Split compound sentences.
- Do NOT extract background facts any reader would already know.
- DO extract the specific gap/problem the paper addresses.
- For results, preserve quantitative specifics (numbers, comparisons).
- For methods, name the method/system if the authors name it.
"""

_VALID_TIERS: set[ModelTier] = {"small", "medium", "heavy"}


def _resolve_model_tier(config: Any) -> ModelTier:
    if config is None:
        return "small"

    pipeline = getattr(config, "pipeline", None)
    if pipeline is not None:
        seed_cfg = getattr(pipeline, "seed", None)
        if isinstance(seed_cfg, Mapping):
            tier = seed_cfg.get("model_tier")
            if isinstance(tier, str) and tier in _VALID_TIERS:
                return cast(ModelTier, tier)

    if isinstance(config, Mapping):
        pipeline_cfg = config.get("pipeline")
        if isinstance(pipeline_cfg, Mapping):
            seed_cfg = pipeline_cfg.get("seed")
            if isinstance(seed_cfg, Mapping):
                tier = seed_cfg.get("model_tier")
                if isinstance(tier, str) and tier in _VALID_TIERS:
                    return cast(ModelTier, tier)

    return "small"


def build_seed_prompt(abstract_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SEED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract the claim skeleton from this abstract.\n\n"
                f"ABSTRACT:\n{abstract_text}"
            ),
        },
    ]


def _seed_tier_candidates(primary: ModelTier) -> list[ModelTier]:
    return [primary]


async def extract_seed(abstract_text: str, config: Any) -> SeedOutput:
    messages = build_seed_prompt(abstract_text)
    primary_tier = _resolve_model_tier(config)
    last_error: Exception | None = None

    for tier in _seed_tier_candidates(primary_tier):
        try:
            result = await call_model_with_fallback(
                tier=tier,
                messages=messages,
                response_schema=FlatSeedOutput,
                config=config,
            )
            if not isinstance(result, FlatSeedOutput):
                raise TypeError("Expected FlatSeedOutput from structured model call.")
            claims: list[RawClaim] = []
            for claim in result.claims:
                try:
                    claims.append(flat_claim_to_raw(claim, fallback_section="abstract"))
                except Exception:
                    continue
            return SeedOutput(claims=claims)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Seed extraction failed without an error.")


__all__ = ["SEED_SYSTEM_PROMPT", "build_seed_prompt", "extract_seed"]
