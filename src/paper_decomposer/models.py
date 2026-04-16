from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Mapping
from typing import Any, TypeVar, get_args, get_origin

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from rich.console import Console

from .config import get_config
from .schema import (
    AppSettings,
    ClaimLocalRole,
    ClaimStructuralHints,
    ClaimType,
    EvidencePointer,
    FlatClaim,
    ModelTier,
    ParentPreference,
    RawClaim,
)

_TOGETHER_BASE_URL = "https://api.together.xyz/v1"
_JSON_REPAIR_SUFFIX = "Respond ONLY with valid JSON matching the schema. No preamble, no markdown fences."
_NONE_LIKE_TOKENS = {"", "none", "n/a", "na", "null", "nil", "new", "unknown", "unspecified"}

_client: AsyncOpenAI | None = None

_COST_TRACKER: dict[str, float | int] = {
    "total_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "input_cost_usd": 0.0,
    "output_cost_usd": 0.0,
    "total_cost_usd": 0.0,
}

TResponseModel = TypeVar("TResponseModel", bound=BaseModel)
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_JSON_WRAPPER_KEYS = ("json", "data", "output", "result")
console = Console()


class ModelPreflightError(RuntimeError):
    pass


class _PreflightStructuredResponse(BaseModel):
    ok: bool


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url=_TOGETHER_BASE_URL,
        )
    return _client


def _resolve_config(config: Any | None) -> Any:
    resolved = config if config is not None else get_config()
    if isinstance(resolved, AppSettings):
        return resolved.raw
    return resolved


def _get_value(container: Any, key: str) -> Any:
    if isinstance(container, Mapping):
        return container[key]
    return getattr(container, key)


def _get_value_or_default(container: Any, key: str, default: Any) -> Any:
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _get_model_cfg(config: Any, tier: ModelTier) -> Any:
    models = _get_value(config, "models")
    return _get_value(models, tier)


def _get_api_cfg(config: Any) -> Any:
    return _get_value(config, "api")


def reset_cost_tracker() -> None:
    _COST_TRACKER.update(
        {
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }
    )


def get_cost_tracker() -> dict[str, float | int]:
    return dict(_COST_TRACKER)


def _track_cost(response: Any, model_cfg: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    input_cost_per_m = float(_get_value_or_default(model_cfg, "input_cost_per_m", 0.0) or 0.0)
    output_cost_per_m = float(_get_value_or_default(model_cfg, "output_cost_per_m", 0.0) or 0.0)

    input_cost = prompt_tokens * input_cost_per_m / 1_000_000
    output_cost = completion_tokens * output_cost_per_m / 1_000_000

    _COST_TRACKER["total_calls"] = int(_COST_TRACKER["total_calls"]) + 1
    _COST_TRACKER["prompt_tokens"] = int(_COST_TRACKER["prompt_tokens"]) + prompt_tokens
    _COST_TRACKER["completion_tokens"] = int(_COST_TRACKER["completion_tokens"]) + completion_tokens
    _COST_TRACKER["input_cost_usd"] = float(_COST_TRACKER["input_cost_usd"]) + input_cost
    _COST_TRACKER["output_cost_usd"] = float(_COST_TRACKER["output_cost_usd"]) + output_cost
    _COST_TRACKER["total_cost_usd"] = (
        float(_COST_TRACKER["input_cost_usd"]) + float(_COST_TRACKER["output_cost_usd"])
    )


def _parse_first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
            return value
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No JSON object/array found in content.", text, 0)


def _extract_structured_json_candidates(content: str) -> list[Any]:
    stripped = content.strip()
    if not stripped:
        raise ValueError("Together response contained empty structured content.")

    text_candidates: list[str] = [stripped]
    text_candidates.extend(
        candidate.strip()
        for candidate in _JSON_FENCE_PATTERN.findall(stripped)
        if candidate and candidate.strip()
    )

    parsed_candidates: list[Any] = []
    seen: set[str] = set()
    last_error: Exception | None = None

    for candidate in text_candidates:
        try:
            parsed = json.loads(candidate)
            key = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                parsed_candidates.append(parsed)
        except json.JSONDecodeError as exc:
            last_error = exc

        decoder = json.JSONDecoder()
        for idx, char in enumerate(candidate):
            if char not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            key = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            parsed_candidates.append(parsed)

    if parsed_candidates:
        return parsed_candidates

    preview = stripped[:240].replace("\n", "\\n")
    raise ValueError(f"Structured response was not valid JSON. Preview: {preview}") from last_error


def _is_list_like_annotation(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin in {list, tuple, set}:
        return True
    if origin in {None, ""}:
        return False
    if origin is not None:
        return any(_is_list_like_annotation(arg) for arg in get_args(annotation))
    return False


def _single_list_field_name(response_schema: type[BaseModel]) -> str | None:
    fields = response_schema.model_fields
    if len(fields) != 1:
        return None

    field_name = next(iter(fields.keys()))
    annotation = fields[field_name].annotation
    if _is_list_like_annotation(annotation):
        return field_name
    return None


def _payload_variants_for_schema(payload: Any, response_schema: type[BaseModel]) -> list[Any]:
    variants: list[Any] = [payload]
    list_field = _single_list_field_name(response_schema)

    if isinstance(payload, dict):
        for wrapper_key in _JSON_WRAPPER_KEYS:
            nested = payload.get(wrapper_key)
            if isinstance(nested, (dict, list)):
                variants.append(nested)

        if list_field is not None:
            if list_field not in payload:
                if "items" in payload and isinstance(payload["items"], list):
                    variants.append({list_field: payload["items"]})
                for value in payload.values():
                    if isinstance(value, list):
                        variants.append({list_field: value})
                        break

    elif isinstance(payload, list) and list_field is not None:
        variants.append({list_field: payload})

    unique_variants: list[Any] = []
    seen: set[str] = set()
    for variant in variants:
        try:
            key = json.dumps(variant, ensure_ascii=False, sort_keys=True)
        except TypeError:
            key = str(variant)
        if key in seen:
            continue
        seen.add(key)
        unique_variants.append(variant)
    return unique_variants


def _parse_structured_content(content: str, response_schema: type[TResponseModel]) -> TResponseModel:
    candidates = _extract_structured_json_candidates(content)
    last_error: ValidationError | None = None

    for candidate in candidates:
        for payload in _payload_variants_for_schema(candidate, response_schema):
            try:
                return response_schema.model_validate(payload)
            except ValidationError as exc:
                last_error = exc

    if last_error is not None:
        raise last_error

    raise ValueError("Structured response did not contain a valid payload for the expected schema.")


def _extract_structured_json(content: str) -> Any:
    candidates = _extract_structured_json_candidates(content)
    return candidates[0]


def _append_json_repair_suffix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patched = [dict(message) for message in messages]
    for idx in range(len(patched) - 1, -1, -1):
        content = patched[idx].get("content")
        if not isinstance(content, str):
            continue
        role = str(patched[idx].get("role", "")).strip().lower()
        if role not in {"user", "system"}:
            continue
        if _JSON_REPAIR_SUFFIX in content:
            return patched
        separator = "" if not content.strip() else "\n\n"
        patched[idx]["content"] = f"{content}{separator}{_JSON_REPAIR_SUFFIX}"
        return patched
    patched.append({"role": "user", "content": _JSON_REPAIR_SUFFIX})
    return patched


def _normalize_claim_type(value: str) -> ClaimType:
    normalized = value.strip().lower()
    aliases = {
        "methods": "method",
        "results": "result",
        "assumptions": "assumption",
        "contexts": "context",
        "limitations": "negative",
        "implication": "assumption",
        "implications": "assumption",
    }
    return ClaimType(aliases.get(normalized, normalized))


def _collapse_ws(value: str) -> str:
    return " ".join(value.split())


def _extract_json_text(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("reason", "why", "because", "text", "statement", "message", "detail", "value"):
            nested = value.get(key)
            if nested is None:
                continue
            parsed = _normalize_optional_text(nested)
            if parsed:
                return parsed

        parts = [_normalize_optional_text(item) for item in value.values()]
        joined = "; ".join(part for part in parts if part)
        return joined or None

    if isinstance(value, (list, tuple, set)):
        parts = [_normalize_optional_text(item) for item in value]
        joined = "; ".join(part for part in parts if part)
        return joined or None

    return None


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = _collapse_ws(value).strip().strip("\"'")
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered in _NONE_LIKE_TOKENS or lowered in {
            "not applicable",
            "not available",
            "not specified",
            "not provided",
        }:
            return None

        if cleaned.startswith(("```", "{", "[")):
            if cleaned.startswith("```"):
                fence_candidates = [candidate.strip() for candidate in _JSON_FENCE_PATTERN.findall(cleaned) if candidate]
                for candidate in fence_candidates:
                    parsed_candidate = _normalize_optional_text(candidate)
                    if parsed_candidate:
                        return parsed_candidate

            try:
                parsed_json = json.loads(cleaned)
            except json.JSONDecodeError:
                return cleaned

            recovered = _extract_json_text(parsed_json)
            if recovered:
                return recovered
            return None

        return cleaned

    if isinstance(value, (int, float, bool)):
        return _collapse_ws(str(value)) or None

    extracted = _extract_json_text(value)
    if extracted:
        return extracted

    return None


def _normalize_local_role(value: str) -> ClaimLocalRole | None:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "top_level_mechanism": "top_level",
        "top_level_method": "top_level",
        "high_level": "top_level",
        "detail": "implementation_detail",
        "impl_detail": "implementation_detail",
        "implementation": "implementation_detail",
        "empirical": "empirical_finding",
        "finding": "empirical_finding",
        "limitation_or_rejection": "limitation",
        "problem_gap": "context_gap",
        "gap": "context_gap",
        "positioning_claim": "positioning",
    }
    candidate = aliases.get(normalized, normalized)
    if candidate in _NONE_LIKE_TOKENS:
        return None
    try:
        return ClaimLocalRole(candidate)
    except ValueError:
        return None


def _normalize_parent_preference(value: str) -> ParentPreference | None:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "method_parent": "method",
        "context_parent": "context",
        "result_parent": "method",
        "auto": "none",
        "unknown": "none",
    }
    candidate = aliases.get(normalized, normalized)
    if candidate in _NONE_LIKE_TOKENS:
        return None
    try:
        return ParentPreference(candidate)
    except ValueError:
        return None


def _normalize_elaborates_seed_id(value: str) -> str | None:
    cleaned = _collapse_ws(value).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in _NONE_LIKE_TOKENS:
        return None
    return cleaned


def _build_structural_hints(fc: FlatClaim, claim_type: ClaimType) -> ClaimStructuralHints | None:
    local_role = _normalize_local_role(fc.local_role)
    preferred_parent = _normalize_parent_preference(fc.preferred_parent_type)
    elaborates_seed_id = _normalize_elaborates_seed_id(fc.elaborates_seed_id)

    if preferred_parent is None:
        if claim_type == ClaimType.context:
            preferred_parent = ParentPreference.context
        elif claim_type in {ClaimType.method, ClaimType.result, ClaimType.assumption, ClaimType.negative}:
            preferred_parent = ParentPreference.method

    if local_role is None:
        if claim_type == ClaimType.result:
            local_role = ClaimLocalRole.empirical_finding
        elif claim_type == ClaimType.assumption:
            local_role = ClaimLocalRole.assumption
        elif claim_type == ClaimType.negative:
            local_role = ClaimLocalRole.limitation
        elif claim_type == ClaimType.context:
            local_role = ClaimLocalRole.context_gap

    if local_role is None and preferred_parent is None and elaborates_seed_id is None:
        return None
    return ClaimStructuralHints(
        elaborates_seed_id=elaborates_seed_id,
        local_role=local_role,
        preferred_parent_type=preferred_parent,
    )


def flat_claim_to_raw(fc: FlatClaim, fallback_section: str = "") -> RawClaim:
    claim_id = _collapse_ws(fc.claim_id).strip() or "claim"
    source_section = _collapse_ws(fc.source_section).strip() or _collapse_ws(fallback_section).strip()
    if not source_section:
        source_section = "unknown"

    evidence_ids: list[str] = []
    for evidence_id in fc.evidence_ids:
        normalized = _collapse_ws(evidence_id).strip()
        if not normalized or normalized in evidence_ids:
            continue
        evidence_ids.append(normalized)

    entity_names: list[str] = []
    for entity_name in fc.entity_names:
        normalized = _collapse_ws(entity_name).strip()
        if not normalized or normalized in entity_names:
            continue
        entity_names.append(normalized)

    rejected_what = _normalize_optional_text(fc.rejected_what)
    rejected_why = _normalize_optional_text(fc.rejected_why)
    claim_type = _normalize_claim_type(fc.claim_type)
    statement = _collapse_ws(fc.statement).strip()
    if not statement:
        raise ValueError("Claim statement cannot be empty after normalization.")

    if claim_type != ClaimType.negative:
        rejected_what = None
        rejected_why = None

    return RawClaim(
        claim_id=claim_id,
        claim_type=claim_type,
        statement=statement,
        source_section=source_section,
        evidence=[EvidencePointer(artifact_id=evidence_id, role="supports") for evidence_id in evidence_ids],
        entity_names=entity_names,
        rejected_what=rejected_what or None,
        rejected_why=rejected_why or None,
        structural_hints=_build_structural_hints(fc, claim_type),
    )


async def call_model(
    tier: ModelTier,
    messages: list[dict[str, Any]],
    response_schema: type[TResponseModel] | None = None,
    config: Any | None = None,
) -> TResponseModel | str:
    cfg = _resolve_config(config)
    model_cfg = _get_model_cfg(cfg, tier)
    api_cfg = _get_api_cfg(cfg)
    schema_payload: dict[str, Any] | None = None

    kwargs_base: dict[str, Any] = {
        "model": _get_value(model_cfg, "model"),
        "temperature": _get_value(model_cfg, "temperature"),
        "max_tokens": _get_value(model_cfg, "max_tokens"),
    }

    if response_schema is not None:
        schema_payload = response_schema.model_json_schema()
        kwargs_base["response_format"] = {
            "type": "json_object",
            "schema": schema_payload,
        }

    max_retries = int(_get_value_or_default(api_cfg, "max_retries", 0))
    retry_backoff_base = float(_get_value_or_default(api_cfg, "retry_backoff_base", 2.0))
    model_name = str(_get_value(model_cfg, "model"))

    prompt_serialized = json.dumps(messages, ensure_ascii=False, default=str)
    prompt_tokens_est = len(prompt_serialized) // 4
    schema_tokens = 0
    if schema_payload is not None:
        schema_tokens = len(json.dumps(schema_payload, ensure_ascii=False, default=str)) // 4

    console.print(
        f"  ({tier}) calling {model_name} | ~{prompt_tokens_est} prompt tokens | ~{schema_tokens} schema tokens",
        style="dim",
    )

    context_length = _get_value_or_default(model_cfg, "context_length", None)
    if isinstance(context_length, int) and context_length > 0:
        total_est = prompt_tokens_est + schema_tokens
        if total_est > int(context_length * 0.8):
            console.print(
                f"  [yellow]WARNING: prompt ({total_est} est.) near context limit ({context_length})[/yellow]"
            )

    last_error: Exception | None = None
    structured_repair_used = False
    current_messages = [dict(message) for message in messages]

    for attempt in range(max_retries + 1):
        kwargs = dict(kwargs_base)
        kwargs["messages"] = current_messages
        started_at = time.monotonic()
        try:
            response = await _get_client().chat.completions.create(**kwargs)
            elapsed = time.monotonic() - started_at
            _track_cost(response, model_cfg)

            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

            content = response.choices[0].message.content
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = str(content)
            content_len = len(content)

            console.print(
                f"  ({tier}) attempt {attempt}: {elapsed:.1f}s | "
                f"{prompt_tokens} in / {completion_tokens} out | content_len={content_len}",
                style="dim",
            )

            if response_schema is not None and not content.strip():
                console.print(
                    f"  ({tier}) [red]EMPTY RESPONSE[/red] — schema_tokens={schema_tokens}"
                )
                if attempt < max_retries:
                    if not structured_repair_used:
                        structured_repair_used = True
                        current_messages = _append_json_repair_suffix(messages)
                    wait = retry_backoff_base**attempt
                    await asyncio.sleep(wait)
                    continue
                raise ValueError("Empty structured content after all retries.")

            if response_schema is not None:
                return _parse_structured_content(content, response_schema)

            return content
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            elapsed = time.monotonic() - started_at
            console.print(
                f"  ({tier}) attempt {attempt} FAILED: {elapsed:.1f}s | {exc}",
                style="red dim",
            )
            last_error = exc
            if response_schema is not None and not structured_repair_used:
                structured_repair_used = True
                current_messages = _append_json_repair_suffix(messages)
            if attempt < max_retries:
                wait = retry_backoff_base**attempt
                await asyncio.sleep(wait)
                continue
            break
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            console.print(
                f"  ({tier}) attempt {attempt} FAILED: {elapsed:.1f}s | {exc}",
                style="red dim",
            )
            last_error = exc
            if attempt < max_retries:
                wait = retry_backoff_base**attempt
                await asyncio.sleep(wait)
                continue
            break

    if last_error is not None:
        raise last_error
    raise RuntimeError("Together API call failed without a response.")


async def call_model_with_fallback(
    tier: ModelTier,
    messages: list[dict[str, Any]],
    response_schema: type[TResponseModel] | None = None,
    config: Any | None = None,
) -> TResponseModel | str:
    if response_schema is None:
        return await call_model(
            tier=tier,
            messages=messages,
            response_schema=None,
            config=config,
        )

    try:
        return await call_model(
            tier=tier,
            messages=messages,
            response_schema=response_schema,
            config=config,
        )
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
        repaired_messages = _append_json_repair_suffix(messages)
        return await call_model(
            tier=tier,
            messages=repaired_messages,
            response_schema=response_schema,
            config=config,
        )


async def preflight_model_tiers(
    tiers: list[ModelTier],
    *,
    config: Any | None = None,
) -> None:
    unique_tiers: list[ModelTier] = []
    for tier in tiers:
        if tier in unique_tiers:
            continue
        unique_tiers.append(tier)

    if not unique_tiers:
        return

    messages = [
        {
            "role": "system",
            "content": "Return strict JSON only.",
        },
        {
            "role": "user",
            "content": 'Return exactly this JSON object: {"ok": true}',
        },
    ]

    failures: list[str] = []
    for tier in unique_tiers:
        try:
            response = await call_model(
                tier=tier,
                messages=messages,
                response_schema=_PreflightStructuredResponse,
                config=config,
            )
            if not isinstance(response, _PreflightStructuredResponse) or response.ok is not True:
                failures.append(f"{tier}: unexpected preflight payload")
        except Exception as exc:
            failures.append(f"{tier}: {exc}")

    if failures:
        detail = "; ".join(failures)
        raise ModelPreflightError(f"Model preflight failed for tier(s): {detail}")


__all__ = [
    "ModelPreflightError",
    "call_model",
    "call_model_with_fallback",
    "flat_claim_to_raw",
    "get_cost_tracker",
    "preflight_model_tiers",
    "reset_cost_tracker",
]
