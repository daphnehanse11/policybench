"""AI-alone evaluation using LiteLLM (no tools provided)."""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable

import litellm
import pandas as pd
from litellm import completion, completion_cost, responses

from policybench.config import MODELS, PROGRAMS
from policybench.prompts import (
    get_variable_description,
    make_no_tools_batch_prompt,
    make_no_tools_batch_repair_prompt,
)
from policybench.scenarios import Scenario, scenario_to_dict

MAX_RETRIES = 2
RETRY_BASE_DELAY = 2
REQUEST_TIMEOUT_SECONDS = 20
GEMINI_PRO_REQUEST_TIMEOUT_SECONDS = 60
XAI_REASONING_REQUEST_TIMEOUT_SECONDS = 60
XAI_GROK_420_REASONING_REQUEST_TIMEOUT_SECONDS = 120
CHECKPOINT_EVERY_ROWS = 25
MAX_REPAIR_ROUNDS = 2
RESUME_METADATA_VERSION = 1
DEFAULT_MAX_COMPLETION_TOKENS = 64
EXTENDED_MAX_COMPLETION_TOKENS = 256
EXPLANATION_MAX_COMPLETION_TOKENS = 1024
GEMINI_JSON_MAX_COMPLETION_TOKENS = 512
GEMINI_PRO_JSON_MAX_COMPLETION_TOKENS = 2048
ANSWER_FUNCTION_NAME = "submit_answers"
CLAUDE_EXPLANATION_CHUNK_SIZE = 3
NON_RETRYABLE_ERRORS = (
    litellm.AuthenticationError,
    litellm.BadRequestError,
    litellm.ContextWindowExceededError,
    litellm.InvalidRequestError,
    litellm.PermissionDeniedError,
    litellm.UnsupportedParamsError,
    litellm.UnprocessableEntityError,
)
MODEL_FATAL_ERRORS = (
    litellm.AuthenticationError,
    litellm.BadRequestError,
    litellm.ContextWindowExceededError,
    litellm.InvalidRequestError,
    litellm.NotFoundError,
    litellm.PermissionDeniedError,
    litellm.UnsupportedParamsError,
    litellm.UnprocessableEntityError,
)
STANDALONE_NUMBER_RE = re.compile(r"^\$?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")


def _format_error(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error).replace(chr(10), ' ')[:500]}"


def _should_retry(error: Exception) -> bool:
    return not isinstance(error, NON_RETRYABLE_ERRORS)


def _get_usage_value(obj, key: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_usage_metadata(
    response, model_id: str, messages: list[dict], content: str
) -> dict:
    usage = getattr(response, "usage", None)
    prompt_tokens_details = _get_usage_value(usage, "prompt_tokens_details")
    if prompt_tokens_details is None:
        prompt_tokens_details = _get_usage_value(usage, "input_tokens_details")
    completion_tokens_details = _get_usage_value(usage, "completion_tokens_details")
    if completion_tokens_details is None:
        completion_tokens_details = _get_usage_value(usage, "output_tokens_details")
    reasoning_tokens = _get_usage_value(usage, "reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _get_usage_value(
            completion_tokens_details, "reasoning_tokens"
        )

    provider_reported_cost_usd = _get_usage_value(usage, "cost")
    reconstructed_cost_usd = None
    try:
        cost_kwargs = {
            "completion_response": response,
            "model": model_id,
        }
        if messages:
            cost_kwargs["messages"] = messages
        if content:
            cost_kwargs["completion"] = content
        reconstructed_cost_usd = completion_cost(**cost_kwargs)
    except Exception:
        reconstructed_cost_usd = None

    total_cost_usd = provider_reported_cost_usd
    if total_cost_usd is None:
        total_cost_usd = reconstructed_cost_usd

    return {
        "prompt_tokens": _get_usage_value(usage, "prompt_tokens")
        or _get_usage_value(usage, "input_tokens"),
        "completion_tokens": _get_usage_value(usage, "completion_tokens")
        or _get_usage_value(usage, "output_tokens"),
        "total_tokens": _get_usage_value(usage, "total_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "cached_prompt_tokens": _get_usage_value(
            prompt_tokens_details, "cached_tokens"
        ),
        "provider_reported_cost_usd": provider_reported_cost_usd,
        "reconstructed_cost_usd": reconstructed_cost_usd,
        "total_cost_usd": total_cost_usd,
        "cost_is_estimated": (
            provider_reported_cost_usd is None and total_cost_usd is not None
        ),
        "estimated_cost_usd": total_cost_usd,
    }


def _parse_standalone_number(text: str) -> float | None:
    cleaned = text.strip()
    if not cleaned or not STANDALONE_NUMBER_RE.fullmatch(cleaned):
        return None
    return float(cleaned.replace(",", "").replace("$", ""))


def _required_explanation_chunk_size(
    model_id: str, include_explanations: bool
) -> int | None:
    if include_explanations and model_id.startswith("claude-"):
        return CLAUDE_EXPLANATION_CHUNK_SIZE
    return None


def _chunk_variables(variables: list[str], chunk_size: int) -> list[list[str]]:
    return [variables[i : i + chunk_size] for i in range(0, len(variables), chunk_size)]


def _sum_optional_field(results: list[dict], field: str) -> float | int | None:
    values = [result.get(field) for result in results if result.get(field) is not None]
    if not values:
        return None
    return sum(values)


def _completion_controls(model_id: str, include_explanations: bool = False) -> dict:
    if model_id.startswith("gemini/"):
        if model_id == "gemini/gemini-3.1-pro-preview":
            return {"max_completion_tokens": GEMINI_PRO_JSON_MAX_COMPLETION_TOKENS}
        return {"max_completion_tokens": GEMINI_JSON_MAX_COMPLETION_TOKENS}
    if model_id.startswith("xai/"):
        if include_explanations:
            return {"max_tokens": EXPLANATION_MAX_COMPLETION_TOKENS}
        return {"max_tokens": EXTENDED_MAX_COMPLETION_TOKENS}
    if model_id.startswith("gpt-5"):
        if include_explanations:
            return {"max_completion_tokens": EXPLANATION_MAX_COMPLETION_TOKENS}
        return {"max_completion_tokens": EXTENDED_MAX_COMPLETION_TOKENS}
    if model_id.startswith("claude-"):
        return {"max_completion_tokens": EXTENDED_MAX_COMPLETION_TOKENS}
    return {"max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS}


def _request_timeout_seconds(model_id: str) -> int:
    if model_id == "gemini/gemini-3.1-pro-preview":
        return GEMINI_PRO_REQUEST_TIMEOUT_SECONDS
    if model_id == "xai/grok-4.20-reasoning":
        return XAI_GROK_420_REASONING_REQUEST_TIMEOUT_SECONDS
    if (
        model_id.startswith("xai/")
        and "reasoning" in model_id
        and "non-reasoning" not in model_id
    ):
        return XAI_REASONING_REQUEST_TIMEOUT_SECONDS
    return REQUEST_TIMEOUT_SECONDS


def _answer_contract_for_model(model_id: str) -> str:
    if model_id.startswith("gemini/"):
        return "json"
    return "tool"


def _uses_responses_api(model_id: str) -> bool:
    return model_id.startswith("gpt-5")


def _chat_completion_request_kwargs(
    scenario: Scenario,
    variables: list[str],
    model_id: str,
    repair: bool = False,
    include_explanations: bool = False,
) -> tuple[list[dict], dict]:
    answer_contract = _answer_contract_for_model(model_id)
    prompt_builder = (
        make_no_tools_batch_repair_prompt if repair else make_no_tools_batch_prompt
    )
    prompt = prompt_builder(
        scenario,
        variables,
        answer_contract=answer_contract,
        include_explanations=include_explanations,
    )
    messages = [{"role": "user", "content": prompt}]
    request_kwargs = {
        "model": model_id,
        "messages": messages,
        "caching": True,
        "timeout": _request_timeout_seconds(model_id),
        **_completion_controls(model_id, include_explanations=include_explanations),
    }
    if answer_contract == "tool":
        tool = _build_answer_tool(
            variables,
            country=scenario.country,
            include_explanations=include_explanations,
        )
        request_kwargs.update(
            {
                "tools": [tool],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": ANSWER_FUNCTION_NAME},
                },
            }
        )
    else:
        request_kwargs["response_format"] = {"type": "json_object"}
    return messages, request_kwargs


def _responses_request_kwargs(
    scenario: Scenario,
    variables: list[str],
    model_id: str,
    repair: bool = False,
    include_explanations: bool = False,
) -> tuple[list[dict], dict]:
    answer_contract = _answer_contract_for_model(model_id)
    prompt_builder = (
        make_no_tools_batch_repair_prompt if repair else make_no_tools_batch_prompt
    )
    prompt = prompt_builder(
        scenario,
        variables,
        answer_contract=answer_contract,
        include_explanations=include_explanations,
    )
    request_kwargs = {
        "model": model_id,
        "input": prompt,
        "timeout": _request_timeout_seconds(model_id),
        "max_output_tokens": _completion_controls(
            model_id,
            include_explanations=include_explanations,
        )["max_completion_tokens"],
    }
    if answer_contract == "tool":
        request_kwargs.update(
            {
                "tools": [
                    _responses_tool_schema(
                        variables,
                        country=scenario.country,
                        include_explanations=include_explanations,
                    )
                ],
                "tool_choice": {
                    "type": "function",
                    "name": ANSWER_FUNCTION_NAME,
                },
            }
        )
    return [{"role": "user", "content": prompt}], request_kwargs


def _build_answer_tool(
    variables: list[str],
    country: str = "us",
    include_explanations: bool = False,
) -> dict:
    answers_schema = {
        "type": "object",
        "properties": {
            variable: {
                "type": "number",
                "description": get_variable_description(variable, country=country),
            }
            for variable in variables
        },
        "required": list(variables),
        "additionalProperties": False,
    }
    parameters = answers_schema
    if include_explanations:
        parameters = {
            "type": "object",
            "properties": {
                "answers": answers_schema,
                "explanations": {
                    "type": "object",
                    "properties": {
                        variable: {
                            "type": "string",
                            "description": (
                                f"Brief explanation for the estimated {variable} value"
                            ),
                        }
                        for variable in variables
                    },
                    "required": list(variables),
                    "additionalProperties": False,
                },
            },
            "required": ["answers", "explanations"],
            "additionalProperties": False,
        }
    return {
        "type": "function",
        "function": {
            "name": ANSWER_FUNCTION_NAME,
            "description": (
                "Submit all requested numeric answers for the benchmark. "
                "Every requested key is required, including keys whose value is 0."
            ),
            "parameters": parameters,
        },
    }


def _responses_tool_schema(
    variables: list[str],
    country: str = "us",
    include_explanations: bool = False,
) -> dict:
    function_schema = _build_answer_tool(
        variables,
        country=country,
        include_explanations=include_explanations,
    )["function"]
    return {
        "type": "function",
        "name": function_schema["name"],
        "description": function_schema["description"],
        "parameters": function_schema["parameters"],
    }


def _normalize_variables(variables: str | Iterable[str]) -> list[str]:
    if isinstance(variables, str):
        return [variables]
    return list(variables)


def _sum_optional_numbers(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _combine_raw_responses(raw_responses: list[str | None]) -> str | None:
    present = [raw_response for raw_response in raw_responses if raw_response]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return json.dumps({"responses": present})


def _missing_variables(predictions: dict[str, float | None]) -> list[str]:
    return [variable for variable, value in predictions.items() if value is None]


def _merge_predictions(
    base: dict[str, float | None],
    incoming: dict[str, float | None],
) -> dict[str, float | None]:
    merged = dict(base)
    for variable, value in incoming.items():
        if value is not None:
            merged[variable] = value
    return merged


def _aggregate_request_results(results: list[dict]) -> dict:
    if not results:
        return {
            "raw_response": None,
            "elapsed_seconds": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "cached_prompt_tokens": None,
            "provider_reported_cost_usd": None,
            "reconstructed_cost_usd": None,
            "total_cost_usd": None,
            "cost_is_estimated": None,
            "estimated_cost_usd": None,
        }

    cost_flags = [result.get("cost_is_estimated") for result in results]
    if any(flag is True for flag in cost_flags):
        cost_is_estimated = True
    elif any(flag is False for flag in cost_flags):
        cost_is_estimated = False
    else:
        cost_is_estimated = None

    return {
        "raw_response": _combine_raw_responses(
            [result.get("raw_response") for result in results]
        ),
        "elapsed_seconds": _sum_optional_numbers(
            result.get("elapsed_seconds") for result in results
        ),
        "prompt_tokens": _sum_optional_numbers(
            result.get("prompt_tokens") for result in results
        ),
        "completion_tokens": _sum_optional_numbers(
            result.get("completion_tokens") for result in results
        ),
        "total_tokens": _sum_optional_numbers(
            result.get("total_tokens") for result in results
        ),
        "reasoning_tokens": _sum_optional_numbers(
            result.get("reasoning_tokens") for result in results
        ),
        "cached_prompt_tokens": _sum_optional_numbers(
            result.get("cached_prompt_tokens") for result in results
        ),
        "provider_reported_cost_usd": _sum_optional_numbers(
            result.get("provider_reported_cost_usd") for result in results
        ),
        "reconstructed_cost_usd": _sum_optional_numbers(
            result.get("reconstructed_cost_usd") for result in results
        ),
        "total_cost_usd": _sum_optional_numbers(
            result.get("total_cost_usd") for result in results
        ),
        "cost_is_estimated": cost_is_estimated,
        "estimated_cost_usd": _sum_optional_numbers(
            result.get("estimated_cost_usd") for result in results
        ),
    }


def extract_number(text: str) -> float | None:
    """Extract a numeric value from a valid answer payload."""
    if not text:
        return None

    full_match = _parse_standalone_number(text)
    if full_match is not None:
        return full_match

    try:
        payload = json.loads(text)
    except Exception:
        return None

    if isinstance(payload, dict) and "answer" in payload:
        return _coerce_prediction_value(payload.get("answer"))

    return None


def _coerce_prediction_value(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        return extract_number(value)
    return None


def _extract_predictions_from_payload(
    payload,
    variables: list[str],
) -> dict[str, float | None]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = None

    if isinstance(payload, dict) and isinstance(payload.get("answers"), dict):
        payload = payload["answers"]

    predictions = {variable: None for variable in variables}
    if isinstance(payload, dict):
        if len(variables) == 1 and "answer" in payload:
            predictions[variables[0]] = _coerce_prediction_value(payload.get("answer"))
            return predictions
        for variable in variables:
            predictions[variable] = _coerce_prediction_value(payload.get(variable))
        return predictions

    return predictions


def _extract_explanations_from_payload(
    payload,
    variables: list[str],
) -> dict[str, str | None]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = None

    explanations = {variable: None for variable in variables}
    if not isinstance(payload, dict):
        return explanations

    explanation_payload = payload.get("explanations")
    if not isinstance(explanation_payload, dict):
        return explanations

    for variable in variables:
        value = explanation_payload.get(variable)
        if isinstance(value, str):
            cleaned = value.strip()
            explanations[variable] = cleaned or None
    return explanations


def _missing_explanations(
    explanations: dict[str, str | None],
    variables: list[str],
) -> list[str]:
    return [
        variable
        for variable in variables
        if not isinstance(explanations.get(variable), str)
        or not explanations.get(variable, "").strip()
    ]


def _get_tool_call_function(tool_call):
    if tool_call is None:
        return None
    if isinstance(tool_call, dict):
        return tool_call.get("function")
    return getattr(tool_call, "function", None)


def _get_function_name(function_call) -> str | None:
    if function_call is None:
        return None
    if isinstance(function_call, dict):
        return function_call.get("name")
    return getattr(function_call, "name", None)


def _get_function_arguments(function_call):
    if function_call is None:
        return None
    if isinstance(function_call, dict):
        return function_call.get("arguments")
    return getattr(function_call, "arguments", None)


def _serialize_function_call(function_call) -> dict | None:
    name = _get_function_name(function_call)
    arguments = _get_function_arguments(function_call)
    if not isinstance(name, str):
        name = None
    if isinstance(arguments, (dict, list)):
        pass
    elif not isinstance(arguments, str):
        arguments = None

    if name is None and arguments is None:
        return None
    if function_call is None:
        return None
    return {
        "name": name,
        "arguments": arguments,
    }


def _serialize_response_payload(
    content, tool_calls=None, function_call=None
) -> str | None:
    payload = {}
    if content is not None:
        payload["content"] = content

    serialized_tool_calls = []
    for tool_call in tool_calls or []:
        function = _get_tool_call_function(tool_call)
        serialized_tool_calls.append(
            {
                "name": _get_function_name(function),
                "arguments": _get_function_arguments(function),
            }
        )
    if serialized_tool_calls:
        payload["tool_calls"] = serialized_tool_calls

    serialized_function_call = _serialize_function_call(function_call)
    if serialized_function_call is not None:
        payload["function_call"] = serialized_function_call

    if not payload:
        return None
    if set(payload) == {"content"} and isinstance(content, str):
        return content
    return json.dumps(payload)


def _get_response_item_attr(item, key: str):
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _responses_content_and_tool_calls(response) -> tuple[str | None, list[dict]]:
    output = getattr(response, "output", None) or []
    content_segments = []
    tool_calls = []
    for item in output:
        item_type = _get_response_item_attr(item, "type")
        if item_type == "function_call":
            tool_calls.append(
                {
                    "function": {
                        "name": _get_response_item_attr(item, "name"),
                        "arguments": _get_response_item_attr(item, "arguments"),
                    }
                }
            )
            continue
        if item_type != "message":
            continue
        for part in _get_response_item_attr(item, "content") or []:
            if _get_response_item_attr(part, "type") != "output_text":
                continue
            text = _get_response_item_attr(part, "text")
            if text:
                content_segments.append(text)

    content = getattr(response, "output_text", None)
    if not content and content_segments:
        content = "\n".join(content_segments)
    return content, tool_calls


def extract_prediction(
    content: str | None,
    tool_calls=None,
    function_call=None,
) -> float | None:
    """Extract a numeric prediction from structured tool output or valid JSON."""
    for tool_call in tool_calls or []:
        function = _get_tool_call_function(tool_call)
        if _get_function_name(function) != ANSWER_FUNCTION_NAME:
            continue
        arguments = _get_function_arguments(function)
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = None
        prediction = extract_number(arguments or "")
        if prediction is not None:
            return prediction

    if _get_function_name(function_call) == ANSWER_FUNCTION_NAME:
        arguments = _get_function_arguments(function_call)
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = None
        prediction = extract_number(arguments or "")
        if prediction is not None:
            return prediction

    return extract_number(content or "")


def extract_predictions(
    content: str | None,
    variables: list[str],
    tool_calls=None,
    function_call=None,
) -> dict[str, float | None]:
    """Extract a mapping of variable -> prediction from tool output or valid JSON."""
    for tool_call in tool_calls or []:
        function = _get_tool_call_function(tool_call)
        if _get_function_name(function) != ANSWER_FUNCTION_NAME:
            continue
        arguments = _get_function_arguments(function)
        predictions = _extract_predictions_from_payload(arguments, variables)
        if any(value is not None for value in predictions.values()):
            return predictions

    if _get_function_name(function_call) == ANSWER_FUNCTION_NAME:
        arguments = _get_function_arguments(function_call)
        predictions = _extract_predictions_from_payload(arguments, variables)
        if any(value is not None for value in predictions.values()):
            return predictions

    return _extract_predictions_from_payload(content, variables)


def extract_explanations(
    content: str | None,
    variables: list[str],
    tool_calls=None,
    function_call=None,
) -> dict[str, str | None]:
    """Extract optional per-variable explanations from structured output."""
    for tool_call in tool_calls or []:
        function = _get_tool_call_function(tool_call)
        if _get_function_name(function) != ANSWER_FUNCTION_NAME:
            continue
        arguments = _get_function_arguments(function)
        explanations = _extract_explanations_from_payload(arguments, variables)
        if any(value is not None for value in explanations.values()):
            return explanations

    if _get_function_name(function_call) == ANSWER_FUNCTION_NAME:
        arguments = _get_function_arguments(function_call)
        explanations = _extract_explanations_from_payload(arguments, variables)
        if any(value is not None for value in explanations.values()):
            return explanations

    return _extract_explanations_from_payload(content, variables)


def _request_predictions_once(
    scenario: Scenario,
    variables: list[str],
    model_id: str,
    *,
    repair: bool = False,
    include_explanations: bool = False,
) -> dict:
    if _uses_responses_api(model_id):
        messages, request_kwargs = _responses_request_kwargs(
            scenario=scenario,
            variables=variables,
            model_id=model_id,
            repair=repair,
            include_explanations=include_explanations,
        )
        request_fn = responses
    else:
        messages, request_kwargs = _chat_completion_request_kwargs(
            scenario=scenario,
            variables=variables,
            model_id=model_id,
            repair=repair,
            include_explanations=include_explanations,
        )
        request_fn = completion

    for attempt in range(MAX_RETRIES):
        try:
            started_at = time.perf_counter()
            response = request_fn(**request_kwargs)
            elapsed_seconds = time.perf_counter() - started_at
            if _uses_responses_api(model_id):
                content, tool_calls = _responses_content_and_tool_calls(response)
                function_call = None
            else:
                message = response.choices[0].message
                content = getattr(message, "content", None)
                tool_calls = getattr(message, "tool_calls", None)
                function_call = getattr(message, "function_call", None)
            raw_response = _serialize_response_payload(
                content=content,
                tool_calls=tool_calls,
                function_call=function_call,
            )
            predictions = extract_predictions(
                content=content,
                variables=variables,
                tool_calls=tool_calls,
                function_call=function_call,
            )
            explanations = extract_explanations(
                content=content,
                variables=variables,
                tool_calls=tool_calls,
                function_call=function_call,
            )
            return {
                "predictions": predictions,
                "explanations": explanations,
                "raw_response": raw_response,
                "elapsed_seconds": elapsed_seconds,
                **_extract_usage_metadata(
                    response,
                    model_id,
                    messages,
                    raw_response or content,
                ),
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1 or not _should_retry(e):
                raise
            delay = RETRY_BASE_DELAY * (2**attempt)
            print(f"  Retry {attempt + 1}: {e!r:.60s}... {delay}s")
            time.sleep(delay)

    raise RuntimeError("Request loop exited unexpectedly")


def run_single_no_tools(
    scenario: Scenario,
    variable: str | Iterable[str],
    model_id: str,
    include_explanations: bool = False,
    _allow_chunking: bool = True,
) -> dict:
    """Run a single scenario for one or more variables without tools."""
    variables = _normalize_variables(variable)
    chunk_size = _required_explanation_chunk_size(model_id, include_explanations)
    if _allow_chunking and chunk_size and len(variables) > chunk_size:
        chunk_results = []
        predictions = {}
        explanations = {}
        errors = []
        for chunk in _chunk_variables(variables, chunk_size):
            chunk_result = run_single_no_tools(
                scenario,
                chunk,
                model_id,
                include_explanations=include_explanations,
                _allow_chunking=False,
            )
            chunk_results.append(chunk_result)
            predictions.update(chunk_result["predictions"])
            explanations.update(
                {
                    variable: value
                    for variable, value in chunk_result.get("explanations", {}).items()
                    if value is not None
                }
            )
            if chunk_result.get("error"):
                errors.append(chunk_result["error"])

        cost_rows = [
            result
            for result in chunk_results
            if result.get("total_cost_usd") is not None
        ]
        raw_response = json.dumps(
            {
                "chunked_responses": [
                    {
                        "variables": chunk,
                        "raw_response": result.get("raw_response"),
                    }
                    for chunk, result in zip(
                        _chunk_variables(variables, chunk_size),
                        chunk_results,
                        strict=True,
                    )
                ]
            }
        )
        return {
            "predictions": predictions,
            "explanations": explanations,
            "prediction": predictions[variables[0]] if len(variables) == 1 else None,
            "error": "; ".join(errors) if errors else None,
            "raw_response": raw_response,
            "elapsed_seconds": _sum_optional_field(chunk_results, "elapsed_seconds"),
            "prompt_tokens": _sum_optional_field(chunk_results, "prompt_tokens"),
            "completion_tokens": _sum_optional_field(
                chunk_results, "completion_tokens"
            ),
            "total_tokens": _sum_optional_field(chunk_results, "total_tokens"),
            "reasoning_tokens": _sum_optional_field(chunk_results, "reasoning_tokens"),
            "cached_prompt_tokens": _sum_optional_field(
                chunk_results, "cached_prompt_tokens"
            ),
            "provider_reported_cost_usd": _sum_optional_field(
                chunk_results, "provider_reported_cost_usd"
            ),
            "reconstructed_cost_usd": _sum_optional_field(
                chunk_results, "reconstructed_cost_usd"
            ),
            "total_cost_usd": _sum_optional_field(chunk_results, "total_cost_usd"),
            "cost_is_estimated": (
                all(bool(result.get("cost_is_estimated")) for result in cost_rows)
                if cost_rows
                else None
            ),
            "estimated_cost_usd": _sum_optional_field(
                chunk_results, "estimated_cost_usd"
            ),
        }

    request_results = []
    initial_result = _request_predictions_once(
        scenario,
        variables,
        model_id,
        repair=False,
        include_explanations=include_explanations,
    )
    request_results.append(initial_result)
    predictions = dict(initial_result["predictions"])
    explanations = dict(initial_result.get("explanations", {}))

    missing = _missing_variables(predictions)
    missing_explanations = (
        _missing_explanations(explanations, variables) if include_explanations else []
    )
    repair_errors = []
    for _ in range(MAX_REPAIR_ROUNDS):
        repair_targets = sorted(set(missing) | set(missing_explanations))
        if not repair_targets:
            break
        try:
            repair_result = _request_predictions_once(
                scenario,
                repair_targets,
                model_id,
                repair=True,
                include_explanations=include_explanations,
            )
        except Exception as error:
            repair_errors.append(_format_error(error))
            break
        request_results.append(repair_result)
        predictions = _merge_predictions(predictions, repair_result["predictions"])
        explanations.update(
            {
                variable: value
                for variable, value in repair_result.get("explanations", {}).items()
                if value is not None
            }
        )
        missing = _missing_variables(predictions)
        missing_explanations = (
            _missing_explanations(explanations, variables)
            if include_explanations
            else []
        )

    if missing:
        repair_errors.append(
            "Missing predictions after repair: " + ", ".join(sorted(missing))
        )
    if include_explanations and missing_explanations:
        repair_errors.append(
            "Missing explanations after repair: "
            + ", ".join(sorted(missing_explanations))
        )

    aggregated = _aggregate_request_results(request_results)
    return {
        "predictions": predictions,
        "explanations": explanations,
        "prediction": predictions[variables[0]] if len(variables) == 1 else None,
        "error": "; ".join(repair_errors) if repair_errors else None,
        **aggregated,
    }


def _load_existing_rows(
    output_path: str | None,
    programs: list[str],
    include_explanations: bool = False,
) -> tuple[list[dict], set[tuple[str, str]]]:
    if not output_path:
        return [], set()

    path = Path(output_path)
    if not path.exists() or path.stat().st_size == 0:
        return [], set()

    existing = pd.read_csv(path)
    if existing.empty:
        return [], set()

    if existing.empty:
        return [], set()

    group_columns = ["model", "scenario_id"]
    completed_keys: set[tuple[str, str]] = set()
    keep_mask = pd.Series(False, index=existing.index)

    for key, group in existing.groupby(group_columns):
        has_all_programs = set(group["variable"]) >= set(programs)
        has_no_errors = "error" not in group.columns or group["error"].isna().all()
        has_all_predictions = (
            "prediction" not in group.columns or group["prediction"].notna().all()
        )
        has_all_explanations = (
            not include_explanations
            or "explanation" not in group.columns
            or group["explanation"].fillna("").str.strip().ne("").all()
        )
        if (
            has_all_programs
            and has_no_errors
            and has_all_predictions
            and has_all_explanations
        ):
            completed_keys.add((key[0], key[1]))
            keep_mask.loc[group.index] = True

    retained = existing.loc[keep_mask].copy()
    rows = retained.to_dict("records")
    return rows, completed_keys


def _output_metadata_path(output_path: str | None) -> Path | None:
    if not output_path:
        return None
    return Path(f"{output_path}.meta.json")


def _serialize_scenario(scenario: Scenario) -> str:
    return json.dumps(
        scenario_to_dict(scenario),
        separators=(",", ":"),
        sort_keys=True,
    )


def _build_resume_metadata(
    *,
    task: str,
    scenarios: list[Scenario],
    models: dict[str, str],
    programs: list[str],
    run_id: str | None,
    include_explanations: bool,
) -> dict:
    scenario_signature = json.dumps(
        [
            {
                "scenario_id": scenario.id,
                "scenario_json": _serialize_scenario(scenario),
            }
            for scenario in scenarios
        ],
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "metadata_version": RESUME_METADATA_VERSION,
        "task": task,
        "run_id": run_id,
        "include_explanations": include_explanations,
        "scenario_count": len(scenarios),
        "scenario_hash": hashlib.sha256(
            scenario_signature.encode("utf-8")
        ).hexdigest(),
        "programs": sorted(programs),
        "models": {name: models[name] for name in sorted(models)},
    }


def _write_resume_metadata(output_path: str | None, metadata: dict) -> None:
    metadata_path = _output_metadata_path(output_path)
    if metadata_path is None:
        return
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _validate_resume_metadata(output_path: str | None, expected: dict) -> None:
    if not output_path:
        return

    path = Path(output_path)
    metadata_path = _output_metadata_path(output_path)
    if metadata_path is None:
        return

    if not path.exists() or path.stat().st_size == 0:
        return
    if not metadata_path.exists():
        raise ValueError(
            f"Existing output at {path} is missing its resume metadata sidecar "
            f"at {metadata_path}. Delete the stale output or use a fresh path."
        )

    existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    mismatches = []
    for key in (
        "metadata_version",
        "task",
        "run_id",
        "include_explanations",
        "scenario_count",
        "scenario_hash",
        "programs",
        "models",
    ):
        if existing.get(key) != expected.get(key):
            mismatches.append(key)

    if mismatches:
        mismatch_list = ", ".join(mismatches)
        raise ValueError(
            f"Existing output at {path} does not match the requested benchmark "
            f"settings ({mismatch_list}). Use a fresh output path instead of "
            "resuming this file."
        )


def _save_checkpoint(
    output_path: str | None,
    rows: list[dict],
    metadata: dict,
) -> None:
    if not output_path:
        return
    pd.DataFrame(rows).to_csv(output_path, index=False)
    _write_resume_metadata(output_path, metadata)


def _load_existing_single_output_rows(
    output_path: str | None,
    include_explanations: bool = False,
) -> tuple[list[dict], set[tuple[str, str, str]]]:
    if not output_path or not Path(output_path).exists():
        return [], set()

    existing = pd.read_csv(output_path)
    required_columns = {"model", "scenario_id", "variable", "prediction"}
    if not required_columns.issubset(existing.columns):
        return [], set()

    error_mask = (
        existing["error"].fillna("").astype(str).str.strip().ne("")
        if "error" in existing.columns
        else pd.Series(False, index=existing.index)
    )
    prediction_mask = existing["prediction"].notna()
    explanation_mask = (
        existing["explanation"].fillna("").astype(str).str.strip().ne("")
        if include_explanations and "explanation" in existing.columns
        else pd.Series(True, index=existing.index)
    )

    keep_mask = ~error_mask & prediction_mask & explanation_mask
    retained = existing.loc[keep_mask].copy()
    completed_keys = {
        (str(row.model), str(row.scenario_id), str(row.variable))
        for row in retained.itertuples()
    }
    return retained.to_dict("records"), completed_keys


def run_no_tools_eval(
    scenarios: list[Scenario],
    models: dict[str, str] | None = None,
    programs: list[str] | None = None,
    output_path: str | None = None,
    run_id: str | None = None,
    include_explanations: bool = False,
) -> pd.DataFrame:
    """Run the AI-alone evaluation across all models.

    If output_path is provided, saves incrementally every 100 rows.

    Returns DataFrame with columns:
        model, scenario_id, variable, prediction, raw_response, error,
        elapsed_seconds, prompt_tokens, completion_tokens, total_tokens,
        reasoning_tokens, cached_prompt_tokens, estimated_cost_usd
    """
    if models is None:
        models = MODELS
    if programs is None:
        programs = PROGRAMS

    resume_metadata = _build_resume_metadata(
        task="eval_no_tools_batch",
        scenarios=scenarios,
        models=models,
        programs=programs,
        run_id=run_id,
        include_explanations=include_explanations,
    )
    _validate_resume_metadata(output_path, resume_metadata)
    all_rows, completed = _load_existing_rows(
        output_path,
        programs,
        include_explanations=include_explanations,
    )
    total = len(models) * len(scenarios)
    done = len(completed)

    for model_name, model_id in models.items():
        model_fatal = False
        for scenario in scenarios:
            key = (model_name, scenario.id)
            if key in completed:
                continue
            try:
                result = run_single_no_tools(
                    scenario,
                    programs,
                    model_id,
                    include_explanations=include_explanations,
                )
                error = result.get("error")
            except Exception as e:
                error = _format_error(e)
                print(f"  ERROR [{model_name}] {scenario.id}: {error}")
                if isinstance(e, MODEL_FATAL_ERRORS):
                    model_fatal = True
                    print(
                        f"  Stopping {model_name} due to fatal error; "
                        "unattempted rows will be retried on resume."
                    )
                    if output_path:
                        _save_checkpoint(output_path, all_rows, resume_metadata)
                    break
                result = {
                    "predictions": {variable: None for variable in programs},
                    "explanations": {variable: None for variable in programs},
                    "raw_response": None,
                    "error": error,
                    "elapsed_seconds": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "reasoning_tokens": None,
                    "cached_prompt_tokens": None,
                    "provider_reported_cost_usd": None,
                    "reconstructed_cost_usd": None,
                    "total_cost_usd": None,
                    "cost_is_estimated": None,
                    "estimated_cost_usd": None,
                }

            batch_size = len(programs)
            call_id = ":".join(
                [part for part in [run_id, model_name, scenario.id] if part is not None]
            )
            elapsed_seconds = result.get("elapsed_seconds")
            prompt_tokens = result.get("prompt_tokens")
            completion_tokens = result.get("completion_tokens")
            total_tokens = result.get("total_tokens")
            reasoning_tokens = result.get("reasoning_tokens")
            cached_prompt_tokens = result.get("cached_prompt_tokens")
            provider_reported_cost_usd = result.get("provider_reported_cost_usd")
            reconstructed_cost_usd = result.get("reconstructed_cost_usd")
            total_cost_usd = result.get("total_cost_usd")
            cost_is_estimated = result.get("cost_is_estimated")
            estimated_cost_usd = result.get("estimated_cost_usd")
            for variable in programs:
                prediction = result["predictions"].get(variable)
                explanation = result.get("explanations", {}).get(variable)
                all_rows.append(
                    {
                        **({"run_id": run_id} if run_id is not None else {}),
                        "call_id": call_id,
                        "model": model_name,
                        "scenario_id": scenario.id,
                        "variable": variable,
                        "prediction": prediction,
                        "explanation": explanation,
                        "raw_response": result["raw_response"],
                        "error": error,
                        "elapsed_seconds": (
                            elapsed_seconds / batch_size
                            if elapsed_seconds is not None
                            else None
                        ),
                        "prompt_tokens": (
                            prompt_tokens / batch_size
                            if prompt_tokens is not None
                            else None
                        ),
                        "completion_tokens": (
                            completion_tokens / batch_size
                            if completion_tokens is not None
                            else None
                        ),
                        "total_tokens": (
                            total_tokens / batch_size
                            if total_tokens is not None
                            else None
                        ),
                        "reasoning_tokens": (
                            reasoning_tokens / batch_size
                            if reasoning_tokens is not None
                            else None
                        ),
                        "cached_prompt_tokens": (
                            cached_prompt_tokens / batch_size
                            if cached_prompt_tokens is not None
                            else None
                        ),
                        "provider_reported_cost_usd": (
                            provider_reported_cost_usd / batch_size
                            if provider_reported_cost_usd is not None
                            else None
                        ),
                        "reconstructed_cost_usd": (
                            reconstructed_cost_usd / batch_size
                            if reconstructed_cost_usd is not None
                            else None
                        ),
                        "total_cost_usd": (
                            total_cost_usd / batch_size
                            if total_cost_usd is not None
                            else None
                        ),
                        "cost_is_estimated": cost_is_estimated,
                        "estimated_cost_usd": (
                            estimated_cost_usd / batch_size
                            if estimated_cost_usd is not None
                            else None
                        ),
                    }
                )

            completed.add(key)
            done += 1
            if done % CHECKPOINT_EVERY_ROWS == 0:
                print(f"  Progress: {done}/{total} ({done * 100 // total}%)")
                if output_path:
                    _save_checkpoint(output_path, all_rows, resume_metadata)
            if model_fatal:
                break

    df = pd.DataFrame(all_rows)
    if output_path:
        _save_checkpoint(output_path, all_rows, resume_metadata)
    return df


def run_no_tools_single_output_eval(
    scenarios: list[Scenario],
    models: dict[str, str] | None = None,
    programs: list[str] | None = None,
    output_path: str | None = None,
    run_id: str | None = None,
    include_explanations: bool = False,
) -> pd.DataFrame:
    """Run AI-alone evaluation one output at a time.

    This is intended for diagnostic sidecars where each response should bind a
    numeric answer and explanation to a single requested variable.
    """
    if models is None:
        models = MODELS
    if programs is None:
        programs = PROGRAMS

    resume_metadata = _build_resume_metadata(
        task="eval_no_tools_single_output",
        scenarios=scenarios,
        models=models,
        programs=programs,
        run_id=run_id,
        include_explanations=include_explanations,
    )
    _validate_resume_metadata(output_path, resume_metadata)
    all_rows, completed = _load_existing_single_output_rows(
        output_path,
        include_explanations=include_explanations,
    )
    total = len(models) * len(scenarios) * len(programs)
    done = len(completed)

    for model_name, model_id in models.items():
        model_fatal = False
        for scenario in scenarios:
            for variable in programs:
                key = (model_name, scenario.id, variable)
                if key in completed:
                    continue
                try:
                    result = run_single_no_tools(
                        scenario,
                        variable,
                        model_id,
                        include_explanations=include_explanations,
                    )
                    error = result.get("error")
                except Exception as e:
                    error = _format_error(e)
                    print(f"  ERROR [{model_name}] {scenario.id} {variable}: {error}")
                    if isinstance(e, MODEL_FATAL_ERRORS):
                        model_fatal = True
                        print(
                            f"  Stopping {model_name} due to fatal error; "
                            "unattempted rows will be retried on resume."
                        )
                        if output_path:
                            _save_checkpoint(output_path, all_rows, resume_metadata)
                        break
                    result = {
                        "predictions": {variable: None},
                        "explanations": {variable: None},
                        "raw_response": None,
                        "error": error,
                        "elapsed_seconds": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "reasoning_tokens": None,
                        "cached_prompt_tokens": None,
                        "provider_reported_cost_usd": None,
                        "reconstructed_cost_usd": None,
                        "total_cost_usd": None,
                        "cost_is_estimated": None,
                        "estimated_cost_usd": None,
                    }

                call_id = ":".join(
                    [
                        part
                        for part in [run_id, model_name, scenario.id, variable]
                        if part is not None
                    ]
                )
                all_rows.append(
                    {
                        **({"run_id": run_id} if run_id is not None else {}),
                        "call_id": call_id,
                        "model": model_name,
                        "scenario_id": scenario.id,
                        "variable": variable,
                        "prediction": result["predictions"].get(variable),
                        "explanation": result.get("explanations", {}).get(variable),
                        "raw_response": result["raw_response"],
                        "error": error,
                        "elapsed_seconds": result.get("elapsed_seconds"),
                        "prompt_tokens": result.get("prompt_tokens"),
                        "completion_tokens": result.get("completion_tokens"),
                        "total_tokens": result.get("total_tokens"),
                        "reasoning_tokens": result.get("reasoning_tokens"),
                        "cached_prompt_tokens": result.get("cached_prompt_tokens"),
                        "provider_reported_cost_usd": result.get(
                            "provider_reported_cost_usd"
                        ),
                        "reconstructed_cost_usd": result.get("reconstructed_cost_usd"),
                        "total_cost_usd": result.get("total_cost_usd"),
                        "cost_is_estimated": result.get("cost_is_estimated"),
                        "estimated_cost_usd": result.get("estimated_cost_usd"),
                    }
                )

                completed.add(key)
                done += 1
                if done % CHECKPOINT_EVERY_ROWS == 0:
                    print(f"  Progress: {done}/{total} ({done * 100 // total}%)")
                    if output_path:
                        _save_checkpoint(output_path, all_rows, resume_metadata)

            if model_fatal:
                break
        if model_fatal:
            continue

    df = pd.DataFrame(all_rows)
    if output_path:
        _save_checkpoint(output_path, all_rows, resume_metadata)
    return df


def run_repeated_no_tools_eval(
    scenarios: list[Scenario],
    repeats: int,
    output_dir: str,
    models: dict[str, str] | None = None,
    programs: list[str] | None = None,
    include_explanations: bool = False,
    single_output: bool = False,
) -> pd.DataFrame:
    """Run repeated AI-alone evaluations, saving one artifact per run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frames = []
    for run_index in range(repeats):
        run_id = f"run_{run_index:03d}"
        run_output_path = output_path / f"{run_id}.csv"
        print(f"=== Starting {run_id} ===")
        runner = run_no_tools_single_output_eval if single_output else run_no_tools_eval
        frame = runner(
            scenarios,
            models=models,
            programs=programs,
            output_path=str(run_output_path),
            run_id=run_id,
            include_explanations=include_explanations,
        )
        if "run_id" not in frame.columns:
            frame = frame.copy()
            frame["run_id"] = run_id
            frame.to_csv(run_output_path, index=False)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_repeated_predictions(runs_dir: str) -> pd.DataFrame:
    """Load per-run prediction CSVs from a directory into one DataFrame."""
    runs_path = Path(runs_dir)
    files = sorted(runs_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No run CSVs found in {runs_path}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "run_id" not in frame.columns:
            frame = frame.copy()
            frame["run_id"] = path.stem
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
