from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import json
import os
import re

import requests
from openai import OpenAI

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

try:
    import cohere
except ImportError:  # pragma: no cover - optional dependency
    cohere = None

try:
    from together import Together
except ImportError:  # pragma: no cover - optional dependency
    Together = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None

try:
    from .prompts import (
        ATLA_PROMPT,
        ATLA_PROMPT_WITH_REFERENCE,
        FLOW_JUDGE_PROMPT,
        JUDGE_SYSTEM_PROMPT,
        PROMETHEUS_PROMPT,
        PROMETHEUS_PROMPT_WITH_REFERENCE,
    )
except ImportError:  # pragma: no cover - script-style fallback for the vendored app
    from prompts import (  # type: ignore[no-redef]
        ATLA_PROMPT,
        ATLA_PROMPT_WITH_REFERENCE,
        FLOW_JUDGE_PROMPT,
        JUDGE_SYSTEM_PROMPT,
        PROMETHEUS_PROMPT,
        PROMETHEUS_PROMPT_WITH_REFERENCE,
    )

if TYPE_CHECKING:
    from atla import Atla as AtlaClient
else:
    AtlaClient = Any

try:
    from atla import Atla as _AtlaRuntimeClient
except ImportError:  # pragma: no cover - optional dependency
    _AtlaRuntimeClient = None

hf_api_key = os.getenv("HF_API_KEY")
flow_judge_api_key = os.getenv("FLOW_JUDGE_API_KEY")
salesforce_api_key = os.getenv("SALESFORCE_API_KEY")

_anthropic_client: Any | None = None
_atla_client: AtlaClient | None = None
_cohere_client: Any | None = None
_openai_clients: dict[tuple[str | None, str, float | None], OpenAI] = {}
_together_client: Any | None = None


def _resolve_env_placeholder(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("$ENV{") and value.endswith("}"):
        return os.getenv(value[5:-1], "")
    return value


def _get_openai_client(model_info: dict[str, Any] | None = None) -> OpenAI:
    resolved_info = model_info or {}
    raw_base_url = _resolve_env_placeholder(resolved_info.get("base_url"))
    base_url = str(raw_base_url).strip() if raw_base_url else None

    raw_api_key = _resolve_env_placeholder(resolved_info.get("api_key"))
    api_key_env = resolved_info.get("api_key_env")
    if not raw_api_key and isinstance(api_key_env, str) and api_key_env:
        raw_api_key = os.getenv(api_key_env, "")
    api_key = str(raw_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY")

    raw_timeout = resolved_info.get("request_timeout_seconds")
    timeout = None if raw_timeout in (None, "") else float(raw_timeout)

    cache_key = (base_url, api_key, timeout)
    client = _openai_clients.get(cache_key)
    if client is not None:
        return client

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout

    client = OpenAI(**client_kwargs)
    _openai_clients[cache_key] = client
    return client


def _get_anthropic_client() -> Any:
    global _anthropic_client
    if _anthropic_client is None:
        if anthropic is None:
            raise RuntimeError(
                "Anthropic support requires the optional `anthropic` package."
            )
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _get_together_client() -> Any:
    global _together_client
    if _together_client is None:
        if Together is None:
            raise RuntimeError(
                "Together support requires the optional `together` package."
            )
        _together_client = Together()
    return _together_client


def _get_cohere_client() -> Any:
    global _cohere_client
    if _cohere_client is None:
        if cohere is None:
            raise RuntimeError("Cohere support requires the optional `cohere` package.")
        _cohere_client = cohere.ClientV2(os.getenv("CO_API_KEY"))
    return _cohere_client


def _get_atla_client() -> AtlaClient:
    global _atla_client
    if _atla_client is None:
        if _AtlaRuntimeClient is None:
            raise RuntimeError(
                "Atla support requires the optional `atla` package to be installed."
            )
        _atla_client = _AtlaRuntimeClient()
    return _atla_client


def _coerce_stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                continue
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
        return "".join(parts)
    text = getattr(value, "text", None)
    if text is not None:
        return str(text)
    return str(value)


def _get_stream_field(payload: Any, *field_names: str) -> str:
    model_extra = getattr(payload, "model_extra", None)
    for field_name in field_names:
        value = None
        if isinstance(payload, dict):
            value = payload.get(field_name)
        else:
            value = getattr(payload, field_name, None)
        if value is None and isinstance(model_extra, dict):
            value = model_extra.get(field_name)
        text = _coerce_stream_text(value)
        if text:
            return text
    return ""


def _build_openai_request_kwargs(
    model_name: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int | None,
    temperature: float,
    model_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_info = model_info or {}
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    if max_tokens is not None:
        if resolved_info.get("base_url"):
            request_kwargs["max_tokens"] = max_tokens
        else:
            request_kwargs["max_completion_tokens"] = max_tokens

    extra_body = _resolve_env_placeholder(
        resolved_info.get("extra_body") or resolved_info.get("thinking_extra_body")
    )
    if isinstance(extra_body, dict):
        request_kwargs["extra_body"] = extra_body

    return request_kwargs


def _build_stream_update(
    *,
    raw_response: Any,
    content: str,
    reasoning_content: str,
) -> dict[str, Any]:
    return {
        "raw_response": raw_response,
        "content": content,
        "reasoning_content": reasoning_content,
    }


def _prepare_model_prompt(
    model_info: dict[str, Any] | None,
    prompt_data: dict[str, Any],
    *,
    use_reference: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    if not model_info:
        return None, "Model not found or unsupported."

    api_model = model_info["api_model"]
    organization = model_info["organization"]

    is_prometheus = organization == "Prometheus"
    is_atla = organization == "Atla"
    is_flow_judge = organization == "Flow AI"
    is_salesforce = organization == "Salesforce"

    system_prompt = (
        None
        if (is_prometheus or is_atla or is_flow_judge or is_salesforce)
        else JUDGE_SYSTEM_PROMPT
    )

    if is_atla or is_salesforce:
        base_prompt = ATLA_PROMPT_WITH_REFERENCE if use_reference else ATLA_PROMPT
    elif is_flow_judge:
        base_prompt = FLOW_JUDGE_PROMPT
    else:
        base_prompt = (
            PROMETHEUS_PROMPT_WITH_REFERENCE if use_reference else PROMETHEUS_PROMPT
        )

    if not (is_prometheus or is_atla or is_flow_judge or is_salesforce):
        base_prompt = base_prompt.replace(
            '3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"',
            '3. Your output format should strictly adhere to JSON as follows: {{"feedback": "<write feedback>", "result": <numerical score>}}. Ensure the output is valid JSON, without additional formatting or explanations.',
        )

    try:
        if not is_flow_judge:
            final_prompt = base_prompt.format(
                human_input=prompt_data["human_input"],
                ai_response=prompt_data["ai_response"],
                ground_truth_input=prompt_data.get("ground_truth_input", ""),
                eval_criteria=prompt_data["eval_criteria"],
                score1_desc=prompt_data["score1_desc"],
                score2_desc=prompt_data["score2_desc"],
                score3_desc=prompt_data["score3_desc"],
                score4_desc=prompt_data["score4_desc"],
                score5_desc=prompt_data["score5_desc"],
            )
        else:
            human_input = f"<user_input>\n{prompt_data['human_input']}\n</user_input>"
            ai_response = f"<response>\n{prompt_data['ai_response']}\n</response>"
            ground_truth = prompt_data.get("ground_truth_input", "")
            response_reference = (
                f"<response_reference>\n{ground_truth}\n</response_reference>"
                if ground_truth
                else ""
            )
            rubric = "".join(
                [
                    f"- Score 1: {prompt_data['score1_desc']}\n",
                    f"- Score 2: {prompt_data['score2_desc']}\n",
                    f"- Score 3: {prompt_data['score3_desc']}\n",
                    f"- Score 4: {prompt_data['score4_desc']}\n",
                    f"- Score 5: {prompt_data['score5_desc']}",
                ]
            )
            inputs = (
                f"{human_input}\n{response_reference}"
                if response_reference
                else human_input
            )
            final_prompt = base_prompt.format(
                INPUTS=inputs,
                OUTPUT=ai_response,
                EVALUATION_CRITERIA=prompt_data["eval_criteria"],
                RUBRIC=rubric,
            )
    except KeyError as exc:
        return None, f"Error formatting prompt: Missing required field {str(exc)}"

    return {
        "api_model": api_model,
        "organization": organization,
        "system_prompt": system_prompt,
        "final_prompt": final_prompt,
    }, None


def get_openai_response(
    model_name,
    prompt,
    model_info=None,
    system_prompt=JUDGE_SYSTEM_PROMPT,
    max_tokens: int | None = None,
    temperature=0,
):
    """Get response from OpenAI API"""
    try:
        response = _get_openai_client(model_info).chat.completions.create(
            **_build_openai_request_kwargs(
                model_name,
                prompt,
                system_prompt,
                max_tokens,
                temperature,
                model_info,
            )
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI model {model_name}: {str(e)}"


def stream_openai_response(
    model_name,
    prompt,
    model_info=None,
    system_prompt=JUDGE_SYSTEM_PROMPT,
    max_tokens: int | None = None,
    temperature=0,
) -> Iterator[dict[str, Any]]:
    """Stream response tokens from an OpenAI-compatible endpoint."""
    full_content = ""
    full_reasoning = ""
    try:
        response = _get_openai_client(model_info).chat.completions.create(
            **_build_openai_request_kwargs(
                model_name,
                prompt,
                system_prompt,
                max_tokens,
                temperature,
                model_info,
            ),
            stream=True,
        )
        for chunk in response:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            reasoning_chunk = _get_stream_field(delta, "reasoning_content", "reasoning")
            content_chunk = _get_stream_field(delta, "content")
            if reasoning_chunk:
                full_reasoning += reasoning_chunk
            if content_chunk:
                full_content += content_chunk
            if reasoning_chunk or content_chunk:
                yield _build_stream_update(
                    raw_response=full_content or full_reasoning,
                    content=full_content,
                    reasoning_content=full_reasoning,
                )
    except Exception:
        fallback_response = get_openai_response(
            model_name,
            prompt,
            model_info,
            system_prompt,
            max_tokens,
            temperature,
        )
        fallback_text = (
            fallback_response
            if isinstance(fallback_response, str)
            else str(fallback_response)
        )
        yield _build_stream_update(
            raw_response=fallback_response,
            content=fallback_text,
            reasoning_content="",
        )


def get_anthropic_response(
    model_name,
    prompt,
    system_prompt=JUDGE_SYSTEM_PROMPT,
    max_tokens: int | None = None,
    temperature=0,
):
    """Get response from Anthropic API"""
    try:
        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        response = _get_anthropic_client().messages.create(**request_kwargs)
        return response.content[0].text
    except Exception as e:
        return f"Error with Anthropic model {model_name}: {str(e)}"


def get_together_response(
    model_name,
    prompt,
    system_prompt=JUDGE_SYSTEM_PROMPT,
    max_tokens: int | None = None,
    temperature=0,
):
    """Get response from Together API"""
    try:
        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        response = _get_together_client().chat.completions.create(**request_kwargs)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with Together model {model_name}: {str(e)}"


def stream_together_response(
    model_name,
    prompt,
    system_prompt=JUDGE_SYSTEM_PROMPT,
    max_tokens: int | None = None,
    temperature=0,
) -> Iterator[dict[str, Any]]:
    """Stream response tokens from Together when available."""
    full_content = ""
    full_reasoning = ""
    try:
        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        response = _get_together_client().chat.completions.create(**request_kwargs)
        for chunk in response:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            reasoning_chunk = _get_stream_field(delta, "reasoning_content", "reasoning")
            content_chunk = _get_stream_field(delta, "content")
            if reasoning_chunk:
                full_reasoning += reasoning_chunk
            if content_chunk:
                full_content += content_chunk
            if reasoning_chunk or content_chunk:
                yield _build_stream_update(
                    raw_response=full_content or full_reasoning,
                    content=full_content,
                    reasoning_content=full_reasoning,
                )
    except Exception:
        fallback_response = get_together_response(
            model_name,
            prompt,
            system_prompt,
            max_tokens,
            temperature,
        )
        fallback_text = (
            fallback_response
            if isinstance(fallback_response, str)
            else str(fallback_response)
        )
        yield _build_stream_update(
            raw_response=fallback_response,
            content=fallback_text,
            reasoning_content="",
        )


def get_prometheus_response(
    model_name,
    prompt,
    system_prompt=None,
    max_tokens: int | None = None,
    temperature=0.01,
):
    """Get response from Hugging Face model"""
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json",
        }

        # Create messages list for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        model_id = "prometheus-eval/prometheus-7b-v2.0"
        if AutoTokenizer is None:
            raise RuntimeError(
                "Prometheus support requires the optional `transformers` package."
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_api_key)
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens if max_tokens is not None else 2048,
                "return_full_text": False,
                "temperature": temperature,
            },
        }

        response = requests.post(
            "https://otb7jglxy6r37af6.us-east-1.aws.endpoints.huggingface.cloud",
            headers=headers,
            json=payload,
        )
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"Error with Hugging Face model {model_name}: {str(e)}"


def get_atla_response(
    model_name,
    prompt,
    system_prompt=None,
    max_tokens: int | None = None,
    temperature=0.01,
):
    """Get response from Atla API"""
    try:
        # Extract components from the prompt data
        model_input = prompt.get("human_input", "")
        model_output = prompt.get("ai_response", "")
        expected_output = prompt.get("ground_truth_input", "")
        evaluation_criteria = prompt.get("eval_criteria", "")

        # Set model_id based on the model name
        if "Mini" in model_name:
            model_id = "atla-selene-mini"
        else:
            model_id = "atla-selene"

        response = _get_atla_client().evaluation.create(
            model_id=model_id,
            model_input=model_input,
            model_output=model_output,
            expected_model_output=expected_output if expected_output else None,
            evaluation_criteria=evaluation_criteria,
        )

        # Return the score and critique directly
        return {
            "score": response.result.evaluation.score,
            "critique": response.result.evaluation.critique,
        }
    except Exception as e:
        return f"Error with Atla model {model_name}: {str(e)}"


def get_flow_judge_response(
    model_name,
    prompt,
    max_tokens: int | None = None,
    temperature=0.1,
    top_p=0.95,
) -> str:
    """Get response from Flow Judge"""
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "stop": None,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            "https://arena.flow-ai.io/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {flow_judge_api_key}",
            },
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error with Flow Judge completions model {model_name}: {str(e)}"


def get_cohere_response(
    model_name,
    prompt,
    system_prompt=JUDGE_SYSTEM_PROMPT,
    max_tokens: int | None = None,
    temperature=0,
):
    """Get response from Cohere API"""
    try:
        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        response = _get_cohere_client().chat(**request_kwargs)
        # Extract the text from the content items
        content_items = response.message.content
        if isinstance(content_items, list):
            # Get the text from the first content item
            return content_items[0].text
        return str(content_items)  # Fallback if it's not a list
    except Exception as e:
        return f"Error with Cohere model {model_name}: {str(e)}"


def get_salesforce_response(
    model_name,
    prompt,
    system_prompt=None,
    max_tokens: int | None = None,
    temperature=0,
):
    """Get response from Salesforce Research API"""
    try:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-Api-Key": salesforce_api_key,
        }

        # Create messages list
        messages = []
        messages.append({"role": "user", "content": prompt})

        json_data = {
            "prompts": messages,
            "temperature": temperature,
            "top_p": 1,
        }
        if max_tokens is not None:
            json_data["max_tokens"] = max_tokens

        response = requests.post(
            "https://gateway.salesforceresearch.ai/sfr-judge/process",
            headers=headers,
            json=json_data,
        )
        response.raise_for_status()
        return response.json()["result"][0]
    except Exception as e:
        return f"Error with Salesforce model {model_name}: {str(e)}"


def get_model_response(
    model_name,
    model_info,
    prompt_data,
    use_reference=False,
    max_tokens: int | None = None,
    temperature=0,
):
    """Get response from appropriate API based on model organization"""
    prepared_prompt, error = _prepare_model_prompt(
        model_info,
        prompt_data,
        use_reference=use_reference,
    )
    if error:
        return error

    assert prepared_prompt is not None
    api_model = prepared_prompt["api_model"]
    organization = prepared_prompt["organization"]
    system_prompt = prepared_prompt["system_prompt"]
    final_prompt = prepared_prompt["final_prompt"]

    try:
        if organization in {"OpenAI", "OpenAI-Compatible", "vLLM"}:
            return get_openai_response(
                api_model,
                final_prompt,
                model_info,
                system_prompt,
                max_tokens,
                temperature,
            )
        elif organization == "Anthropic":
            return get_anthropic_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature
            )
        elif organization == "Prometheus":
            return get_prometheus_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature=0.01
            )
        elif organization == "Atla":
            response = get_atla_response(
                api_model, prompt_data, system_prompt, max_tokens, temperature
            )
            # Response now contains score and critique directly
            if (
                isinstance(response, dict)
                and "score" in response
                and "critique" in response
            ):
                score = str(response["score"])
                critique = response["critique"]
                return score, critique
            else:
                return "Error", str(response)
        elif organization == "Cohere":
            return get_cohere_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature
            )
        elif organization == "Flow AI":
            return get_flow_judge_response(api_model, final_prompt)
        elif organization == "Salesforce":
            response = get_salesforce_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature
            )
            return response
        else:
            # All other organizations use Together API
            return get_together_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature
            )
    except Exception as e:
        return f"Error with {organization} model {model_name}: {str(e)}"


def stream_model_response(
    model_name,
    model_info,
    prompt_data,
    use_reference=False,
    max_tokens: int | None = None,
    temperature=0,
) -> Iterator[dict[str, Any]]:
    """Stream model updates when the backend supports token streaming."""
    prepared_prompt, error = _prepare_model_prompt(
        model_info,
        prompt_data,
        use_reference=use_reference,
    )
    if error:
        yield _build_stream_update(
            raw_response=error,
            content=error,
            reasoning_content="",
        )
        return

    assert prepared_prompt is not None
    api_model = prepared_prompt["api_model"]
    organization = prepared_prompt["organization"]
    system_prompt = prepared_prompt["system_prompt"]
    final_prompt = prepared_prompt["final_prompt"]

    if organization in {"OpenAI", "OpenAI-Compatible", "vLLM"}:
        yield from stream_openai_response(
            api_model,
            final_prompt,
            model_info,
            system_prompt,
            max_tokens,
            temperature,
        )
        return

    if organization not in {
        "Anthropic",
        "Prometheus",
        "Atla",
        "Cohere",
        "Flow AI",
        "Salesforce",
    }:
        yield from stream_together_response(
            api_model,
            final_prompt,
            system_prompt,
            max_tokens,
            temperature,
        )
        return

    response = get_model_response(
        model_name,
        model_info,
        prompt_data,
        use_reference=use_reference,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    preview = response if isinstance(response, str) else str(response)
    yield _build_stream_update(
        raw_response=response,
        content=preview,
        reasoning_content="",
    )


def parse_model_response(response):
    try:
        # Debug print
        print(f"Raw model response: {response}")

        # If response is already a tuple (from Atla/Salesforce), use it directly
        if isinstance(response, tuple):
            return response

        # If response is already a dictionary, use it directly
        if isinstance(response, dict):
            return str(response.get("result", "N/A")), response.get("feedback", "N/A")

        # First try to parse the entire response as JSON
        try:
            data = json.loads(response)
            return str(data.get("result", "N/A")), data.get("feedback", "N/A")
        except json.JSONDecodeError:
            # If that fails, check if this is a Salesforce response
            if "**Reasoning:**" in response or "**Result:**" in response:
                # Use ATLA parser for Salesforce responses only
                return salesforce_parse_model_response(response)

            # Otherwise try to find JSON within the response
            json_match = re.search(r"{.*}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return str(data.get("result", "N/A")), data.get("feedback", "N/A")
            else:
                return (
                    "Error",
                    f"Invalid response format returned - here is the raw model response: {response}",
                )

    except Exception as e:
        # Debug print for error case
        print(f"Failed to parse response: {str(e)}")

        # If the error message itself contains valid JSON, try to parse that
        try:
            error_json_match = re.search(r"{.*}", str(e), re.DOTALL)
            if error_json_match:
                data = json.loads(error_json_match.group(0))
                return str(data.get("result", "N/A")), data.get("feedback", "N/A")
        except Exception:
            pass

        return "Error", f"Failed to parse response: {response}"


def prometheus_parse_model_response(output):
    try:
        print(f"Raw model response: {output}")
        output = output.strip()

        # Remove "Feedback:" prefix if present (case insensitive)
        output = re.sub(r"^feedback:\s*", "", output, flags=re.IGNORECASE)

        # New pattern to match [RESULT] X at the beginning
        begin_result_pattern = r"^\[RESULT\]\s*(\d+)\s*\n*(.*?)$"
        begin_match = re.search(begin_result_pattern, output, re.DOTALL | re.IGNORECASE)
        if begin_match:
            score = int(begin_match.group(1))
            feedback = begin_match.group(2).strip()
            return str(score), feedback

        # Existing patterns for end-of-string results...
        pattern = r"(.*?)\s*\[RESULT\]\s*[\(\[]?(\d+)[\)\]]?"
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        if match:
            feedback = match.group(1).strip()
            score = int(match.group(2))
            return str(score), feedback

        # If no match, try to match "... Score: X"
        pattern = r"(.*?)\s*(?:Score|Result)\s*:\s*[\(\[]?(\d+)[\)\]]?"
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        if match:
            feedback = match.group(1).strip()
            score = int(match.group(2))
            return str(score), feedback

        # Pattern to handle [Score X] at the end
        pattern = r"(.*?)\s*\[(?:Score|Result)\s*[\(\[]?(\d+)[\)\]]?\]$"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            feedback = match.group(1).strip()
            score = int(match.group(2))
            return str(score), feedback

        # Final fallback attempt
        pattern = r"[\(\[]?(\d+)[\)\]]?\s*\]?$"
        match = re.search(pattern, output)
        if match:
            score = int(match.group(1))
            feedback = output[: match.start()].rstrip()
            # Remove any trailing brackets from feedback
            feedback = re.sub(r"\s*\[[^\]]*$", "", feedback).strip()
            return str(score), feedback

        return "Error", f"Failed to parse response: {output}"

    except Exception as e:
        print(f"Failed to parse response: {str(e)}")
        return "Error", f"Exception during parsing: {str(e)}"


def salesforce_parse_model_response(output):
    """Parse response from Salesforce model"""
    try:
        print(f"Raw Salesforce model response: {output}")
        output = output.strip()

        # Look for the Reasoning and Result sections
        reasoning_match = re.search(
            r"\*\*Reasoning:\*\*(.*?)(?=\*\*Result:|$)", output, re.DOTALL
        )
        result_match = re.search(r"\*\*Result:\*\*\s*(\d+)", output)

        if reasoning_match and result_match:
            feedback = reasoning_match.group(1).strip()
            score = result_match.group(1)
            return str(score), feedback

        return "Error", f"Failed to parse Salesforce response format: {output}"

    except Exception as e:
        print(f"Failed to parse Salesforce response: {str(e)}")
        return "Error", f"Exception during parsing: {str(e)}"


def flow_judge_parse_model_response(output):
    try:
        print(f"Raw model response: {output}")
        # Convert multiple line breaks to single ones and strip whitespace
        output = re.sub(r"\n{2,}", "\n", output.strip())

        # Compile regex patterns
        feedback_pattern = re.compile(r"<feedback>\s*(.*?)\s*</feedback>", re.DOTALL)
        score_pattern = re.compile(r"<score>\s*(\d+)\s*</score>", re.DOTALL)

        feedback_match = feedback_pattern.search(output)
        score_match = score_pattern.search(output)

        if feedback_match and score_match:
            feedback = feedback_match.group(1).strip()
            score = int(score_match.group(1).strip())
            return str(score), feedback

        return "Error", f"Failed to parse response: {output}"

    except Exception as e:
        print(f"Failed to parse response: {str(e)}")
        return "Error", f"Exception during parsing: {str(e)}"
