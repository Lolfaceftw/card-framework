"""CARD Script Summarizer with DeepSeek and transcript-grounded speaker validation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Literal, TypedDict, cast

from openai import OpenAI
from openai import APIError as OpenAIAPIError
from pydantic import BaseModel, Field, ValidationError

from .logging_utils import configure_logging
from .speaker_validation import (
    DialogueLinePayload,
    ValidatedDialogueLine,
    build_segment_speaker_map,
    collect_allowed_speakers,
    format_segments_for_prompt,
    load_transcript_segments,
    validate_and_repair_dialogue,
)

DEEPSEEK_STABLE_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_BETA_BASE_URL = "https://api.deepseek.com/beta"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_RETRIES_DEFAULT = 2
DEFAULT_MAX_COMPLETION_TOKENS = 8192
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_HTTP_RETRIES = 1
DEFAULT_TEMPERATURE = 0.2
SUMMARY_LINE_COUNT_MIN = 12
SUMMARY_LINE_COUNT_MAX = 60
ERROR_DIGEST_MAX_CHARS = 280

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ErrorType = Literal[
    "api_error",
    "malformed_json",
    "truncated_json",
    "schema_validation",
    "empty_response",
]
EndpointMode = Literal["beta", "stable"]


class DialogueLine(BaseModel):
    """Represent one model-generated dialogue line."""

    speaker: str = Field(..., description="Speaker label from the allowed transcript speakers.")
    text: str = Field(..., description="Natural dialogue text, about 10-15 seconds of speech.")
    emo_text: str = Field(..., description="Short emotion/tone description.")
    emo_alpha: float = Field(0.6, description="Emotion intensity, typically 0.5 to 0.9.")
    source_segment_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Transcript segment IDs used as evidence for this line.",
    )


class PodcastScript(BaseModel):
    """Represent model response payload."""

    dialogue: list[DialogueLine]


class FinalScriptLine(TypedDict):
    """Represent saved output line consumed by the downstream pipeline."""

    speaker: str
    voice_sample: str
    use_emo_text: bool
    emo_text: str
    emo_alpha: float
    text: str
    source_segment_ids: list[str]
    validation_status: str
    repair_reason: str | None


@dataclass(slots=True, frozen=True)
class RetryContext:
    """Represent context from the previous failed generation attempt."""

    attempt_index: int
    endpoint_mode: EndpointMode
    error_type: ErrorType
    error_digest: str


@dataclass(slots=True, frozen=True)
class DeepSeekRequestSettings:
    """Represent tunable DeepSeek request settings."""

    model: str
    max_completion_tokens: int
    request_timeout_seconds: float
    http_retries: int
    temperature: float
    auto_beta: bool = True


@dataclass(slots=True, frozen=True)
class GenerationSuccess:
    """Represent a successful structured generation result."""

    script: PodcastScript
    endpoint_mode: EndpointMode
    used_json_repair: bool


@dataclass(slots=True, frozen=True)
class GenerationAttemptError(Exception):
    """Wrap a generation failure with endpoint attribution."""

    endpoint_mode: EndpointMode
    cause: Exception


def _error_digest(error: Exception) -> str:
    """Create a compact, single-line error digest for logs and retry prompts."""
    digest = " ".join(str(error).split())
    return digest[:ERROR_DIGEST_MAX_CHARS]


def _classify_json_error(error: JSONDecodeError) -> ErrorType:
    """Classify JSON decode failures as malformed vs likely truncated."""
    lowered_message = error.msg.lower()
    if "unterminated" in lowered_message or "expecting value" in lowered_message:
        return "truncated_json"
    if error.pos >= max(0, len(error.doc) - 2):
        return "truncated_json"
    return "malformed_json"


def _build_system_prompt(
    allowed_speakers: set[str],
    target_min_lines: int,
    target_max_lines: int,
) -> str:
    """Build a constrained prompt with explicit objective and acceptance checks.

    Args:
        allowed_speakers: Speaker labels allowed in the response.
        target_min_lines: Lower bound of desired output dialogue line count.
        target_max_lines: Upper bound of desired output dialogue line count.

    Returns:
        Prompt string for the LLM.
    """
    speaker_list = ", ".join(sorted(allowed_speakers))
    sample_speaker = sorted(allowed_speakers)[0]
    return f"""Objective:
- Rewrite the transcript into concise, engaging podcast dialogue while preserving factual meaning.

Context:
- This is for CARD (Constraint-aware Audio Resynthesis).
- Transcript lines include evidence metadata in the form [segment_id|speaker]: text.
- Allowed speakers are exactly: {speaker_list}

Inputs:
- A transcript where each line has:
  - segment_id: unique identifier (example: seg_00012)
  - speaker: one allowed speaker label
  - text: source utterance

Output contract:
- Return a JSON object with shape:
  {{
    "dialogue": [
      {{
        "speaker": "ONE_ALLOWED_SPEAKER",
        "text": "spoken dialogue line",
        "emo_text": "brief emotional style",
        "emo_alpha": 0.6,
        "source_segment_ids": ["seg_00012", "seg_00013"]
      }}
    ]
  }}
- Output STRICT JSON ONLY: no markdown fences, no prose before/after JSON.
- The response must end with a fully closed JSON object.
- Each line must include non-empty source_segment_ids.
- Target {target_min_lines} to {target_max_lines} dialogue lines total.
- Keep each line concise (prefer <= 220 characters) and about 10-15 seconds of speech.

Rules:
- Never invent, rename, or merge speaker labels.
- speaker must be exactly one of the allowed speakers.
- Every source_segment_id must come from the input transcript.
- All source_segment_ids for a single line must belong to the same speaker.
- Avoid meta-summary phrases like "the speaker discusses".
- Preserve key claims and tone.

Examples:
- Input excerpt:
  [seg_00000|{sample_speaker}]: Welcome back to the show.
  [seg_00001|{sample_speaker}]: Today we are breaking down the market shift.
- Valid output excerpt:
  {{
    "dialogue": [
      {{
        "speaker": "{sample_speaker}",
        "text": "Welcome back. Today we're breaking down the market shift.",
        "emo_text": "Warm and focused",
        "emo_alpha": 0.65,
        "source_segment_ids": ["seg_00000", "seg_00001"]
      }}
    ]
  }}

Evaluation:
- Before answering, verify:
  1) JSON is valid and matches schema exactly.
  2) Each line has non-empty source_segment_ids.
  3) Each line uses only allowed speakers.
  4) Each line's source_segment_ids belong to one speaker only.
  5) Output is only JSON and is complete (not truncated).
"""


def _build_retry_instruction(context: RetryContext) -> str:
    """Build retry guidance that explains the previous failure."""
    return (
        "Retry correction request:\n"
        f"- Previous attempt: {context.attempt_index}\n"
        f"- Previous endpoint: {context.endpoint_mode}\n"
        f"- Failure type: {context.error_type}\n"
        f"- Failure detail: {context.error_digest}\n"
        "- Fix the exact issue and regenerate a complete STRICT JSON object only.\n"
        "- Do not include markdown, explanations, or trailing text.\n"
    )


def _strip_markdown_fence(content: str) -> str:
    """Strip a top-level Markdown code fence around a JSON blob when present."""
    stripped = content.strip()
    if not stripped.startswith("```"):
        return content

    lines = stripped.splitlines()
    if len(lines) < 3:
        return content

    opening = lines[0].strip().lower()
    closing = lines[-1].strip()
    if not opening.startswith("```") or closing != "```":
        return content
    return "\n".join(lines[1:-1]).strip()


def _extract_first_json_object(content: str) -> str | None:
    """Extract the first balanced JSON object from arbitrary text."""
    start = content.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    return None


def _decode_podcast_script(raw_content: str) -> tuple[PodcastScript, bool]:
    """Decode and validate LLM output with one balanced repair pass.

    Args:
        raw_content: Raw content returned from DeepSeek.

    Returns:
        Tuple of validated `PodcastScript` and whether local repair was used.

    Raises:
        JSONDecodeError: If JSON parsing fails after recovery attempt.
        ValidationError: If parsed payload fails schema validation.
        ValueError: If no parseable JSON object exists in content.
    """
    direct_payload = json.loads(raw_content)
    try:
        return PodcastScript(**direct_payload), False
    except ValidationError:
        # Schema errors are surfaced to caller as-is.
        raise


def _decode_podcast_script_with_fallback(raw_content: str) -> tuple[PodcastScript, bool]:
    """Decode and validate LLM output using balanced JSON recovery."""
    try:
        return _decode_podcast_script(raw_content)
    except JSONDecodeError as original_error:
        stripped_content = _strip_markdown_fence(raw_content)
        candidate = _extract_first_json_object(stripped_content)
        if not candidate:
            raise original_error
        repaired_payload = json.loads(candidate)
        return PodcastScript(**repaired_payload), True


def _looks_like_beta_endpoint_error(error: Exception) -> bool:
    """Return True when an exception likely indicates beta endpoint incompatibility."""
    if not isinstance(error, OpenAIAPIError):
        return False

    status_code = getattr(error, "status_code", None)
    lowered = str(error).lower()
    endpoint_keywords = ("beta", "not found", "unsupported", "unknown path", "invalid url")

    if status_code == 404:
        return True
    if status_code in {400, 405, 422} and any(keyword in lowered for keyword in endpoint_keywords):
        return True
    return False


def _build_deepseek_client(
    api_key: str,
    base_url: str,
    request_timeout_seconds: float,
    http_retries: int,
) -> OpenAI:
    """Create a DeepSeek client with request timeout and transport retries."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=request_timeout_seconds,
        max_retries=http_retries,
    )


def _request_deepseek_completion(
    client: OpenAI,
    settings: DeepSeekRequestSettings,
    transcript_text: str,
    system_prompt: str,
    retry_context: RetryContext | None,
    endpoint_mode: EndpointMode,
) -> tuple[PodcastScript, bool]:
    """Request completion from a specific endpoint and parse JSON output."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if retry_context is not None:
        messages.append(
            {
                "role": "system",
                "content": _build_retry_instruction(retry_context),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": f"Here is the raw transcript:\n\n{transcript_text}",
        }
    )

    # DeepSeek's OpenAI-compatible endpoint accepts these parameters at runtime.
    # The upstream OpenAI SDK type stubs are stricter than the provider surface,
    # so we cast request payload fields for static typing compatibility.
    response = client.chat.completions.create(
        model=settings.model,
        messages=cast(Any, messages),
        response_format=cast(Any, {"type": "json_object"}),
        temperature=settings.temperature,
        max_tokens=settings.max_completion_tokens,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("DeepSeek returned empty response.")

    parsed_script, used_repair = _decode_podcast_script_with_fallback(content)
    if used_repair:
        logger.warning(
            "Used balanced local JSON recovery for endpoint=%s.",
            endpoint_mode,
        )
    return parsed_script, used_repair


def _target_line_bounds(segment_count: int) -> tuple[int, int]:
    """Estimate desired summary line-count bounds from transcript segment count."""
    estimated = max(1, segment_count // 2)
    lower_bound = max(SUMMARY_LINE_COUNT_MIN, estimated - 8)
    upper_bound = min(SUMMARY_LINE_COUNT_MAX, estimated + 8)
    if lower_bound > upper_bound:
        lower_bound = upper_bound
    return lower_bound, upper_bound


def generate_summary_deepseek(
    transcript_text: str,
    api_key: str,
    allowed_speakers: set[str],
    settings: DeepSeekRequestSettings,
    segment_count: int,
    retry_context: RetryContext | None,
) -> GenerationSuccess:
    """Generate a structured summary from transcript text via DeepSeek.

    Args:
        transcript_text: Prompt-ready transcript with segment IDs.
        api_key: DeepSeek API key.
        allowed_speakers: Speaker labels allowed in output.
        settings: Request settings for DeepSeek chat completions.
        segment_count: Number of input transcript segments.
        retry_context: Previous-attempt failure context.

    Returns:
        Structured generation result and endpoint metadata.

    Raises:
        OpenAIAPIError: Upstream API issues that should trigger retry.
        JSONDecodeError: Response content is not parseable JSON.
        ValidationError: Parsed payload is not compatible with `PodcastScript`.
        ValueError: Empty response payload.
    """
    target_min_lines, target_max_lines = _target_line_bounds(segment_count)
    system_prompt = _build_system_prompt(
        allowed_speakers=allowed_speakers,
        target_min_lines=target_min_lines,
        target_max_lines=target_max_lines,
    )

    endpoint_plan: list[tuple[EndpointMode, str]] = []
    if settings.auto_beta:
        endpoint_plan.append(("beta", DEEPSEEK_BETA_BASE_URL))
    endpoint_plan.append(("stable", DEEPSEEK_STABLE_BASE_URL))

    last_error: Exception | None = None
    for endpoint_mode, base_url in endpoint_plan:
        endpoint_token_budget = settings.max_completion_tokens
        if endpoint_mode == "stable":
            endpoint_token_budget = min(endpoint_token_budget, 4096)

        endpoint_settings = DeepSeekRequestSettings(
            model=settings.model,
            max_completion_tokens=endpoint_token_budget,
            request_timeout_seconds=settings.request_timeout_seconds,
            http_retries=settings.http_retries,
            temperature=settings.temperature,
            auto_beta=settings.auto_beta,
        )
        client = _build_deepseek_client(
            api_key=api_key,
            base_url=base_url,
            request_timeout_seconds=endpoint_settings.request_timeout_seconds,
            http_retries=endpoint_settings.http_retries,
        )

        logger.info(
            "Sending transcript to DeepSeek endpoint=%s model=%s max_tokens=%d temperature=%.2f timeout=%.1fs retries=%d",
            endpoint_mode,
            endpoint_settings.model,
            endpoint_settings.max_completion_tokens,
            endpoint_settings.temperature,
            endpoint_settings.request_timeout_seconds,
            endpoint_settings.http_retries,
        )

        try:
            script, used_repair = _request_deepseek_completion(
                client=client,
                settings=endpoint_settings,
                transcript_text=transcript_text,
                system_prompt=system_prompt,
                retry_context=retry_context,
                endpoint_mode=endpoint_mode,
            )
            return GenerationSuccess(
                script=script,
                endpoint_mode=endpoint_mode,
                used_json_repair=used_repair,
            )
        except Exception as error:  # noqa: BLE001
            last_error = error
            if endpoint_mode == "beta" and _looks_like_beta_endpoint_error(error):
                logger.warning(
                    "DeepSeek beta endpoint failed (%s). Falling back to stable endpoint.",
                    _error_digest(error),
                )
                continue
            raise GenerationAttemptError(endpoint_mode=endpoint_mode, cause=error) from error

    if last_error is not None:
        raise GenerationAttemptError(endpoint_mode="stable", cause=last_error) from last_error
    raise RuntimeError("No DeepSeek endpoint candidates were available.")


def post_process_script(
    lines: list[ValidatedDialogueLine], voice_sample_dir: str
) -> list[FinalScriptLine]:
    """Inject voice paths and validation metadata into final output schema.

    Args:
        lines: Validated and repaired dialogue lines.
        voice_sample_dir: Directory that stores speaker reference wav files.

    Returns:
        Final JSON-ready script lines.
    """
    final_output: list[FinalScriptLine] = []

    for line in lines:
        voice_path = os.path.join(voice_sample_dir, f"{line.speaker}.wav")
        normalized_voice_path = voice_path.replace("\\", "/")
        if not os.path.exists(voice_path):
            logger.warning(
                "Voice sample not found for speaker '%s': %s",
                line.speaker,
                voice_path,
            )

        final_output.append(
            {
                "speaker": line.speaker,
                "voice_sample": normalized_voice_path,
                "use_emo_text": True,
                "emo_text": line.emo_text,
                "emo_alpha": line.emo_alpha,
                "text": line.text,
                "source_segment_ids": line.source_segment_ids,
                "validation_status": line.validation_status,
                "repair_reason": line.repair_reason,
            }
        )

    return final_output


def _classify_generation_error(error: Exception) -> ErrorType:
    """Classify generation/parsing exception into retry-guidance categories."""
    if isinstance(error, JSONDecodeError):
        return _classify_json_error(error)
    if isinstance(error, ValidationError):
        return "schema_validation"
    if isinstance(error, ValueError) and "empty response" in str(error).lower():
        return "empty_response"
    return "api_error"


def main() -> int:
    """Run the DeepSeek summarizer with strict transcript-grounded speaker validation."""
    parser = argparse.ArgumentParser(
        description="CARD Script Summarizer & Emotion Annotator (Deepseek)"
    )
    parser.add_argument(
        "--transcript", required=True, help="Path to input WhisperX JSON transcript"
    )
    parser.add_argument(
        "--voice-dir",
        required=True,
        help="Directory where separated speaker audios are stored",
    )
    parser.add_argument(
        "--output",
        default="summarized_script.json",
        help="Path to save output JSON",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help="DeepSeek API Key",
    )
    parser.add_argument(
        "--model",
        default=DEEPSEEK_MODEL,
        help=f"DeepSeek model to use (default: {DEEPSEEK_MODEL})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES_DEFAULT,
        help=(
            "Max retry attempts for LLM generation and validation "
            f"(default: {MAX_RETRIES_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help=(
            "Maximum output token budget for DeepSeek responses; "
            f"default: {DEFAULT_MAX_COMPLETION_TOKENS}"
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "HTTP request timeout in seconds for DeepSeek requests; "
            f"default: {DEFAULT_TIMEOUT_SECONDS}"
        ),
    )
    parser.add_argument(
        "--http-retries",
        type=int,
        default=DEFAULT_HTTP_RETRIES,
        help=f"Transport retries per HTTP request; default: {DEFAULT_HTTP_RETRIES}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature for DeepSeek generation; default: {DEFAULT_TEMPERATURE}",
    )
    args = parser.parse_args()
    configure_logging()

    if not args.api_key:
        logger.error("No API key provided. Set DEEPSEEK_API_KEY or pass --api-key.")
        return 1

    settings = DeepSeekRequestSettings(
        model=args.model,
        max_completion_tokens=max(256, int(args.max_completion_tokens)),
        request_timeout_seconds=max(5.0, float(args.request_timeout_seconds)),
        http_retries=max(0, int(args.http_retries)),
        temperature=min(2.0, max(0.0, float(args.temperature))),
        auto_beta=True,
    )
    logger.info(
        "DeepSeek settings: model=%s max_tokens=%d timeout=%.1fs http_retries=%d temperature=%.2f endpoint_mode=auto_beta",
        settings.model,
        settings.max_completion_tokens,
        settings.request_timeout_seconds,
        settings.http_retries,
        settings.temperature,
    )

    logger.info("Loading transcript from %s", args.transcript)
    segments = load_transcript_segments(args.transcript)
    if not segments:
        logger.error("No transcript segments found in %s", args.transcript)
        return 1

    allowed_speakers = collect_allowed_speakers(segments)
    if not allowed_speakers:
        logger.error("No speaker labels found in transcript.")
        return 1

    segment_speaker_map = build_segment_speaker_map(segments)
    transcript_text = format_segments_for_prompt(segments)
    logger.info(
        "Loaded %d transcript segments across %d speakers.",
        len(segments),
        len(allowed_speakers),
    )

    max_attempts = max(1, args.max_retries + 1)
    validated_lines: list[ValidatedDialogueLine] | None = None
    retry_context: RetryContext | None = None

    for attempt in range(1, max_attempts + 1):
        logger.info("Summarization attempt %d/%d.", attempt, max_attempts)
        try:
            generation_result = generate_summary_deepseek(
                transcript_text=transcript_text,
                api_key=args.api_key,
                allowed_speakers=allowed_speakers,
                settings=settings,
                segment_count=len(segments),
                retry_context=retry_context,
            )
            structured_script = generation_result.script
        except Exception as exc:  # noqa: BLE001
            failure_endpoint: EndpointMode = "stable"
            root_error: Exception = exc
            if isinstance(exc, GenerationAttemptError):
                failure_endpoint = exc.endpoint_mode
                root_error = exc.cause

            classified_error = _classify_generation_error(root_error)
            retry_context = RetryContext(
                attempt_index=attempt,
                endpoint_mode=failure_endpoint,
                error_type=classified_error,
                error_digest=_error_digest(root_error),
            )
            if attempt < max_attempts:
                logger.warning(
                    "Generation failed on attempt %d: type=%s detail=%s",
                    attempt,
                    classified_error,
                    retry_context.error_digest,
                )
                sleep_seconds = min(8.0, float(2 ** (attempt - 1)))
                logger.info("Retrying after %.1f seconds backoff.", sleep_seconds)
                time.sleep(sleep_seconds)
                continue
            logger.error(
                "Generation failed after %d attempts: type=%s detail=%s",
                max_attempts,
                classified_error,
                retry_context.error_digest,
            )
            return 1

        payload_lines: list[DialogueLinePayload] = [
            cast(DialogueLinePayload, dialogue_line.model_dump())
            for dialogue_line in structured_script.dialogue
        ]
        validation_report = validate_and_repair_dialogue(
            payload_lines,
            allowed_speakers=allowed_speakers,
            segment_speaker_map=segment_speaker_map,
        )

        logger.info(
            "Validation result: total_lines=%d repaired_lines=%d issues=%d endpoint=%s json_repair=%s",
            len(validation_report.lines),
            validation_report.repaired_lines,
            len(validation_report.issues),
            generation_result.endpoint_mode,
            generation_result.used_json_repair,
        )
        if validation_report.is_valid:
            validated_lines = validation_report.lines
            break

        for issue in validation_report.issues:
            logger.error("Validation issue: %s", issue)

        retry_context = RetryContext(
            attempt_index=attempt,
            endpoint_mode=generation_result.endpoint_mode,
            error_type="schema_validation",
            error_digest=(
                f"validation issues={len(validation_report.issues)}; "
                f"first={validation_report.issues[0] if validation_report.issues else 'none'}"
            ),
        )
        if attempt < max_attempts:
            logger.warning(
                "Validation failed on attempt %d; retrying generation with corrective feedback.",
                attempt,
            )
            sleep_seconds = min(8.0, float(2 ** (attempt - 1)))
            logger.info("Retrying after %.1f seconds backoff.", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        logger.error("Validation failed after %d attempts.", max_attempts)
        return 1

    if validated_lines is None:
        logger.error("No validated dialogue was produced.")
        return 1

    final_json = post_process_script(validated_lines, args.voice_dir)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(final_json, output_file, indent=2)

    logger.info("Success: summarized script saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
