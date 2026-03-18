"""Automatic transcript sampling workflow for SPoRC with Anthropic."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from card_framework.agents.utils import count_words
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_LLM_SAMPLING_ROOT,
    DEFAULT_LONGFORM_MANIFEST_PATH,
    DURATION_PRESET_SECONDS,
    PreparedBenchmarkSample,
)
from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    iter_prepared_dataset_samples,
    load_prepared_dataset_manifest,
)
from card_framework.orchestration.transcript import Transcript
from card_framework.providers.response_callbacks import (
    CompositeResponseCallback,
    RichConsoleResponseCallback,
)
from card_framework.shared.env import load_repo_dotenv

_DEFAULT_DURATION_PRESET = "30s"
_DEFAULT_EPISODES = 20
_DEFAULT_WORDS_PER_MINUTE = 170.0
_DEFAULT_SAMPLE_FACTOR = 5.0
_DEFAULT_MODEL = "claude-opus-4-6"
_DEFAULT_THINKING_EFFORT = "medium"
_DEFAULT_MAX_OUTPUT_TOKENS = 16_000
_VALID_THINKING_EFFORTS = {"low", "medium", "high", "max"}
_LLM_PRESET_SECONDS = ("30s", "1m", "5m")
_TARGET_WORD_TOLERANCE_RATIO = 0.15
_SUMMARY_FILENAME = "run_summary.json"
_SELECTED_SAMPLES_JSONL_FILENAME = "selected_samples.jsonl"
_SELECTED_SAMPLES_CSV_FILENAME = "selected_samples.csv"
_SELECTED_SEGMENTS_TXT_FILENAME = "selected_segments.txt"
_EPISODE_RANGE_PATTERN = re.compile(
    r"^\s*\[?\s*(\d+)\s*(?:,\s*|\-\s*)(\d+)\s*\]?\s*$"
)


@dataclass(slots=True, frozen=True)
class _RunPaths:
    """Store resolved filesystem paths for one sampler run."""

    run_root: Path
    summary_path: Path
    selected_samples_jsonl_path: Path
    selected_samples_csv_path: Path
    episodes_root: Path


@dataclass(slots=True, frozen=True)
class _EpisodeSelection:
    """Represent one persisted selected segment."""

    order: int
    segment_text: str
    rationale: str
    estimated_words: int
    actual_words: int
    speaker_label: str


class _StreamRecorderCallback:
    """Write streamed thought and content events to one JSONL trace file."""

    def __init__(self, *, stream_path: Path, attempt_index: int) -> None:
        self._handle = stream_path.open("a", encoding="utf-8")
        self._attempt_index = attempt_index

    def _write_event(self, event: str, **payload: object) -> None:
        record = {
            "attempt_index": self._attempt_index,
            "event": event,
            "timestamp_utc": _utc_now(),
            **payload,
        }
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def on_start(self, agent_name: str) -> None:
        self._write_event("start", agent_name=agent_name)

    def on_thought_token(self, token: str) -> None:
        self._write_event("thinking_token", token=token)

    def on_content_token(self, token: str) -> None:
        self._write_event("content_token", token=token)

    def on_complete(self) -> None:
        self._write_event("complete")

    def close(self) -> None:
        self._handle.close()


def run_llm_sampling_session(
    args: argparse.Namespace,
    *,
    console: Console | None = None,
) -> int:
    """Run the automatic Anthropic-backed SPoRC transcript sampler."""
    load_repo_dotenv(search_cwd=True)
    active_console = console if console is not None else Console()

    try:
        provider_class = _load_anthropic_provider_class()
        settings = _resolve_settings(args)
        _validate_startup(settings=settings)
        run_paths = _create_run_paths(settings["sampling_root"])
        run_paths.run_root.mkdir(parents=True, exist_ok=True)
        run_paths.episodes_root.mkdir(parents=True, exist_ok=True)
        prepared_samples = _load_episode_selection(
            manifest_path=settings["dataset_manifest"],
            episodes_requested=settings["episodes"],
            episode_range=settings["episode_range"],
        )
    except Exception as exc:
        _render_error(active_console, str(exc))
        return 1

    provider = provider_class(
        api_key=str(settings["api_key"]),
        model=str(settings["model"]),
        enable_thinking=bool(settings["enable_thinking"]),
        thinking=(
            {"type": "adaptive", "display": "summarized"}
            if settings["enable_thinking"]
            else None
        ),
        output_config=(
            {"effort": str(settings["thinking_effort"])}
            if settings["enable_thinking"]
            else None
        ),
        temperature=(None if settings["enable_thinking"] else 0.0),
        default_max_tokens=int(settings["max_output_tokens"]),
    )

    run_id = run_paths.run_root.name
    summary_payload = _build_initial_summary_payload(
        run_id=run_id,
        settings=settings,
        prepared_samples=prepared_samples,
    )
    selected_rows: list[dict[str, Any]] = []
    _write_run_exports(
        run_paths=run_paths,
        summary_payload=summary_payload,
        selected_rows=selected_rows,
    )
    _render_run_banner(
        active_console,
        run_id=run_id,
        settings=settings,
        run_paths=run_paths,
    )

    exit_code = 0
    try:
        for episode_order, sample in enumerate(prepared_samples, start=1):
            manifest_episode_number = _resolve_manifest_episode_number(
                settings=settings,
                episode_order=episode_order,
            )
            episode_result = _run_episode_sampling(
                provider=provider,
                sample=sample,
                episode_order=episode_order,
                manifest_episode_number=manifest_episode_number,
                settings=settings,
                run_paths=run_paths,
                console=active_console,
            )
            summary_payload["episodes"].append(episode_result["summary"])
            if episode_result["selected_rows"]:
                selected_rows.extend(episode_result["selected_rows"])
            if episode_result["summary"]["status"] != "completed":
                exit_code = 1
            summary_payload["status"] = (
                "completed"
                if exit_code == 0
                and len(summary_payload["episodes"]) == len(prepared_samples)
                else "completed_with_failures"
            )
            summary_payload["updated_at_utc"] = _utc_now()
            summary_payload["selected_sample_count"] = len(selected_rows)
            summary_payload["completed_episode_count"] = sum(
                1
                for episode in summary_payload["episodes"]
                if episode["status"] == "completed"
            )
            _write_run_exports(
                run_paths=run_paths,
                summary_payload=summary_payload,
                selected_rows=selected_rows,
            )
    except KeyboardInterrupt:
        summary_payload["status"] = "cancelled"
        summary_payload["updated_at_utc"] = _utc_now()
        _write_run_exports(
            run_paths=run_paths,
            summary_payload=summary_payload,
            selected_rows=selected_rows,
        )
        active_console.print(
            Panel.fit(
                "\n".join(
                    [
                        "LLM sampling cancelled.",
                        f"Run summary: {run_paths.summary_path}",
                        f"Selections JSONL: {run_paths.selected_samples_jsonl_path}",
                    ]
                ),
                title="Sampling Cancelled",
                border_style="yellow",
            )
        )
        return 130
    finally:
        closeable_client = getattr(provider, "_client", None)
        close_fn = getattr(closeable_client, "close", None)
        if callable(close_fn):
            close_fn()

    _render_completion_banner(
        active_console,
        run_id=run_id,
        run_paths=run_paths,
        selected_sample_count=len(selected_rows),
        failed_episode_count=sum(
            1
            for episode in summary_payload["episodes"]
            if episode["status"] != "completed"
        ),
    )
    return exit_code


def _resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve CLI arguments into a validated settings payload."""
    preset = str(getattr(args, "duration_preset", _DEFAULT_DURATION_PRESET) or "").strip()
    if preset not in _LLM_PRESET_SECONDS:
        raise RuntimeError(
            "SPoRC llm-sample only supports duration presets 30s, 1m, and 5m."
        )

    episodes = int(getattr(args, "episodes", _DEFAULT_EPISODES) or _DEFAULT_EPISODES)
    episode_range = _parse_episode_selection(
        str(getattr(args, "select_episode", "") or "")
    )
    if episode_range is None and episodes <= 0:
        raise RuntimeError("Episode count must be greater than zero.")
    if episode_range is not None:
        episodes = episode_range[1] - episode_range[0] + 1

    words_per_minute = float(
        getattr(args, "words_per_minute", _DEFAULT_WORDS_PER_MINUTE)
        or _DEFAULT_WORDS_PER_MINUTE
    )
    if words_per_minute <= 0:
        raise RuntimeError("Words per minute must be greater than zero.")

    sample_factor = float(
        getattr(args, "sample_factor", _DEFAULT_SAMPLE_FACTOR)
        or _DEFAULT_SAMPLE_FACTOR
    )
    if sample_factor <= 0:
        raise RuntimeError("Sample factor must be greater than zero.")

    requested_samples = int(getattr(args, "samples", 0) or 0)
    if requested_samples < 0:
        raise RuntimeError("Sample count must be zero or greater.")

    model = str(getattr(args, "model", _DEFAULT_MODEL) or "").strip() or _DEFAULT_MODEL
    thinking_effort = str(
        getattr(args, "thinking_effort", _DEFAULT_THINKING_EFFORT) or ""
    ).strip().lower() or _DEFAULT_THINKING_EFFORT
    if thinking_effort not in _VALID_THINKING_EFFORTS:
        raise RuntimeError("Thinking effort must be one of: low, medium, high, max.")

    max_output_tokens = int(
        getattr(args, "max_output_tokens", _DEFAULT_MAX_OUTPUT_TOKENS)
        or _DEFAULT_MAX_OUTPUT_TOKENS
    )
    if max_output_tokens <= 0:
        raise RuntimeError("Max output tokens must be greater than zero.")

    dataset_manifest = Path(
        str(getattr(args, "dataset_manifest", DEFAULT_LONGFORM_MANIFEST_PATH) or "")
    ).resolve()
    sampling_root = Path(
        str(getattr(args, "sampling_root", DEFAULT_LLM_SAMPLING_ROOT) or "")
    ).resolve()
    enable_thinking = not bool(getattr(args, "disable_thinking", False))
    sampling_mode = "fixed_samples" if requested_samples > 0 else "target_words"
    target_words = (
        None
        if requested_samples > 0
        else _compute_target_words(
            DURATION_PRESET_SECONDS[preset],
            words_per_minute=words_per_minute,
            sample_factor=sample_factor,
        )
    )
    api_key = str(os.environ.get("ANTHROPIC_API_KEY", "")).strip()

    return {
        "dataset_manifest": dataset_manifest,
        "sampling_root": sampling_root,
        "sampling_mode": sampling_mode,
        "duration_preset": preset,
        "preset_seconds": DURATION_PRESET_SECONDS[preset],
        "episodes": episodes,
        "episode_range": episode_range,
        "requested_samples": requested_samples if requested_samples > 0 else None,
        "words_per_minute": words_per_minute,
        "sample_factor": sample_factor,
        "target_words": target_words,
        "model": model,
        "thinking_effort": thinking_effort,
        "enable_thinking": enable_thinking,
        "max_output_tokens": max_output_tokens,
        "api_key": api_key,
    }


def _validate_startup(*, settings: dict[str, Any]) -> None:
    """Fail fast on missing startup requirements."""
    manifest_path = Path(settings["dataset_manifest"])
    if not manifest_path.exists():
        raise RuntimeError(
            "SPoRC llm-sample requires an existing manifest. "
            f"Expected: {manifest_path}"
        )
    if not str(settings["api_key"]).strip():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Add it to the process environment or "
            "the repository-root .env file before running `llm-sample`."
        )


def _load_anthropic_provider_class() -> type[Any]:
    """Import the shared Anthropic provider lazily for this command."""
    from card_framework.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider


def _create_run_paths(sampling_root: Path) -> _RunPaths:
    """Create resolved artifact paths for one llm-sample run."""
    run_id = (
        datetime.now(timezone.utc).strftime("llm_sample_%Y%m%dT%H%M%SZ_")
        + uuid4().hex[:8]
    )
    run_root = (sampling_root / run_id).resolve()
    return _RunPaths(
        run_root=run_root,
        summary_path=(run_root / _SUMMARY_FILENAME).resolve(),
        selected_samples_jsonl_path=(run_root / _SELECTED_SAMPLES_JSONL_FILENAME).resolve(),
        selected_samples_csv_path=(run_root / _SELECTED_SAMPLES_CSV_FILENAME).resolve(),
        episodes_root=(run_root / "episodes").resolve(),
    )


def _load_episode_prefix(
    *,
    manifest_path: Path,
    episodes_requested: int,
) -> tuple[PreparedBenchmarkSample, ...]:
    """Load the requested manifest prefix for automatic sampling."""
    manifest = load_prepared_dataset_manifest(manifest_path, include_samples=False)
    if episodes_requested > int(manifest.sample_count):
        raise RuntimeError(
            "Requested episode prefix exceeds the available manifest length: "
            f"{episodes_requested} > {manifest.sample_count}."
        )
    prepared_samples = tuple(
        iter_prepared_dataset_samples(
            manifest_path,
            max_samples=episodes_requested,
        )
    )
    if len(prepared_samples) != episodes_requested:
        raise RuntimeError(
            "Unable to load the requested SPoRC episode prefix from the manifest."
        )
    return prepared_samples


def _parse_episode_selection(raw_value: str) -> tuple[int, int] | None:
    """Parse one 1-based inclusive episode range like `[7,10]`."""
    normalized = str(raw_value or "").strip()
    if not normalized:
        return None
    match = _EPISODE_RANGE_PATTERN.fullmatch(normalized)
    if match is None:
        raise RuntimeError(
            "Episode selection must look like `[start,end]`, `start,end`, or "
            "`start-end`, using 1-based inclusive episode numbers."
        )
    start_index = int(match.group(1))
    end_index = int(match.group(2))
    if start_index <= 0 or end_index <= 0:
        raise RuntimeError("Episode selection indices must be greater than zero.")
    if end_index < start_index:
        raise RuntimeError(
            "Episode selection end must be greater than or equal to the start."
        )
    return start_index, end_index


def _load_episode_selection(
    *,
    manifest_path: Path,
    episodes_requested: int,
    episode_range: tuple[int, int] | None,
) -> tuple[PreparedBenchmarkSample, ...]:
    """Load either the manifest prefix or one explicit 1-based inclusive range."""
    if episode_range is None:
        return _load_episode_prefix(
            manifest_path=manifest_path,
            episodes_requested=episodes_requested,
        )

    manifest = load_prepared_dataset_manifest(manifest_path, include_samples=False)
    manifest_sample_count = int(manifest.sample_count)
    start_index, end_index = episode_range
    if end_index > manifest_sample_count:
        raise RuntimeError(
            "Requested episode range exceeds the available manifest length: "
            f"{end_index} > {manifest_sample_count}."
        )

    selected_samples = [
        sample
        for manifest_index, sample in enumerate(
            iter_prepared_dataset_samples(manifest_path, max_samples=end_index),
            start=1,
        )
        if manifest_index >= start_index
    ]
    expected_count = end_index - start_index + 1
    if len(selected_samples) != expected_count:
        raise RuntimeError(
            "Unable to load the requested SPoRC episode range from the manifest."
        )
    return tuple(selected_samples)


def _build_initial_summary_payload(
    *,
    run_id: str,
    settings: dict[str, Any],
    prepared_samples: tuple[PreparedBenchmarkSample, ...],
) -> dict[str, Any]:
    """Build the initial run summary structure."""
    return {
        "run_id": run_id,
        "status": "running",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "manifest_path": str(settings["dataset_manifest"]),
        "duration_preset": str(settings["duration_preset"]),
        "preset_seconds": int(settings["preset_seconds"]),
        "episodes_requested": int(settings["episodes"]),
        "selected_episode_range": (
            list(settings["episode_range"])
            if settings["episode_range"] is not None
            else None
        ),
        "sampling_mode": str(settings["sampling_mode"]),
        "target_samples_per_episode": settings["requested_samples"],
        "words_per_minute": float(settings["words_per_minute"]),
        "sample_factor": float(settings["sample_factor"]),
        "target_words_per_episode": settings["target_words"],
        "model": str(settings["model"]),
        "thinking_enabled": bool(settings["enable_thinking"]),
        "thinking_effort": str(settings["thinking_effort"]),
        "max_output_tokens": int(settings["max_output_tokens"]),
        "selected_sample_count": 0,
        "completed_episode_count": 0,
        "total_episode_count": len(prepared_samples),
        "episodes": [],
    }


def _write_run_exports(
    *,
    run_paths: _RunPaths,
    summary_payload: dict[str, Any],
    selected_rows: list[dict[str, Any]],
) -> None:
    """Write live run summary plus selection exports."""
    run_paths.summary_path.write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with run_paths.selected_samples_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row) + "\n")

    fieldnames = (
        list(selected_rows[0].keys())
        if selected_rows
        else [
            "run_id",
            "duration_preset",
            "target_words",
            "requested_samples",
            "episode_order",
            "manifest_episode_number",
            "sample_id",
            "episode_title",
            "episode_key",
            "transcript_path",
            "selection_order",
            "speaker_label",
            "segment_text",
            "rationale",
            "estimated_words",
            "actual_words",
        ]
    )
    with run_paths.selected_samples_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)


def _run_episode_sampling(
    *,
    provider: Any,
    sample: PreparedBenchmarkSample,
    episode_order: int,
    manifest_episode_number: int,
    settings: dict[str, Any],
    run_paths: _RunPaths,
    console: Console,
) -> dict[str, Any]:
    """Execute automatic sampling for one transcript episode."""
    transcript_path = Path(sample.transcript_path).resolve()
    episode_title = (
        str(sample.metadata.get("episode_title", "") or "").strip() or sample.sample_id
    )
    episode_dir = (run_paths.episodes_root / sample.sample_id).resolve()
    episode_dir.mkdir(parents=True, exist_ok=True)
    request_path = (episode_dir / "request.json").resolve()
    stream_path = (episode_dir / "stream.jsonl").resolve()
    response_path = (episode_dir / "response.json").resolve()
    selected_segments_path = (episode_dir / _SELECTED_SEGMENTS_TXT_FILENAME).resolve()

    try:
        transcript_payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript = Transcript.from_mapping(transcript_payload)
        transcript_word_count = sum(
            count_words(segment.text) for segment in transcript.segments
        )
        transcript_text = _render_transcript_for_prompt(transcript)
    except Exception as exc:
        failure_status = _classify_episode_error(str(exc))
        response_path.write_text(
            json.dumps(
                {
                    "run_id": run_paths.run_root.name,
                    "episode_order": episode_order,
                    "manifest_episode_number": manifest_episode_number,
                    "sample_id": sample.sample_id,
                    "episode_title": episode_title,
                    "status": failure_status,
                    "attempts": [],
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _render_episode_failure(console, validation_message=str(exc))
        return {
            "summary": {
                "episode_order": episode_order,
                "manifest_episode_number": manifest_episode_number,
                "sample_id": sample.sample_id,
                "episode_key": sample.episode_key,
                "episode_title": episode_title,
                "transcript_path": str(transcript_path),
                "transcript_word_count": 0,
                "target_words": settings["target_words"],
                "requested_samples": settings["requested_samples"],
                "selected_words": 0,
                "selection_count": 0,
                "status": failure_status,
                "request_path": str(request_path),
                "response_path": str(response_path),
                "stream_path": str(stream_path),
                "selected_segments_path": "",
            },
            "selected_rows": [],
        }

    _render_episode_header(
        console,
        manifest_episode_number=manifest_episode_number,
        episode_title=episode_title,
        sample_id=sample.sample_id,
        transcript_word_count=transcript_word_count,
        target_label=_episode_target_label(settings),
    )

    system_prompt = _build_system_prompt(
        sampling_mode=str(settings["sampling_mode"]),
        target_words=settings["target_words"],
        requested_samples=settings["requested_samples"],
    )
    base_user_prompt = _build_user_prompt(
        sample=sample,
        transcript_text=transcript_text,
        sampling_mode=str(settings["sampling_mode"]),
        target_words=settings["target_words"],
        requested_samples=settings["requested_samples"],
        transcript_word_count=transcript_word_count,
    )
    tool_schema = _submit_samples_tool_schema()
    tool_choice = {"type": "function", "function": {"name": "submit_samples"}}

    attempts_payload: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    final_status = "failed_validation"
    final_selected_words = 0

    for attempt_index in (1, 2):
        prompt_blocks = [base_user_prompt]
        if attempts_payload:
            prior_attempt = attempts_payload[-1]
            prompt_blocks.append(
                _build_correction_prompt(
                    validation_message=str(prior_attempt["validation"]["message"]),
                    previous_response=prior_attempt["assistant_message"],
                    sampling_mode=str(settings["sampling_mode"]),
                    target_words=settings["target_words"],
                    requested_samples=settings["requested_samples"],
                )
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(prompt_blocks)},
        ]
        request_payload = {
            "attempt_index": attempt_index,
            "messages": messages,
            "tools": [tool_schema],
            "tool_choice": tool_choice,
            "max_output_tokens": int(settings["max_output_tokens"]),
            "thinking_enabled": bool(settings["enable_thinking"]),
            "thinking_effort": str(settings["thinking_effort"]),
            "model": str(settings["model"]),
        }
        _append_attempt_request(
            request_path=request_path,
            run_id=run_paths.run_root.name,
            episode_order=episode_order,
            manifest_episode_number=manifest_episode_number,
            sample=sample,
            episode_title=episode_title,
            transcript_path=transcript_path,
            transcript_word_count=transcript_word_count,
            target_words=settings["target_words"],
            requested_samples=settings["requested_samples"],
            request_payload=request_payload,
        )

        stream_recorder = _StreamRecorderCallback(
            stream_path=stream_path,
            attempt_index=attempt_index,
        )
        provider.set_response_callback(
            CompositeResponseCallback(
                RichConsoleResponseCallback(),
                stream_recorder,
            )
        )
        if hasattr(provider, "last_response_details"):
            provider.last_response_details = {}
        try:
            try:
                assistant_message = provider.chat(
                    messages=messages,
                    tools=[tool_schema],
                    tool_choice=tool_choice,
                    max_tokens=int(settings["max_output_tokens"]),
                )
            except Exception as exc:
                validation = {
                    "status": _classify_episode_error(str(exc)),
                    "message": str(exc),
                    "actual_selected_words": 0,
                }
                attempts_payload.append(
                    {
                        "attempt_index": attempt_index,
                        "assistant_message": {},
                        "provider_trace": dict(
                            getattr(provider, "last_response_details", {})
                        ),
                        "validation": validation,
                    }
                )
                response_path.write_text(
                    json.dumps(
                        {
                            "run_id": run_paths.run_root.name,
                            "episode_order": episode_order,
                            "manifest_episode_number": manifest_episode_number,
                            "sample_id": sample.sample_id,
                            "episode_title": episode_title,
                            "status": validation["status"],
                            "attempts": attempts_payload,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                final_status = str(validation["status"])
                _render_episode_failure(
                    console,
                    validation_message=str(validation["message"]),
                )
                break
        finally:
            stream_recorder.close()

        parsed_samples, validation = _validate_submission(
            assistant_message=assistant_message.model_dump(),
            episode_key=sample.episode_key,
            target_words=settings["target_words"],
            requested_samples=settings["requested_samples"],
        )
        attempts_payload.append(
            {
                "attempt_index": attempt_index,
                "assistant_message": assistant_message.model_dump(),
                "provider_trace": dict(getattr(provider, "last_response_details", {})),
                "validation": validation,
            }
        )
        response_path.write_text(
            json.dumps(
                {
                    "run_id": run_paths.run_root.name,
                    "episode_order": episode_order,
                    "manifest_episode_number": manifest_episode_number,
                    "sample_id": sample.sample_id,
                    "episode_title": episode_title,
                    "status": validation["status"],
                    "attempts": attempts_payload,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if validation["status"] == "completed":
            final_status = "completed"
            final_selected_words = int(validation["actual_selected_words"])
            selected_rows = _build_selected_rows(
                run_id=run_paths.run_root.name,
                duration_preset=str(settings["duration_preset"]),
                target_words=settings["target_words"],
                requested_samples=settings["requested_samples"],
                episode_order=episode_order,
                manifest_episode_number=manifest_episode_number,
                sample=sample,
                episode_title=episode_title,
                transcript_path=transcript_path,
                parsed_samples=parsed_samples,
            )
            _write_episode_segments_text(
                output_path=selected_segments_path,
                episode_title=episode_title,
                sample_id=sample.sample_id,
                parsed_samples=parsed_samples,
            )
            _render_episode_success(
                console,
                selected_words=final_selected_words,
                selection_count=len(parsed_samples),
            )
            break

        final_status = str(validation["status"])
        if attempt_index == 1:
            _render_episode_retry(
                console,
                validation_message=str(validation["message"]),
            )
        else:
            _render_episode_failure(
                console,
                validation_message=str(validation["message"]),
            )

    response_doc = json.loads(response_path.read_text(encoding="utf-8"))
    response_doc["status"] = final_status
    response_doc["selected_words"] = final_selected_words
    response_doc["selected_samples"] = selected_rows
    response_doc["selected_segments_path"] = (
        str(selected_segments_path) if selected_rows else ""
    )
    response_path.write_text(
        json.dumps(response_doc, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "summary": {
            "episode_order": episode_order,
            "manifest_episode_number": manifest_episode_number,
            "sample_id": sample.sample_id,
            "episode_key": sample.episode_key,
            "episode_title": episode_title,
            "transcript_path": str(transcript_path),
            "transcript_word_count": transcript_word_count,
            "target_words": settings["target_words"],
            "requested_samples": settings["requested_samples"],
            "selected_words": final_selected_words,
            "selection_count": len(selected_rows),
            "status": final_status,
            "request_path": str(request_path),
            "response_path": str(response_path),
            "stream_path": str(stream_path),
            "selected_segments_path": (
                str(selected_segments_path) if selected_rows else ""
            ),
        },
        "selected_rows": selected_rows,
    }


def _append_attempt_request(
    *,
    request_path: Path,
    run_id: str,
    episode_order: int,
    manifest_episode_number: int,
    sample: PreparedBenchmarkSample,
    episode_title: str,
    transcript_path: Path,
    transcript_word_count: int,
    target_words: int | None,
    requested_samples: int | None,
    request_payload: dict[str, Any],
) -> None:
    """Append one attempt request into the episode request artifact."""
    if request_path.exists():
        request_doc = json.loads(request_path.read_text(encoding="utf-8"))
        request_doc["attempt_requests"].append(request_payload)
    else:
        request_doc = {
            "run_id": run_id,
            "episode_order": episode_order,
            "manifest_episode_number": manifest_episode_number,
            "sample_id": sample.sample_id,
            "episode_key": sample.episode_key,
            "episode_title": episode_title,
            "transcript_path": str(transcript_path),
            "transcript_word_count": transcript_word_count,
            "target_words": target_words,
            "requested_samples": requested_samples,
            "attempt_requests": [request_payload],
        }
    request_path.write_text(
        json.dumps(request_doc, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_system_prompt(
    *,
    sampling_mode: str,
    target_words: int | None,
    requested_samples: int | None,
) -> str:
    """Build the fixed system prompt for transcript sampling."""
    if sampling_mode == "fixed_samples":
        assert requested_samples is not None
        goal_text = f"select exactly {requested_samples} transcript-grounded source segments"
        budget_text = "Return exactly the requested number of samples."
    else:
        assert target_words is not None
        goal_text = (
            "select transcript-grounded source segments that a researcher or downstream "
            f"system should rely on to build an approximately {target_words}-word summary"
        )
        budget_text = "Keep the total selected words close to the target."
    return (
        "You are an expert transcript sampler for summary research. "
        f"Your job is to {goal_text}. "
        "You are not writing the summary itself. "
        "Call `submit_samples` exactly once. "
        "Keep every sample grounded in the transcript and focused on summary-relevant material. "
        "Preserve what the speaker said, including disfluencies, repetitions, and filler words. "
        "You may lightly clean up only transcript formatting noise such as broken spacing, "
        "fragmentation, or punctuation when needed so each sample is understandable to a "
        "human reviewer. "
        "Do not invent major details that are unsupported by the transcript. "
        "Every sample must include a direct `speaker_label` copied from the transcript tags. "
        "Prefer segments that capture the episode arc, major decisions, conflicts, reveals, resolutions, and distinctive speaker phrasing. "
        f"{budget_text}"
    )


def _build_user_prompt(
    *,
    sample: PreparedBenchmarkSample,
    transcript_text: str,
    sampling_mode: str,
    target_words: int | None,
    requested_samples: int | None,
    transcript_word_count: int,
) -> str:
    """Build the per-episode user prompt."""
    episode_title = (
        str(sample.metadata.get("episode_title", "") or "").strip() or sample.sample_id
    )
    target_line = (
        f"Requested samples: {requested_samples}"
        if sampling_mode == "fixed_samples"
        else f"Target selected words: {target_words}"
    )
    return "\n".join(
        [
            f"Episode key: {sample.episode_key}",
            f"Episode title: {episode_title}",
            f"Transcript words: {transcript_word_count}",
            target_line,
            (
                "Stay grounded in the transcript. Preserve what the speaker said, "
                "including disfluencies, repetitions, and filler words. You may "
                "lightly clean up only transcript formatting noise such as broken "
                "spacing, fragmentation, or punctuation so the segment is "
                "understandable to a human reviewer. Avoid unsupported details."
            ),
            "",
            "Transcript:",
            transcript_text,
        ]
    )


def _build_correction_prompt(
    *,
    sampling_mode: str,
    validation_message: str,
    previous_response: dict[str, Any],
    target_words: int | None,
    requested_samples: int | None,
) -> str:
    """Build the single repair prompt used after one invalid submission."""
    target_line = (
        f"Requested samples: {requested_samples}"
        if sampling_mode == "fixed_samples"
        else f"Target selected words: {target_words}"
    )
    return "\n".join(
        [
            "Your previous submission was invalid.",
            f"Validation issue: {validation_message}",
            target_line,
            (
                "Remember: stay grounded in the transcript. Preserve what the speaker "
                "said, including disfluencies, repetitions, and filler words. You may "
                "lightly clean up only transcript formatting noise such as broken "
                "spacing, fragmentation, or punctuation for readability. Avoid "
                "unsupported details."
            ),
            "Correct the submission and call `submit_samples` exactly once.",
            "Previous assistant response:",
            json.dumps(previous_response, ensure_ascii=False),
        ]
    )


def _submit_samples_tool_schema() -> dict[str, Any]:
    """Return the strict tool schema used for structured sample submission."""
    return {
        "type": "function",
        "function": {
            "name": "submit_samples",
            "description": "Submit the selected transcript-grounded sample segments.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "episode_key": {"type": "string"},
                    "requested_samples": {"type": "integer"},
                    "target_words": {"type": "integer"},
                    "selected_words": {"type": "integer"},
                    "samples": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "order": {"type": "integer"},
                                "speaker_label": {"type": "string"},
                                "segment_text": {"type": "string"},
                                "rationale": {"type": "string"},
                                "estimated_words": {"type": "integer"},
                            },
                            "required": [
                                "order",
                                "speaker_label",
                                "segment_text",
                                "rationale",
                                "estimated_words",
                            ],
                        },
                    },
                },
                "required": [
                    "episode_key",
                    "samples",
                ],
            },
        },
    }


def _validate_submission(
    *,
    assistant_message: dict[str, Any],
    episode_key: str,
    target_words: int | None,
    requested_samples: int | None,
) -> tuple[list[_EpisodeSelection], dict[str, Any]]:
    """Validate the model's structured submission."""
    raw_tool_calls = assistant_message.get("tool_calls")
    if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
        return [], {
            "status": "failed_validation",
            "message": "Model did not call submit_samples.",
            "actual_selected_words": 0,
        }

    submit_call: dict[str, Any] | None = None
    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_payload = tool_call.get("function", {})
        function = function_payload if isinstance(function_payload, dict) else {}
        if str(function.get("name") or "") == "submit_samples":
            submit_call = function
            break
    if submit_call is None:
        return [], {
            "status": "failed_validation",
            "message": "Model did not call the submit_samples tool.",
            "actual_selected_words": 0,
        }

    try:
        payload = json.loads(str(submit_call.get("arguments") or "{}"))
    except json.JSONDecodeError:
        return [], {
            "status": "failed_validation",
            "message": "submit_samples arguments were not valid JSON.",
            "actual_selected_words": 0,
        }
    if not isinstance(payload, dict):
        return [], {
            "status": "failed_validation",
            "message": "submit_samples arguments must be a JSON object.",
            "actual_selected_words": 0,
        }

    payload_episode_key = str(payload.get("episode_key") or "").strip()
    if payload_episode_key and payload_episode_key != episode_key:
        return [], {
            "status": "failed_validation",
            "message": (
                "submit_samples episode_key did not match the active episode. "
                f"Expected {episode_key}, got {payload_episode_key}."
            ),
            "actual_selected_words": 0,
        }

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        return [], {
            "status": "failed_validation",
            "message": "submit_samples returned an empty samples list.",
            "actual_selected_words": 0,
        }

    parsed_samples: list[_EpisodeSelection] = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            return [], {
                "status": "failed_validation",
                "message": "Each sample must be an object.",
                "actual_selected_words": 0,
            }
        segment_text = str(raw_sample.get("segment_text") or "").strip()
        rationale = str(raw_sample.get("rationale") or "").strip()
        speaker_label = str(
            raw_sample.get("speaker_label") or raw_sample.get("speaker_hint") or ""
        ).strip()
        if not segment_text or not rationale or not speaker_label:
            return [], {
                "status": "failed_validation",
                "message": (
                    "Each sample requires non-empty speaker_label, segment_text, "
                    "and rationale."
                ),
                "actual_selected_words": 0,
            }
        estimated_words = int(raw_sample.get("estimated_words") or 0)
        actual_words = count_words(segment_text)
        if estimated_words <= 0 or actual_words <= 0:
            return [], {
                "status": "failed_validation",
                "message": "Each sample requires positive estimated_words and non-empty text.",
                "actual_selected_words": 0,
            }
        parsed_samples.append(
            _EpisodeSelection(
                order=int(raw_sample.get("order") or len(parsed_samples) + 1),
                speaker_label=speaker_label,
                segment_text=segment_text,
                rationale=rationale,
                estimated_words=estimated_words,
                actual_words=actual_words,
            )
        )

    parsed_samples.sort(key=lambda item: item.order)
    actual_selected_words = sum(item.actual_words for item in parsed_samples)
    if requested_samples is not None and len(parsed_samples) != requested_samples:
        return parsed_samples, {
            "status": "failed_validation",
            "message": (
                "Model returned the wrong number of samples. "
                f"Requested={requested_samples}, actual={len(parsed_samples)}."
            ),
            "actual_selected_words": actual_selected_words,
        }

    if target_words is None:
        return parsed_samples, {
            "status": "completed",
            "message": "Submission accepted.",
            "actual_selected_words": actual_selected_words,
        }

    min_target_words = max(
        1,
        int(round(target_words * (1.0 - _TARGET_WORD_TOLERANCE_RATIO))),
    )
    max_target_words = max(
        min_target_words,
        int(round(target_words * (1.0 + _TARGET_WORD_TOLERANCE_RATIO))),
    )
    if actual_selected_words < min_target_words or actual_selected_words > max_target_words:
        return parsed_samples, {
            "status": "failed_validation",
            "message": (
                "Selected words were outside the allowed tolerance band. "
                f"Target={target_words}, actual={actual_selected_words}, "
                f"allowed_range={min_target_words}-{max_target_words}."
            ),
            "actual_selected_words": actual_selected_words,
        }

    return parsed_samples, {
        "status": "completed",
        "message": "Submission accepted.",
        "actual_selected_words": actual_selected_words,
    }


def _classify_episode_error(message: str) -> str:
    """Classify one provider error into a stable episode status."""
    normalized = message.lower()
    if "context" in normalized or "too long" in normalized or "too many tokens" in normalized:
        return "failed_context_limit"
    return "failed_api"


def _build_selected_rows(
    *,
    run_id: str,
    duration_preset: str,
    target_words: int | None,
    requested_samples: int | None,
    episode_order: int,
    manifest_episode_number: int,
    sample: PreparedBenchmarkSample,
    episode_title: str,
    transcript_path: Path,
    parsed_samples: list[_EpisodeSelection],
) -> list[dict[str, Any]]:
    """Build export rows for accepted sample selections."""
    return [
        {
            "run_id": run_id,
            "duration_preset": duration_preset,
            "target_words": target_words,
            "requested_samples": requested_samples,
            "episode_order": episode_order,
            "manifest_episode_number": manifest_episode_number,
            "sample_id": sample.sample_id,
            "episode_title": episode_title,
            "episode_key": sample.episode_key,
            "transcript_path": str(transcript_path),
            "selection_order": selection.order,
            "speaker_label": selection.speaker_label,
            "segment_text": selection.segment_text,
            "rationale": selection.rationale,
            "estimated_words": selection.estimated_words,
            "actual_words": selection.actual_words,
        }
        for selection in parsed_samples
    ]

def _write_episode_segments_text(
    *,
    output_path: Path,
    episode_title: str,
    sample_id: str,
    parsed_samples: list[_EpisodeSelection],
) -> None:
    """Write one plain-text summary of the accepted episode segments."""
    lines = [
        f"Episode: {episode_title}",
        f"Sample ID: {sample_id}",
        "",
    ]
    for selection in parsed_samples:
        lines.extend(
            [
                f"{selection.order}. [{selection.speaker_label}] {selection.segment_text}",
                f"Rationale: {selection.rationale}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _render_transcript_for_prompt(transcript: Transcript) -> str:
    """Render a transcript into one prompt-friendly transcript block."""
    lines: list[str] = []
    for segment in transcript.segments:
        text = str(segment.text or "").strip()
        if not text:
            continue
        lines.append(
            (
                f"[{_format_timestamp(segment.start_time)}-{_format_timestamp(segment.end_time)}] "
                f"[{segment.speaker}] {text}"
            )
        )
    if not lines:
        raise RuntimeError("Transcript contains no usable text segments.")
    return "\n".join(lines)


def _format_timestamp(milliseconds: int) -> str:
    """Format one millisecond timestamp into `HH:MM:SS`."""
    total_seconds = max(0, int(round(milliseconds / 1000.0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _compute_target_words(
    preset_seconds: int,
    *,
    words_per_minute: float,
    sample_factor: float,
) -> int:
    """Compute the per-episode target words for one preset."""
    return max(
        1,
        int(round(words_per_minute * (preset_seconds / 60.0) * sample_factor)),
    )


def _utc_now() -> str:
    """Return the current UTC timestamp as ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def _render_run_banner(
    console: Console,
    *,
    run_id: str,
    settings: dict[str, Any],
    run_paths: _RunPaths,
) -> None:
    """Render the run settings banner."""
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Run ID:[/bold] {run_id}",
                    f"[bold]Manifest:[/bold] {settings['dataset_manifest']}",
                    _render_target_label(settings),
                    _render_episode_selection_label(settings),
                    f"[bold]Model:[/bold] {settings['model']}",
                    (
                        f"[bold]Thinking:[/bold] "
                        f"{'adaptive' if settings['enable_thinking'] else 'disabled'}"
                    ),
                    f"[bold]Output root:[/bold] {run_paths.run_root}",
                ]
            ),
            title="SPoRC LLM Sampler",
            border_style="green",
        )
    )


def _render_target_label(settings: dict[str, Any]) -> str:
    """Render the run-level target label for the startup banner."""
    if settings["sampling_mode"] == "fixed_samples":
        return (
            "[bold]Per-episode target:[/bold] "
            f"{settings['requested_samples']} sample(s)"
        )
    return (
        "[bold]Per-episode target:[/bold] "
        f"{settings['words_per_minute']:.1f} WPM x "
        f"{settings['preset_seconds'] / 60.0:.2f} min x "
        f"{settings['sample_factor']:.2f} = {settings['target_words']} words"
    )


def _render_episode_selection_label(settings: dict[str, Any]) -> str:
    """Render the run-level episode selection label for the startup banner."""
    episode_range = settings.get("episode_range")
    if isinstance(episode_range, tuple) and len(episode_range) == 2:
        return (
            "[bold]Episodes:[/bold] "
            f"manifest positions {episode_range[0]}-{episode_range[1]}"
        )
    return (
        "[bold]Episodes:[/bold] "
        f"first {settings['episodes']} from manifest prefix"
    )


def _episode_target_label(settings: dict[str, Any]) -> str:
    """Render the per-episode target label for progress tables."""
    if settings["sampling_mode"] == "fixed_samples":
        return f"{settings['requested_samples']} sample(s)"
    return f"{settings['target_words']} words"


def _render_episode_header(
    console: Console,
    *,
    manifest_episode_number: int,
    episode_title: str,
    sample_id: str,
    transcript_word_count: int,
    target_label: str,
) -> None:
    """Render a per-episode progress table before streaming begins."""
    table = Table(
        title=f"Episode {manifest_episode_number}: {episode_title}",
        box=None,
        show_header=False,
    )
    table.add_row("Sample ID", sample_id)
    table.add_row("Transcript words", str(transcript_word_count))
    table.add_row("Target", target_label)
    console.print(Panel.fit(table, border_style="cyan"))


def _render_episode_retry(console: Console, *, validation_message: str) -> None:
    """Render one retry notice."""
    console.print(
        Panel.fit(
            validation_message,
            title="Retrying Episode",
            border_style="yellow",
        )
    )


def _render_episode_success(
    console: Console,
    *,
    selected_words: int,
    selection_count: int,
) -> None:
    """Render the accepted selection summary for one episode."""
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Selected words: {selected_words}",
                    f"Saved segments: {selection_count}",
                ]
            ),
            title="Episode Completed",
            border_style="green",
        )
    )


def _render_episode_failure(console: Console, *, validation_message: str) -> None:
    """Render the final failure state for one episode."""
    console.print(
        Panel.fit(
            validation_message,
            title="Episode Failed",
            border_style="red",
        )
    )


def _resolve_manifest_episode_number(
    *,
    settings: dict[str, Any],
    episode_order: int,
) -> int:
    """Map one local run order onto its manifest episode number."""
    episode_range = settings.get("episode_range")
    if isinstance(episode_range, tuple) and len(episode_range) == 2:
        return int(episode_range[0]) + int(episode_order) - 1
    return int(episode_order)


def _render_completion_banner(
    console: Console,
    *,
    run_id: str,
    run_paths: _RunPaths,
    selected_sample_count: int,
    failed_episode_count: int,
) -> None:
    """Render the final run summary banner."""
    title = (
        "Sampling Complete"
        if failed_episode_count == 0
        else "Sampling Complete With Failures"
    )
    border_style = "green" if failed_episode_count == 0 else "yellow"
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Run ID:[/bold] {run_id}",
                    f"[bold]Selected samples:[/bold] {selected_sample_count}",
                    f"[bold]Failed episodes:[/bold] {failed_episode_count}",
                    f"[bold]Summary:[/bold] {run_paths.summary_path}",
                    f"[bold]Selections JSONL:[/bold] {run_paths.selected_samples_jsonl_path}",
                ]
            ),
            title=title,
            border_style=border_style,
        )
    )


def _render_error(console: Console, message: str) -> None:
    """Render one startup or fatal error panel."""
    console.print(Panel.fit(message, title="LLM Sampler Error", border_style="red"))
