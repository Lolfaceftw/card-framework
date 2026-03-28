"""Automatic transcript sampling workflow for SPoRC with Anthropic."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
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
from card_framework.shared.paths import DEFAULT_CONFIG_PATH, REPO_ROOT

_DEFAULT_DURATION_PRESET = "30s"
_DEFAULT_EPISODES = 20
_DEFAULT_WORDS_PER_MINUTE = 170.0
_DEFAULT_SAMPLE_FACTOR = 5.0
_DEFAULT_MODEL = "claude-opus-4-6"
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_THINKING_EFFORT = "medium"
_DEFAULT_MAX_OUTPUT_TOKENS = 16_000
_DEFAULT_MAX_TOOL_ROUNDS = 4
_VALID_THINKING_EFFORTS = {"low", "medium", "high", "max"}
_VALID_LLM_PROVIDERS = {"anthropic", "vllm"}
_LLM_PRESET_SECONDS = ("30s", "1m", "5m", "10m", "15m")
_DURATION_TARGET_TOLERANCE_RATIO = 0.05
_SUMMARY_FILENAME = "run_summary.json"
_SELECTED_SAMPLES_JSONL_FILENAME = "selected_samples.jsonl"
_SELECTED_SAMPLES_CSV_FILENAME = "selected_samples.csv"
_SELECTED_SEGMENTS_TXT_FILENAME = "selected_segments.txt"
_COMBINED_SEGMENTS_TXT_FILENAME = "combined_segments.txt"
_SOURCE_AUDIO_TIMEOUT = (10.0, 300.0)
_SOURCE_AUDIO_HEADERS = {"User-Agent": "card-framework-sporc-llm-sampler/1.0"}
_SEAM_MAX_OFF_SPEAKER_WORDS = 3
_SEAM_MAX_OFF_SPEAKER_RATIO = 0.25
_EPISODE_RANGE_PATTERN = re.compile(
    r"^\s*\[?\s*(\d+)\s*(?:,\s*|\-\s*)(\d+)\s*\]?\s*$"
)
_MEASURE_SELECTION_DURATION_TOOL_NAME = "measure_selection_duration"
_SUBMIT_SAMPLES_TOOL_NAME = "submit_samples"
_VALID_SPEAKER_MODES = {"single_speaker_repair", "multi_speaker_exchange"}
_RAW_TRANSCRIPT_SPEAKER_MODE = "raw_transcript_span"
_RAW_EPISODE_KEY_FIELDS = (
    "mp3url",
    "mp3Url",
    "mp3_url",
    "episodeMp3Url",
    "episode_mp3_url",
    "fullPotentialOutPath",
    "potentialOutPath",
    "episodeUrl",
    "episode_url",
)
_RAW_TURN_TEXT_FIELDS = ("turnText", "content", "text", "utterance", "transcript")
_RAW_TURN_SPEAKER_FIELDS = ("speakers", "speaker", "speakerLabel", "speaker_label")
_RAW_SOURCE_MODE = "dataset_raw_turns"
_TRANSCRIBED_SOURCE_MODE = "stage1_transcribed"
_PREPARED_SOURCE_MODE = "prepared_transcript"


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
    speaker_mode: str
    speaker_labels: tuple[str, ...]
    speaker_label: str | None
    source_start_index: int
    source_end_index: int
    source_start_timestamp: str
    source_end_timestamp: str
    start_time_ms: int | None = None
    end_time_ms: int | None = None
    duration_ms: int | None = None
    source_segment_count: int | None = None


@dataclass(slots=True, frozen=True)
class _RawTranscriptTurn:
    """Represent one raw SPoRC speaker-turn row for prompt rendering."""

    index: int
    speaker_label: str
    text: str
    start_time_ms: int
    end_time_ms: int
    overlap_speakers: tuple[str, ...] = ()


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
        raw_turns_by_episode = (
            _load_selected_raw_turns(
                manifest_path=settings["dataset_manifest"],
                prepared_samples=prepared_samples,
            )
            if settings["text_only"] and not settings["transcribe_enabled"]
            else {}
        )
    except Exception as exc:
        _render_error(active_console, str(exc))
        return 1

    provider = _create_llm_sampler_provider(settings=settings)
    settings["resolved_model"] = _resolve_provider_model_label(
        provider=provider,
        fallback_model=str(settings["model"]),
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
            if settings["text_only"]:
                episode_result = _run_episode_text_only_sampling(
                    provider=provider,
                    sample=sample,
                    raw_turns=tuple(raw_turns_by_episode.get(sample.episode_key, ())),
                    episode_order=episode_order,
                    manifest_episode_number=manifest_episode_number,
                    settings=settings,
                    run_paths=run_paths,
                    console=active_console,
                )
            else:
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
    text_only = bool(getattr(args, "text_only", False))
    transcribe_enabled = bool(getattr(args, "transcribe_enabled", False))
    provider_name = str(getattr(args, "provider", _DEFAULT_PROVIDER) or "").strip().lower()
    if provider_name not in _VALID_LLM_PROVIDERS:
        raise RuntimeError("Sampler provider must be one of: anthropic, vllm.")
    preset = str(getattr(args, "duration_preset", _DEFAULT_DURATION_PRESET) or "").strip()
    if preset not in _LLM_PRESET_SECONDS:
        raise RuntimeError(
            "SPoRC llm-sample only supports duration presets 30s, 1m, 5m, 10m, and 15m."
        )

    episodes = int(getattr(args, "episodes", _DEFAULT_EPISODES) or _DEFAULT_EPISODES)
    episode_range = _parse_episode_selection(
        str(getattr(args, "select_episode", "") or "")
    )
    if episode_range is None and episodes <= 0:
        raise RuntimeError("Episode count must be greater than zero.")
    if episode_range is not None:
        episodes = episode_range[1] - episode_range[0] + 1

    requested_samples = int(getattr(args, "samples", 0) or 0)
    if requested_samples < 0:
        raise RuntimeError("Sample count must be zero or greater.")
    if text_only and requested_samples <= 0:
        raise RuntimeError(
            "SPoRC llm-sample --text-only requires --samples to be greater than zero."
        )

    words_per_minute = float(
        getattr(args, "words_per_minute", _DEFAULT_WORDS_PER_MINUTE)
        or _DEFAULT_WORDS_PER_MINUTE
    )
    sample_factor = float(
        getattr(args, "sample_factor", _DEFAULT_SAMPLE_FACTOR)
        or _DEFAULT_SAMPLE_FACTOR
    )
    if not text_only:
        if words_per_minute <= 0:
            raise RuntimeError("Words per minute must be greater than zero.")
        if sample_factor <= 0:
            raise RuntimeError("Sample factor must be greater than zero.")

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
    max_tool_rounds = int(
        getattr(args, "max_tool_rounds", _DEFAULT_MAX_TOOL_ROUNDS)
        or _DEFAULT_MAX_TOOL_ROUNDS
    )
    if max_tool_rounds <= 0:
        raise RuntimeError("Max tool rounds must be greater than zero.")

    dataset_manifest = Path(
        str(getattr(args, "dataset_manifest", DEFAULT_LONGFORM_MANIFEST_PATH) or "")
    ).resolve()
    sampling_root = Path(
        str(getattr(args, "sampling_root", DEFAULT_LLM_SAMPLING_ROOT) or "")
    ).resolve()
    audio_config_path = Path(
        str(getattr(args, "audio_config", DEFAULT_CONFIG_PATH) or DEFAULT_CONFIG_PATH)
    ).resolve()
    audio_device_override = str(getattr(args, "audio_device", "") or "").strip()
    enable_thinking = not bool(getattr(args, "disable_thinking", False))
    sampling_mode = (
        "text_only_segments"
        if text_only
        else ("fixed_samples" if requested_samples > 0 else "target_duration")
    )
    target_words = (
        None
        if requested_samples > 0 or text_only
        else _compute_target_words(
            DURATION_PRESET_SECONDS[preset],
            words_per_minute=words_per_minute,
            sample_factor=sample_factor,
        )
    )
    anthropic_api_key = str(os.environ.get("ANTHROPIC_API_KEY", "")).strip()
    vllm_url = (
        str(os.environ.get("VLLM_URL", "")).strip()
        or str(os.environ.get("CARD_FRAMEWORK_VLLM_URL", "")).strip()
        or str(os.environ.get("VLLM_BASE_URL", "")).strip()
        or str(os.environ.get("CARD_FRAMEWORK_VLLM_BASE_URL", "")).strip()
    )
    vllm_api_key = (
        str(os.environ.get("VLLM_API_KEY", "")).strip()
        or str(os.environ.get("CARD_FRAMEWORK_VLLM_API_KEY", "")).strip()
        or "EMPTY"
    )

    return {
        "dataset_manifest": dataset_manifest,
        "sampling_root": sampling_root,
        "provider": provider_name,
        "text_only": text_only,
        "transcribe_enabled": transcribe_enabled,
        "source_mode": (
            _TRANSCRIBED_SOURCE_MODE
            if transcribe_enabled
            else (_RAW_SOURCE_MODE if text_only else _PREPARED_SOURCE_MODE)
        ),
        "sampling_mode": sampling_mode,
        "duration_preset": preset,
        "preset_seconds": DURATION_PRESET_SECONDS[preset],
        "max_tool_rounds": (1 if text_only else max_tool_rounds),
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
        "duration_tolerance_ratio": float(_DURATION_TARGET_TOLERANCE_RATIO),
        "audio_config_path": audio_config_path,
        "audio_device_override": audio_device_override,
        "api_key": anthropic_api_key,
        "vllm_url": vllm_url,
        "vllm_api_key": vllm_api_key,
    }


def _validate_startup(*, settings: dict[str, Any]) -> None:
    """Fail fast on missing startup requirements."""
    manifest_path = Path(settings["dataset_manifest"])
    if not manifest_path.exists():
        raise RuntimeError(
            "SPoRC llm-sample requires an existing manifest. "
            f"Expected: {manifest_path}"
        )
    if (
        settings["text_only"]
        and not settings["transcribe_enabled"]
        and _manifest_uses_episode_assets_summary(manifest_path)
    ):
        raise RuntimeError(
            "SPoRC llm-sample --text-only cannot read "
            "`episode_assets/prepared_episodes_manifest.json` directly because "
            "that summary does not include the raw SPoRC turn-data table. Use "
            "the default transcript mode, or use a prepared dataset manifest."
        )
    if settings["provider"] == "anthropic" and not str(settings["api_key"]).strip():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Add it to the process environment or "
            "the repository-root .env file before running `llm-sample`."
        )
    if settings["provider"] == "vllm" and not str(settings["vllm_url"]).strip():
        raise RuntimeError(
            "VLLM_URL is missing. Add it to the process environment or the "
            "repository-root .env file before running `llm-sample --provider vllm`."
        )
    if settings["transcribe_enabled"]:
        audio_config_path = Path(settings["audio_config_path"])
        if not audio_config_path.exists():
            raise RuntimeError(
                "SPoRC llm-sample --transcribe-enabled requires an existing audio "
                f"config path. Expected: {audio_config_path}"
            )
        audio_runtime = _load_sampler_audio_runtime(
            config_path=audio_config_path,
            device_override=str(settings.get("audio_device_override", "") or ""),
        )
        settings["audio_cfg"] = audio_runtime["audio_cfg"]
        settings["audio_device"] = audio_runtime["audio_device"]
        settings["audio_diarizer_provider"] = audio_runtime["audio_diarizer_provider"]
        if str(audio_runtime["audio_diarizer_provider"]) == "single_speaker":
            raise RuntimeError(
                "SPoRC llm-sample --transcribe-enabled requires an overlap-capable "
                "diarizer. The resolved audio runtime downgraded to "
                "`single_speaker`, so overlapping speakers would not be preserved. "
                "Run on a host that supports the configured diarizer or change the "
                "audio config explicitly."
            )


def _load_anthropic_provider_class() -> type[Any]:
    """Import the shared Anthropic provider lazily for this command."""
    from card_framework.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider


def _load_vllm_provider_class() -> type[Any]:
    """Import the shared vLLM provider lazily for this command."""
    from card_framework.providers.vllm_provider import VLLMProvider

    return VLLMProvider


def _create_llm_sampler_provider(*, settings: dict[str, Any]) -> Any:
    """Build the configured provider instance for llm-sample."""
    if settings["provider"] == "vllm":
        provider_class = _load_vllm_provider_class()
        return provider_class(
            base_url=str(settings["vllm_url"]),
            api_key=str(settings["vllm_api_key"]),
            enable_thinking=bool(settings["enable_thinking"]),
            request_timeout_seconds=30.0,
        )

    provider_class = _load_anthropic_provider_class()
    return provider_class(
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


def _resolve_provider_model_label(*, provider: Any, fallback_model: str) -> str:
    """Return the best available display label for the active sampler model."""
    for attribute_name in ("model_id", "model"):
        value = str(getattr(provider, attribute_name, "") or "").strip()
        if value:
            return value
    return fallback_model


def _load_sampler_audio_runtime(
    *,
    config_path: Path,
    device_override: str,
) -> dict[str, Any]:
    """Load and normalize the stage-1 audio runtime used by llm-sample."""
    from copy import deepcopy

    from card_framework.audio_pipeline.runtime import resolve_device
    from card_framework.cli.calibrate import load_audio_config
    from card_framework.shared.platform_support import apply_runtime_platform_fallbacks

    audio_cfg = deepcopy(load_audio_config(REPO_ROOT, config_path=config_path))
    if device_override:
        audio_cfg["device"] = device_override
    runtime_cfg: dict[str, Any] = {"audio": audio_cfg}
    fallback_applied = apply_runtime_platform_fallbacks(runtime_cfg)
    resolved_audio_cfg = dict(runtime_cfg.get("audio", {}))
    diarization_cfg = (
        dict(resolved_audio_cfg.get("diarization", {}))
        if isinstance(resolved_audio_cfg.get("diarization", {}), dict)
        else {}
    )
    return {
        "audio_cfg": resolved_audio_cfg,
        "audio_device": resolve_device(str(resolved_audio_cfg.get("device", "auto"))),
        "audio_diarizer_provider": str(diarization_cfg.get("provider", "")).strip().lower(),
        "platform_fallback_applied": bool(fallback_applied),
    }


def _build_sampler_audio_to_script_orchestrator(audio_cfg: dict[str, Any]) -> Any:
    """Build the shared stage-1 audio orchestrator lazily for llm-sample."""
    from card_framework.audio_pipeline import build_audio_to_script_orchestrator

    return build_audio_to_script_orchestrator(audio_cfg)


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


def _load_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    """Load one sampler manifest payload from disk."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Sampler manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Sampler manifest must decode to a JSON object: {manifest_path}"
        )
    return payload


def _manifest_uses_episode_assets_summary(manifest_path: Path) -> bool:
    """Return whether the supplied manifest is an episode-assets summary."""
    payload = _load_manifest_payload(manifest_path)
    raw_episodes = payload.get("episodes")
    return isinstance(raw_episodes, list)


def _load_episode_assets_samples(
    *,
    manifest_path: Path,
    max_samples: int | None = None,
) -> tuple[PreparedBenchmarkSample, ...]:
    """Load sampler-ready samples from prepared episode-assets summary output."""
    payload = _load_manifest_payload(manifest_path)
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list):
        raise RuntimeError(
            "Episode-assets manifest must contain an `episodes` array: "
            f"{manifest_path}"
        )
    selected_rows = (
        raw_episodes
        if max_samples is None
        else raw_episodes[: max(0, int(max_samples))]
    )
    samples: list[PreparedBenchmarkSample] = []
    for index, raw_episode in enumerate(selected_rows, start=1):
        row = raw_episode if isinstance(raw_episode, dict) else {}
        transcript_path = Path(
            str(
                row.get("source_transcript_path")
                or row.get("transcript_path")
                or row.get("sampled_transcript_path")
                or ""
            )
        ).expanduser().resolve()
        if not transcript_path.exists():
            raise RuntimeError(
                "Episode-assets summary row points at a missing transcript: "
                f"{transcript_path}"
            )
        transcript_payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript = Transcript.from_mapping(transcript_payload)
        metadata = dict(transcript.metadata)
        sample_id = (
            str(row.get("sample_id") or metadata.get("sample_id") or "").strip()
            or f"episode_asset_{index:05d}"
        )
        episode_title = (
            str(row.get("episode_title") or metadata.get("episode_title") or "").strip()
            or sample_id
        )
        metadata["episode_title"] = episode_title
        for metadata_key in (
            "source_audio_url",
            "source_audio_path",
            "speaker_samples_manifest_path",
            "source_transcript_path",
        ):
            raw_value = str(row.get(metadata_key) or "").strip()
            if raw_value and not str(metadata.get(metadata_key) or "").strip():
                metadata[metadata_key] = raw_value
        total_duration_ms = max(
            (int(segment.end_time) for segment in transcript.segments),
            default=0,
        )
        samples.append(
            PreparedBenchmarkSample(
                sample_id=sample_id,
                dataset="sporc_episode_assets",
                episode_key=(
                    str(metadata.get("episode_key") or "").strip() or sample_id
                ),
                transcript_path=str(transcript_path),
                reference_summary=(
                    str(metadata.get("reference_summary") or "").strip()
                    or "Episode-assets sampled transcript."
                ),
                target_seconds=max(1, int(round(total_duration_ms / 1000.0)))
                if total_duration_ms > 0
                else 1,
                metadata=metadata,
            )
        )
    return tuple(samples)


def _load_episode_prefix(
    *,
    manifest_path: Path,
    episodes_requested: int,
) -> tuple[PreparedBenchmarkSample, ...]:
    """Load the requested manifest prefix for automatic sampling."""
    if _manifest_uses_episode_assets_summary(manifest_path):
        prepared_samples = _load_episode_assets_samples(
            manifest_path=manifest_path,
            max_samples=episodes_requested,
        )
        if len(prepared_samples) != episodes_requested:
            raise RuntimeError(
                "Requested episode prefix exceeds the available episode-assets "
                f"summary length: {episodes_requested} > {len(prepared_samples)}."
            )
        return prepared_samples

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
    if _manifest_uses_episode_assets_summary(manifest_path):
        all_samples = _load_episode_assets_samples(manifest_path=manifest_path)
        if episode_range is None:
            if episodes_requested > len(all_samples):
                raise RuntimeError(
                    "Requested episode prefix exceeds the available episode-assets "
                    f"summary length: {episodes_requested} > {len(all_samples)}."
                )
            return tuple(all_samples[:episodes_requested])
        start_index = int(episode_range[0])
        end_index = int(episode_range[1])
        if end_index > len(all_samples):
            raise RuntimeError(
                "Requested episode range exceeds the available episode-assets "
                f"summary length: {end_index} > {len(all_samples)}."
            )
        return tuple(all_samples[start_index - 1 : end_index])

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


def _load_selected_raw_turns(
    *,
    manifest_path: Path,
    prepared_samples: tuple[PreparedBenchmarkSample, ...],
) -> dict[str, tuple[_RawTranscriptTurn, ...]]:
    """Load raw SPoRC turns for the selected manifest episodes in one streaming pass."""
    manifest = load_prepared_dataset_manifest(manifest_path, include_samples=False)
    turn_data_path = manifest.turn_data_path.resolve()
    if not turn_data_path.exists():
        raise RuntimeError(f"Raw SPoRC turn data file was not found: {turn_data_path}")

    selected_episode_keys = {
        str(sample.episode_key).strip()
        for sample in prepared_samples
        if str(sample.episode_key).strip()
    }
    selected_turns: dict[str, tuple[_RawTranscriptTurn, ...]] = {
        episode_key: tuple() for episode_key in selected_episode_keys
    }
    remaining_episode_keys = set(selected_episode_keys)
    current_episode_key = ""
    current_turns: list[dict[str, Any]] | None = None

    open_fn = gzip.open if turn_data_path.suffix == ".gz" else open
    with open_fn(turn_data_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                continue
            episode_key = _raw_turn_episode_key(payload)
            if not episode_key:
                continue

            if not current_episode_key:
                current_episode_key = episode_key
                current_turns = [] if episode_key in selected_episode_keys else None
            elif episode_key != current_episode_key:
                if current_turns is not None:
                    selected_turns[current_episode_key] = _finalize_raw_turns(current_turns)
                    remaining_episode_keys.discard(current_episode_key)
                    if not remaining_episode_keys:
                        break
                current_episode_key = episode_key
                current_turns = [] if episode_key in selected_episode_keys else None

            if current_turns is None:
                continue

            text = _raw_turn_text(payload)
            speakers = _raw_turn_speakers(payload)
            if not text or not speakers:
                continue
            current_turns.append(
                {
                    "speaker_label": _render_raw_speaker_label(speakers),
                    "text": text,
                    "start_time_ms": _seconds_to_milliseconds(
                        _raw_turn_float(payload, "start", "startTime", "start_time")
                    ),
                    "end_time_ms": _seconds_to_milliseconds(
                        _raw_turn_float(payload, "end", "endTime", "end_time")
                    ),
                }
            )

    if current_turns is not None and current_episode_key in remaining_episode_keys:
        selected_turns[current_episode_key] = _finalize_raw_turns(current_turns)
        remaining_episode_keys.discard(current_episode_key)

    if remaining_episode_keys:
        raise RuntimeError(
            "No raw dataset turns were found for selected manifest episodes: "
            + ", ".join(sorted(remaining_episode_keys))
        )
    return selected_turns


def _finalize_raw_turns(raw_turns: list[dict[str, Any]]) -> tuple[_RawTranscriptTurn, ...]:
    """Sort and index raw turn payloads for prompt rendering."""
    ordered_turns = sorted(
        raw_turns,
        key=lambda item: (
            int(item["start_time_ms"]),
            int(item["end_time_ms"]),
        ),
    )
    return tuple(
        _RawTranscriptTurn(
            index=index,
            speaker_label=str(turn["speaker_label"]),
            text=str(turn["text"]),
            start_time_ms=int(turn["start_time_ms"]),
            end_time_ms=int(turn["end_time_ms"]),
            overlap_speakers=tuple(
                str(speaker).strip()
                for speaker in turn.get("overlap_speakers", ())
                if str(speaker).strip()
            ),
        )
        for index, turn in enumerate(ordered_turns)
    )


def _raw_turn_episode_key(record: dict[str, Any]) -> str:
    """Return the raw dataset episode key for one speaker-turn row."""
    for field_name in _RAW_EPISODE_KEY_FIELDS:
        value = str(record.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _raw_turn_text(record: dict[str, Any]) -> str:
    """Return raw transcript text for one speaker-turn row."""
    for field_name in _RAW_TURN_TEXT_FIELDS:
        value = record.get(field_name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _raw_turn_speakers(record: dict[str, Any]) -> list[str]:
    """Return the raw speaker labels for one speaker-turn row."""
    for field_name in _RAW_TURN_SPEAKER_FIELDS:
        raw_value = record.get(field_name)
        values = _coerce_string_list(raw_value)
        if values:
            return values
    return []


def _coerce_string_list(value: Any) -> list[str]:
    """Coerce a scalar or sequence into a list of non-empty strings."""
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [text]


def _render_raw_speaker_label(values: list[str]) -> str:
    """Render one stable speaker label for a raw turn row."""
    distinct = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if not distinct:
        return "UNKNOWN"
    if len(distinct) == 1:
        return distinct[0]
    return " + ".join(distinct)


def _raw_turn_float(record: dict[str, Any], *field_names: str) -> float:
    """Return the first parseable float among the provided raw-turn fields."""
    for field_name in field_names:
        value = record.get(field_name)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            try:
                return float(stripped)
            except ValueError:
                continue
    return 0.0


def _seconds_to_milliseconds(value: float) -> int:
    """Convert one seconds value into rounded milliseconds."""
    return max(0, int(round(float(value) * 1000.0)))


def _resolve_existing_local_path(raw_value: str) -> Path | None:
    """Resolve an existing local filesystem path from one raw metadata value."""
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.exists():
        return candidate
    return None


def _resolve_sample_source_audio_url(sample: PreparedBenchmarkSample) -> str:
    """Resolve the source audio URL/path for one prepared SPoRC sample."""
    for value in (
        str(sample.metadata.get("source_audio_url", "")).strip(),
        str(sample.metadata.get("episode_key", "")).strip(),
        str(sample.episode_key).strip(),
    ):
        if value:
            return value
    raise RuntimeError(
        f"Prepared SPoRC sample is missing source audio metadata: {sample.sample_id}"
    )


def _source_audio_destination(*, episode_dir: Path, sample: PreparedBenchmarkSample, source_audio_url: str) -> Path:
    """Return the local audio-cache destination for one sample source URL."""
    from urllib.parse import urlparse

    parsed = urlparse(source_audio_url)
    suffix = Path(parsed.path).suffix or ".mp3"
    return (episode_dir / "source_audio" / f"{sample.sample_id}{suffix}").resolve()


def _ensure_local_sampler_source_audio(
    *,
    sample: PreparedBenchmarkSample,
    episode_dir: Path,
) -> tuple[str, Path]:
    """Resolve or download the source audio used for stage-1 sampler transcription."""
    import requests

    source_audio_url = _resolve_sample_source_audio_url(sample)
    existing_source_audio_path = str(sample.metadata.get("source_audio_path", "")).strip()
    if existing_source_audio_path:
        existing_candidate = _resolve_existing_local_path(existing_source_audio_path)
        if existing_candidate is not None:
            return source_audio_url, existing_candidate.resolve()

    destination = _source_audio_destination(
        episode_dir=episode_dir,
        sample=sample,
        source_audio_url=source_audio_url,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    from urllib.parse import urlparse

    parsed = urlparse(source_audio_url)
    if parsed.scheme in {"http", "https"}:
        if destination.exists() and destination.stat().st_size > 0:
            return source_audio_url, destination
        temp_path = destination.with_suffix(destination.suffix + ".tmp")
        temp_path.unlink(missing_ok=True)
        try:
            with requests.get(
                source_audio_url,
                stream=True,
                timeout=_SOURCE_AUDIO_TIMEOUT,
                headers=_SOURCE_AUDIO_HEADERS,
            ) as response:
                response.raise_for_status()
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            temp_path.replace(destination)
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Failed to download source audio for llm-sample "
                f"'{sample.sample_id}' from {source_audio_url!r}: {exc}"
            ) from exc
        return source_audio_url, destination

    existing_candidate = _resolve_existing_local_path(source_audio_url)
    if existing_candidate is None:
        raise RuntimeError(
            "Source audio path does not exist for llm-sample "
            f"{sample.sample_id}: {source_audio_url}"
        )
    return source_audio_url, existing_candidate.resolve()


def _ensure_stage_one_source_transcript(
    *,
    sample: PreparedBenchmarkSample,
    episode_title: str,
    episode_dir: Path,
    settings: dict[str, Any],
    console: Console,
) -> tuple[Transcript, Path, Path]:
    """Generate or reuse the stage-1 transcript used as the sampler source."""
    source_root = (episode_dir / "stage1_source").resolve()
    transcript_path = (source_root / "transcript.json").resolve()
    audio_stage_root = (source_root / "audio_stage").resolve()
    if transcript_path.exists():
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript = Transcript.from_mapping(payload)
        source_audio_value = str(transcript.metadata.get("source_audio_path", "")).strip()
        source_audio_path = _resolve_existing_local_path(source_audio_value)
        if source_audio_path is None:
            _, source_audio_path = _ensure_local_sampler_source_audio(
                sample=sample,
                episode_dir=episode_dir,
            )
        console.print(
            f"[cyan]Reusing cached stage-1 sampler transcript for {sample.sample_id}.[/cyan]"
        )
        return transcript, transcript_path, source_audio_path

    source_root.mkdir(parents=True, exist_ok=True)
    source_audio_url, source_audio_path = _ensure_local_sampler_source_audio(
        sample=sample,
        episode_dir=episode_dir,
    )
    console.print(
        (
            f"[cyan]Running stage-1 source generation for {sample.sample_id}: "
            "source separation, Granite ASR, and diarization.[/cyan]"
        )
    )
    orchestrator = _build_sampler_audio_to_script_orchestrator(
        dict(settings["audio_cfg"])
    )
    result = orchestrator.run(
        input_audio_path=source_audio_path,
        output_transcript_path=transcript_path,
        work_dir=audio_stage_root,
        device=str(settings["audio_device"]),
        metadata_overrides={
            "dataset": "sporc",
            "episode_key": sample.episode_key,
            "episode_title": episode_title,
            "source_audio_url": source_audio_url,
            "llm_sampler_source_mode": _TRANSCRIBED_SOURCE_MODE,
        },
    )
    return (
        Transcript.from_mapping(result.transcript_payload),
        transcript_path,
        source_audio_path,
    )


def _transcript_to_source_turns(transcript: Transcript) -> tuple[_RawTranscriptTurn, ...]:
    """Project a transcript JSON payload into text-only source-turn rows."""
    turns: list[_RawTranscriptTurn] = []
    for index, segment in enumerate(transcript.segments):
        text = str(segment.text or "").strip()
        if not text:
            continue
        turns.append(
            _RawTranscriptTurn(
                index=index,
                speaker_label=str(segment.speaker or "UNKNOWN").strip() or "UNKNOWN",
                text=text,
                start_time_ms=int(segment.start_time),
                end_time_ms=int(segment.end_time),
                overlap_speakers=tuple(
                    str(speaker).strip()
                    for speaker in (
                        segment.extras.get("overlap_speakers", [])
                        if isinstance(segment.extras, dict)
                        else []
                    )
                    if str(speaker).strip()
                ),
            )
        )
    return tuple(turns)


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
        "text_only": bool(settings["text_only"]),
        "transcribe_enabled": bool(settings["transcribe_enabled"]),
        "source_mode": str(settings["source_mode"]),
        "duration_preset": str(settings["duration_preset"]),
        "preset_seconds": int(settings["preset_seconds"]),
        "target_seconds_per_episode": (
            None if settings["text_only"] else int(settings["preset_seconds"])
        ),
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
        "provider": str(settings["provider"]),
        "model": str(settings.get("resolved_model") or settings["model"]),
        "requested_model": str(settings["model"]),
        "audio_config_path": (
            str(settings["audio_config_path"])
            if settings["transcribe_enabled"]
            else ""
        ),
        "audio_device": (
            str(settings.get("audio_device") or settings.get("audio_device_override") or "")
            if settings["transcribe_enabled"]
            else ""
        ),
        "audio_diarizer_provider": (
            str(settings.get("audio_diarizer_provider") or "")
            if settings["transcribe_enabled"]
            else ""
        ),
        "thinking_enabled": bool(settings["enable_thinking"]),
        "thinking_effort": str(settings["thinking_effort"]),
        "max_output_tokens": int(settings["max_output_tokens"]),
        "max_tool_rounds": int(settings["max_tool_rounds"]),
        "duration_tolerance_ratio": float(_DURATION_TARGET_TOLERANCE_RATIO),
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
            "text_only",
            "transcribe_enabled",
            "source_mode",
            "duration_preset",
            "target_seconds",
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
            "speaker_mode",
            "speaker_labels",
            "source_start_index",
            "source_end_index",
            "source_start_timestamp",
            "source_end_timestamp",
            "start_time",
            "end_time",
            "duration_ms",
            "source_segment_count",
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


def _run_episode_text_only_sampling(
    *,
    provider: Any,
    sample: PreparedBenchmarkSample,
    raw_turns: tuple[_RawTranscriptTurn, ...],
    episode_order: int,
    manifest_episode_number: int,
    settings: dict[str, Any],
    run_paths: _RunPaths,
    console: Console,
) -> dict[str, Any]:
    """Execute one text-only raw-transcript sampling pass for one episode."""
    episode_title = (
        str(sample.metadata.get("episode_title", "") or "").strip() or sample.sample_id
    )
    episode_dir = (run_paths.episodes_root / sample.sample_id).resolve()
    episode_dir.mkdir(parents=True, exist_ok=True)
    request_path = (episode_dir / "request.json").resolve()
    stream_path = (episode_dir / "stream.jsonl").resolve()
    response_path = (episode_dir / "response.json").resolve()
    selected_segments_path = (episode_dir / _SELECTED_SEGMENTS_TXT_FILENAME).resolve()
    combined_segments_path = (episode_dir / _COMBINED_SEGMENTS_TXT_FILENAME).resolve()
    transcript_path = Path(sample.transcript_path).resolve()
    source_label = "Raw transcript"
    source_guidance = (
        "Each line below is a raw speaker-turn fragment. "
        "If one spoken line is split across adjacent fragments from the same speaker, "
        "you may select that contiguous index range as one segment."
    )
    effective_raw_turns = raw_turns
    try:
        if settings["transcribe_enabled"]:
            source_transcript, transcript_path, _ = _ensure_stage_one_source_transcript(
                sample=sample,
                episode_title=episode_title,
                episode_dir=episode_dir,
                settings=settings,
                console=console,
            )
            effective_raw_turns = _transcript_to_source_turns(source_transcript)
            source_label = "Stage-1 transcript"
            source_guidance = (
                "Each line below is a diarized stage-1 speaker segment produced from "
                "source-separated audio. If one spoken line is split across adjacent "
                "same-speaker segments, you may select that contiguous index range as one segment."
            )
    except Exception as exc:
        failure_message = str(exc)
        _write_episode_response(
            response_path=response_path,
            run_id=run_paths.run_root.name,
            episode_order=episode_order,
            manifest_episode_number=manifest_episode_number,
            sample_id=sample.sample_id,
            episode_title=episode_title,
            status=_classify_episode_error(failure_message),
            attempts_payload=[],
            selected_rows=[],
            selected_segments_path=None,
            selected_words=0,
            selected_duration_ms=0,
            error=failure_message,
            combined_segments_path=None,
        )
        _render_episode_failure(console, validation_message=failure_message)
        return {
            "summary": {
                "episode_order": episode_order,
                "manifest_episode_number": manifest_episode_number,
                "sample_id": sample.sample_id,
                "episode_key": sample.episode_key,
                "episode_title": episode_title,
                "transcript_path": str(transcript_path),
                "transcript_word_count": 0,
                "raw_turn_count": 0,
                "source_mode": str(settings["source_mode"]),
                "target_seconds": None,
                "target_words": None,
                "requested_samples": settings["requested_samples"],
                "selected_words": 0,
                "selected_duration_ms": 0,
                "selected_duration_seconds": 0.0,
                "selection_count": 0,
                "status": _classify_episode_error(failure_message),
                "request_path": str(request_path),
                "response_path": str(response_path),
                "stream_path": str(stream_path),
                "selected_segments_path": "",
                "combined_segments_path": "",
            },
            "selected_rows": [],
        }

    if not effective_raw_turns:
        failure_message = (
            "No usable stage-1 transcript segments were found for the episode."
            if settings["transcribe_enabled"]
            else "No usable raw transcript turns were found for the episode."
        )
        _write_episode_response(
            response_path=response_path,
            run_id=run_paths.run_root.name,
            episode_order=episode_order,
            manifest_episode_number=manifest_episode_number,
            sample_id=sample.sample_id,
            episode_title=episode_title,
            status="failed_validation",
            attempts_payload=[],
            selected_rows=[],
            selected_segments_path=None,
            selected_words=0,
            selected_duration_ms=0,
            error=failure_message,
            combined_segments_path=None,
        )
        _render_episode_failure(console, validation_message=failure_message)
        return {
            "summary": {
                "episode_order": episode_order,
                "manifest_episode_number": manifest_episode_number,
                "sample_id": sample.sample_id,
                "episode_key": sample.episode_key,
                "episode_title": episode_title,
                "transcript_path": str(transcript_path),
                "transcript_word_count": 0,
                "raw_turn_count": 0,
                "source_mode": str(settings["source_mode"]),
                "target_seconds": None,
                "target_words": None,
                "requested_samples": settings["requested_samples"],
                "selected_words": 0,
                "selected_duration_ms": 0,
                "selected_duration_seconds": 0.0,
                "selection_count": 0,
                "status": "failed_validation",
                "request_path": str(request_path),
                "response_path": str(response_path),
                "stream_path": str(stream_path),
                "selected_segments_path": "",
                "combined_segments_path": "",
            },
            "selected_rows": [],
        }

    transcript_word_count = sum(count_words(turn.text) for turn in effective_raw_turns)
    _render_episode_header(
        console,
        manifest_episode_number=manifest_episode_number,
        episode_title=episode_title,
        sample_id=sample.sample_id,
        transcript_word_count=transcript_word_count,
        target_label=_episode_target_label(settings),
    )

    system_prompt = _build_text_only_system_prompt(
        requested_samples=int(settings["requested_samples"] or 0),
        source_mode=str(settings["source_mode"]),
    )
    base_user_prompt = _build_text_only_user_prompt(
        sample=sample,
        raw_turns=effective_raw_turns,
        requested_samples=int(settings["requested_samples"] or 0),
        source_label=source_label,
        source_guidance=source_guidance,
    )
    tool_schema = _submit_text_only_segments_tool_schema()
    tool_choice = {"type": "function", "function": {"name": _SUBMIT_SAMPLES_TOOL_NAME}}
    request_payload = {
        "attempt_index": 1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_user_prompt},
        ],
        "tools": [tool_schema],
        "tool_choice": tool_choice,
        "max_output_tokens": int(settings["max_output_tokens"]),
        "thinking_enabled": bool(settings["enable_thinking"]),
        "thinking_effort": str(settings["thinking_effort"]),
        "model": str(settings.get("resolved_model") or settings["model"]),
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
        target_seconds=None,
        target_words=None,
        requested_samples=settings["requested_samples"],
        request_payload=request_payload,
    )

    attempts_payload: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    stream_recorder = _StreamRecorderCallback(
        stream_path=stream_path,
        attempt_index=1,
    )
    provider.set_response_callback(
        CompositeResponseCallback(
            RichConsoleResponseCallback(),
            stream_recorder,
        )
    )
    if hasattr(provider, "last_response_details"):
        provider.last_response_details = {}

    assistant_message: Any | None = None
    try:
        try:
            assistant_message = provider.chat(
                messages=request_payload["messages"],
                tools=[tool_schema],
                tool_choice=tool_choice,
                max_tokens=int(settings["max_output_tokens"]),
            )
        except Exception as exc:
            validation = {
                "status": _classify_episode_error(str(exc)),
                "message": str(exc),
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
            attempts_payload.append(
                {
                    "attempt_index": 1,
                    "assistant_message": {},
                    "provider_trace": dict(getattr(provider, "last_response_details", {})),
                    "tool_results": [],
                    "validation": validation,
                }
            )
            _write_episode_response(
                response_path=response_path,
                run_id=run_paths.run_root.name,
                episode_order=episode_order,
                manifest_episode_number=manifest_episode_number,
                sample_id=sample.sample_id,
                episode_title=episode_title,
                status=str(validation["status"]),
                attempts_payload=attempts_payload,
                selected_rows=[],
                selected_segments_path=None,
                selected_words=0,
                selected_duration_ms=0,
                combined_segments_path=None,
            )
            _render_episode_failure(console, validation_message=str(validation["message"]))
            return {
                "summary": {
                    "episode_order": episode_order,
                    "manifest_episode_number": manifest_episode_number,
                    "sample_id": sample.sample_id,
                    "episode_key": sample.episode_key,
                    "episode_title": episode_title,
                    "transcript_path": str(transcript_path),
                    "transcript_word_count": transcript_word_count,
                    "raw_turn_count": len(effective_raw_turns),
                    "source_mode": str(settings["source_mode"]),
                    "target_seconds": None,
                    "target_words": None,
                    "requested_samples": settings["requested_samples"],
                    "selected_words": 0,
                    "selected_duration_ms": 0,
                    "selected_duration_seconds": 0.0,
                    "selection_count": 0,
                    "status": str(validation["status"]),
                    "request_path": str(request_path),
                    "response_path": str(response_path),
                    "stream_path": str(stream_path),
                    "selected_segments_path": "",
                    "combined_segments_path": "",
                },
                "selected_rows": [],
            }
    finally:
        stream_recorder.close()

    assistant_payload = assistant_message.model_dump() if assistant_message is not None else {}
    parsed_samples, validation = _validate_text_only_submission(
        assistant_message=assistant_payload,
        raw_turns=effective_raw_turns,
        episode_key=sample.episode_key,
        requested_samples=int(settings["requested_samples"] or 0),
    )
    attempts_payload.append(
        {
            "attempt_index": 1,
            "assistant_message": assistant_payload,
            "provider_trace": dict(getattr(provider, "last_response_details", {})),
            "tool_results": [],
            "validation": validation,
        }
    )

    final_status = str(validation["status"])
    final_selected_words = int(validation.get("actual_selected_words") or 0)
    final_selected_duration_ms = int(validation.get("actual_selected_duration_ms") or 0)
    if final_status == "completed":
        selected_rows = _build_selected_rows(
            run_id=run_paths.run_root.name,
            text_only=True,
            transcribe_enabled=bool(settings["transcribe_enabled"]),
            source_mode=str(settings["source_mode"]),
            duration_preset=str(settings["duration_preset"]),
            target_seconds=None,
            target_words=None,
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
        _write_combined_segments_text(
            output_path=combined_segments_path,
            parsed_samples=parsed_samples,
        )
        _write_episode_response(
            response_path=response_path,
            run_id=run_paths.run_root.name,
            episode_order=episode_order,
            manifest_episode_number=manifest_episode_number,
            sample_id=sample.sample_id,
            episode_title=episode_title,
            status=final_status,
            attempts_payload=attempts_payload,
            selected_rows=selected_rows,
            selected_segments_path=selected_segments_path,
            selected_words=final_selected_words,
            selected_duration_ms=final_selected_duration_ms,
            combined_segments_path=combined_segments_path,
        )
        _render_episode_success(
            console,
            selected_words=final_selected_words,
            selected_duration_ms=final_selected_duration_ms,
            selection_count=len(parsed_samples),
        )
    else:
        _write_episode_response(
            response_path=response_path,
            run_id=run_paths.run_root.name,
            episode_order=episode_order,
            manifest_episode_number=manifest_episode_number,
            sample_id=sample.sample_id,
            episode_title=episode_title,
            status=final_status,
            attempts_payload=attempts_payload,
            selected_rows=[],
            selected_segments_path=None,
            selected_words=final_selected_words,
            selected_duration_ms=final_selected_duration_ms,
            combined_segments_path=None,
        )
        _render_episode_failure(console, validation_message=str(validation["message"]))

    return {
        "summary": {
            "episode_order": episode_order,
            "manifest_episode_number": manifest_episode_number,
            "sample_id": sample.sample_id,
            "episode_key": sample.episode_key,
            "episode_title": episode_title,
            "transcript_path": str(transcript_path),
            "transcript_word_count": transcript_word_count,
            "raw_turn_count": len(effective_raw_turns),
            "source_mode": str(settings["source_mode"]),
            "target_seconds": None,
            "target_words": None,
            "requested_samples": settings["requested_samples"],
            "selected_words": final_selected_words,
            "selected_duration_ms": final_selected_duration_ms,
            "selected_duration_seconds": round(final_selected_duration_ms / 1000.0, 3),
            "selection_count": len(selected_rows),
            "status": final_status,
            "request_path": str(request_path),
            "response_path": str(response_path),
            "stream_path": str(stream_path),
            "selected_segments_path": (
                str(selected_segments_path) if selected_rows else ""
            ),
            "combined_segments_path": (
                str(combined_segments_path) if selected_rows else ""
            ),
        },
        "selected_rows": selected_rows,
    }


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
    episode_title = (
        str(sample.metadata.get("episode_title", "") or "").strip() or sample.sample_id
    )
    episode_dir = (run_paths.episodes_root / sample.sample_id).resolve()
    episode_dir.mkdir(parents=True, exist_ok=True)
    request_path = (episode_dir / "request.json").resolve()
    stream_path = (episode_dir / "stream.jsonl").resolve()
    response_path = (episode_dir / "response.json").resolve()
    selected_segments_path = (episode_dir / _SELECTED_SEGMENTS_TXT_FILENAME).resolve()
    transcript_path = Path(sample.transcript_path).resolve()
    transcript_source_description = "Prepared transcript source."

    try:
        if settings["transcribe_enabled"]:
            transcript, transcript_path, _ = _ensure_stage_one_source_transcript(
                sample=sample,
                episode_title=episode_title,
                episode_dir=episode_dir,
                settings=settings,
                console=console,
            )
            transcript_source_description = (
                "Transcript source: fresh stage-1 transcript built from source-separated "
                "audio, Granite ASR, and diarization."
            )
        else:
            transcript_payload = json.loads(transcript_path.read_text(encoding="utf-8"))
            transcript = Transcript.from_mapping(transcript_payload)
        transcript_word_count = sum(
            count_words(segment.text) for segment in transcript.segments
        )
        transcript_text = _render_transcript_for_prompt(transcript)
    except Exception as exc:
        failure_status = _classify_episode_error(str(exc))
        _write_episode_response(
            response_path=response_path,
            run_id=run_paths.run_root.name,
            episode_order=episode_order,
            manifest_episode_number=manifest_episode_number,
            sample_id=sample.sample_id,
            episode_title=episode_title,
            status=failure_status,
            attempts_payload=[],
            selected_rows=[],
            selected_segments_path=None,
            selected_words=0,
            selected_duration_ms=0,
            error=str(exc),
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
                "source_mode": str(settings["source_mode"]),
                "target_seconds": int(settings["preset_seconds"]),
                "target_words": settings["target_words"],
                "requested_samples": settings["requested_samples"],
                "selected_words": 0,
                "selected_duration_ms": 0,
                "selected_duration_seconds": 0.0,
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
        target_seconds=int(settings["preset_seconds"]),
        target_words=settings["target_words"],
        requested_samples=settings["requested_samples"],
    )
    base_user_prompt = _build_user_prompt(
        sample=sample,
        transcript_text=transcript_text,
        sampling_mode=str(settings["sampling_mode"]),
        target_seconds=int(settings["preset_seconds"]),
        target_words=settings["target_words"],
        requested_samples=settings["requested_samples"],
        transcript_word_count=transcript_word_count,
        source_description=transcript_source_description,
    )
    tool_schemas = [
        _submit_samples_tool_schema(),
        _measure_selection_duration_tool_schema(),
    ]
    tool_choice = {"type": "auto"}

    attempts_payload: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    final_status = "failed_validation"
    final_selected_words = 0
    final_selected_duration_ms = 0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_user_prompt},
    ]
    max_tool_rounds = int(settings["max_tool_rounds"])
    for attempt_index in range(1, max_tool_rounds + 1):
        request_payload = {
            "attempt_index": attempt_index,
            "messages": messages,
            "tools": tool_schemas,
            "tool_choice": tool_choice,
            "max_output_tokens": int(settings["max_output_tokens"]),
            "thinking_enabled": bool(settings["enable_thinking"]),
            "thinking_effort": str(settings["thinking_effort"]),
            "model": str(settings.get("resolved_model") or settings["model"]),
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
            target_seconds=int(settings["preset_seconds"]),
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

        assistant_message: Any | None = None
        try:
            try:
                assistant_message = provider.chat(
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice=tool_choice,
                    max_tokens=int(settings["max_output_tokens"]),
                )
            except Exception as exc:
                validation = {
                    "status": _classify_episode_error(str(exc)),
                    "message": str(exc),
                    "actual_selected_words": 0,
                    "actual_selected_duration_ms": 0,
                }
                attempts_payload.append(
                    {
                        "attempt_index": attempt_index,
                        "assistant_message": {},
                        "provider_trace": dict(
                            getattr(provider, "last_response_details", {})
                        ),
                        "tool_results": [],
                        "validation": validation,
                    }
                )
                final_status = str(validation["status"])
                _write_episode_response(
                    response_path=response_path,
                    run_id=run_paths.run_root.name,
                    episode_order=episode_order,
                    manifest_episode_number=manifest_episode_number,
                    sample_id=sample.sample_id,
                    episode_title=episode_title,
                    status=final_status,
                    attempts_payload=attempts_payload,
                    selected_rows=selected_rows,
                    selected_segments_path=None,
                    selected_words=final_selected_words,
                    selected_duration_ms=final_selected_duration_ms,
                )
                _render_episode_failure(
                    console,
                    validation_message=str(validation["message"]),
                )
                break
        finally:
            stream_recorder.close()

        if assistant_message is None:
            break

        assistant_payload = assistant_message.model_dump()
        submit_tool_call = _find_tool_call_by_name(
            assistant_message=assistant_payload,
            tool_name=_SUBMIT_SAMPLES_TOOL_NAME,
        )
        if submit_tool_call is not None:
            parsed_samples, validation = _validate_submission(
                assistant_message=assistant_payload,
                transcript=transcript,
                episode_key=sample.episode_key,
                sampling_mode=str(settings["sampling_mode"]),
                target_seconds=int(settings["preset_seconds"]),
                duration_tolerance_ratio=float(
                    settings["duration_tolerance_ratio"]
                ),
                target_words=settings["target_words"],
                requested_samples=settings["requested_samples"],
            )
            attempts_payload.append(
                {
                    "attempt_index": attempt_index,
                    "assistant_message": assistant_payload,
                    "provider_trace": dict(
                        getattr(provider, "last_response_details", {})
                    ),
                    "tool_results": [],
                    "validation": validation,
                }
            )

            if validation["status"] == "completed":
                final_status = "completed"
                final_selected_words = int(validation["actual_selected_words"])
                final_selected_duration_ms = int(
                    validation["actual_selected_duration_ms"]
                )
                selected_rows = _build_selected_rows(
                    run_id=run_paths.run_root.name,
                    text_only=False,
                    transcribe_enabled=bool(settings["transcribe_enabled"]),
                    source_mode=str(settings["source_mode"]),
                    duration_preset=str(settings["duration_preset"]),
                    target_seconds=int(settings["preset_seconds"]),
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
                _write_episode_response(
                    response_path=response_path,
                    run_id=run_paths.run_root.name,
                    episode_order=episode_order,
                    manifest_episode_number=manifest_episode_number,
                    sample_id=sample.sample_id,
                    episode_title=episode_title,
                    status=final_status,
                    attempts_payload=attempts_payload,
                    selected_rows=selected_rows,
                    selected_segments_path=selected_segments_path,
                    selected_words=final_selected_words,
                    selected_duration_ms=final_selected_duration_ms,
                )
                _render_episode_success(
                    console,
                    selected_words=final_selected_words,
                    selected_duration_ms=final_selected_duration_ms,
                    selection_count=len(parsed_samples),
                )
            else:
                final_status = str(validation["status"])
                final_selected_duration_ms = int(
                    validation.get("actual_selected_duration_ms") or 0
                )
                _write_episode_response(
                    response_path=response_path,
                    run_id=run_paths.run_root.name,
                    episode_order=episode_order,
                    manifest_episode_number=manifest_episode_number,
                    sample_id=sample.sample_id,
                    episode_title=episode_title,
                    status=final_status,
                    attempts_payload=attempts_payload,
                    selected_rows=selected_rows,
                    selected_segments_path=None,
                    selected_words=final_selected_words,
                    selected_duration_ms=final_selected_duration_ms,
                )
                _render_episode_failure(
                    console,
                    validation_message=str(validation["message"]),
                )
            break

        measure_tool_calls = _find_tool_calls_by_name(
            assistant_message=assistant_payload,
            tool_name=_MEASURE_SELECTION_DURATION_TOOL_NAME,
        )
        unsupported_tool_names = _unsupported_tool_names(assistant_payload)
        if unsupported_tool_names:
            validation = {
                "status": "failed_validation",
                "message": (
                    "Model called unsupported tools: "
                    + ", ".join(sorted(unsupported_tool_names))
                ),
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
            attempts_payload.append(
                {
                    "attempt_index": attempt_index,
                    "assistant_message": assistant_payload,
                    "provider_trace": dict(
                        getattr(provider, "last_response_details", {})
                    ),
                    "tool_results": [],
                    "validation": validation,
                }
            )
            final_status = str(validation["status"])
            _write_episode_response(
                response_path=response_path,
                run_id=run_paths.run_root.name,
                episode_order=episode_order,
                manifest_episode_number=manifest_episode_number,
                sample_id=sample.sample_id,
                episode_title=episode_title,
                status=final_status,
                attempts_payload=attempts_payload,
                selected_rows=selected_rows,
                selected_segments_path=None,
                selected_words=final_selected_words,
                selected_duration_ms=final_selected_duration_ms,
            )
            _render_episode_failure(
                console,
                validation_message=str(validation["message"]),
            )
            break
        if not measure_tool_calls:
            validation = {
                "status": "failed_validation",
                "message": (
                    "Model did not call submit_samples or "
                    "measure_selection_duration."
                ),
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
            attempts_payload.append(
                {
                    "attempt_index": attempt_index,
                    "assistant_message": assistant_payload,
                    "provider_trace": dict(
                        getattr(provider, "last_response_details", {})
                    ),
                    "tool_results": [],
                    "validation": validation,
                }
            )
            final_status = str(validation["status"])
            _write_episode_response(
                response_path=response_path,
                run_id=run_paths.run_root.name,
                episode_order=episode_order,
                manifest_episode_number=manifest_episode_number,
                sample_id=sample.sample_id,
                episode_title=episode_title,
                status=final_status,
                attempts_payload=attempts_payload,
                selected_rows=selected_rows,
                selected_segments_path=None,
                selected_words=final_selected_words,
                selected_duration_ms=final_selected_duration_ms,
            )
            _render_episode_failure(
                console,
                validation_message=str(validation["message"]),
            )
            break

        tool_results, tool_messages = _execute_measure_selection_duration_tool_calls(
            tool_calls=measure_tool_calls,
            transcript=transcript,
            target_seconds=int(settings["preset_seconds"]),
            duration_tolerance_ratio=float(settings["duration_tolerance_ratio"]),
        )
        final_selected_duration_ms = int(
            tool_results[-1]["result"].get("total_duration_ms") or 0
        )
        attempts_payload.append(
            {
                "attempt_index": attempt_index,
                "assistant_message": assistant_payload,
                "provider_trace": dict(getattr(provider, "last_response_details", {})),
                "tool_results": tool_results,
                "validation": {
                    "status": "tool_result_emitted",
                    "message": "Returned local duration measurements.",
                    "actual_selected_words": 0,
                    "actual_selected_duration_ms": final_selected_duration_ms,
                },
            }
        )
        if attempt_index == max_tool_rounds:
            parsed_samples, validation = _validate_last_measured_selection(
                arguments=(
                    tool_results[-1]["arguments"]
                    if tool_results
                    else {}
                ),
                transcript=transcript,
                sampling_mode=str(settings["sampling_mode"]),
                target_seconds=int(settings["preset_seconds"]),
                duration_tolerance_ratio=float(
                    settings["duration_tolerance_ratio"]
                ),
                requested_samples=settings["requested_samples"],
            )
            if validation["status"] == "completed":
                attempts_payload[-1]["validation"] = {
                    **validation,
                    "message": (
                        "Accepted the last locally measured selection after "
                        "the model reached the tool-round limit before calling "
                        "submit_samples."
                    ),
                }
                final_status = "completed"
                final_selected_words = int(validation["actual_selected_words"])
                final_selected_duration_ms = int(
                    validation["actual_selected_duration_ms"]
                )
                selected_rows = _build_selected_rows(
                    run_id=run_paths.run_root.name,
                    text_only=False,
                    transcribe_enabled=bool(settings["transcribe_enabled"]),
                    source_mode=str(settings["source_mode"]),
                    duration_preset=str(settings["duration_preset"]),
                    target_seconds=int(settings["preset_seconds"]),
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
                _write_episode_response(
                    response_path=response_path,
                    run_id=run_paths.run_root.name,
                    episode_order=episode_order,
                    manifest_episode_number=manifest_episode_number,
                    sample_id=sample.sample_id,
                    episode_title=episode_title,
                    status=final_status,
                    attempts_payload=attempts_payload,
                    selected_rows=selected_rows,
                    selected_segments_path=selected_segments_path,
                    selected_words=final_selected_words,
                    selected_duration_ms=final_selected_duration_ms,
                )
                _render_episode_success(
                    console,
                    selected_words=final_selected_words,
                    selected_duration_ms=final_selected_duration_ms,
                    selection_count=len(parsed_samples),
                )
                break
            attempts_payload[-1]["validation"] = validation
            final_status = str(validation["status"])
            final_selected_duration_ms = int(
                validation.get("actual_selected_duration_ms") or 0
            )
            _write_episode_response(
                response_path=response_path,
                run_id=run_paths.run_root.name,
                episode_order=episode_order,
                manifest_episode_number=manifest_episode_number,
                sample_id=sample.sample_id,
                episode_title=episode_title,
                status=final_status,
                attempts_payload=attempts_payload,
                selected_rows=selected_rows,
                selected_segments_path=None,
                selected_words=final_selected_words,
                selected_duration_ms=final_selected_duration_ms,
            )
            _render_episode_failure(
                console,
                validation_message=str(validation["message"]),
            )
            break
        _write_episode_response(
            response_path=response_path,
            run_id=run_paths.run_root.name,
            episode_order=episode_order,
            manifest_episode_number=manifest_episode_number,
            sample_id=sample.sample_id,
            episode_title=episode_title,
            status="running",
            attempts_payload=attempts_payload,
            selected_rows=selected_rows,
            selected_segments_path=None,
            selected_words=final_selected_words,
            selected_duration_ms=final_selected_duration_ms,
        )
        messages = [
            *messages,
            assistant_payload,
            *tool_messages,
        ]
    else:
        final_status = "failed_validation"
        validation_message = (
            "Model did not submit samples within the allowed tool rounds. "
            f"Max rounds={max_tool_rounds}."
        )
        attempts_payload.append(
            {
                "attempt_index": max_tool_rounds,
                "assistant_message": {},
                "provider_trace": {},
                "tool_results": [],
                "validation": {
                    "status": final_status,
                    "message": validation_message,
                    "actual_selected_words": 0,
                    "actual_selected_duration_ms": final_selected_duration_ms,
                },
            }
        )
        _write_episode_response(
            response_path=response_path,
            run_id=run_paths.run_root.name,
            episode_order=episode_order,
            manifest_episode_number=manifest_episode_number,
            sample_id=sample.sample_id,
            episode_title=episode_title,
            status=final_status,
            attempts_payload=attempts_payload,
            selected_rows=selected_rows,
            selected_segments_path=None,
            selected_words=final_selected_words,
            selected_duration_ms=final_selected_duration_ms,
        )
        _render_episode_failure(console, validation_message=validation_message)

    return {
        "summary": {
            "episode_order": episode_order,
            "manifest_episode_number": manifest_episode_number,
            "sample_id": sample.sample_id,
            "episode_key": sample.episode_key,
            "episode_title": episode_title,
            "transcript_path": str(transcript_path),
            "transcript_word_count": transcript_word_count,
            "source_mode": str(settings["source_mode"]),
            "target_seconds": int(settings["preset_seconds"]),
            "target_words": settings["target_words"],
            "requested_samples": settings["requested_samples"],
            "selected_words": final_selected_words,
            "selected_duration_ms": final_selected_duration_ms,
            "selected_duration_seconds": round(
                final_selected_duration_ms / 1000.0,
                3,
            ),
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
    target_seconds: int | None,
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
            "target_seconds": target_seconds,
            "target_words": target_words,
            "requested_samples": requested_samples,
            "attempt_requests": [request_payload],
        }
    request_path.write_text(
        json.dumps(request_doc, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_episode_response(
    *,
    response_path: Path,
    run_id: str,
    episode_order: int,
    manifest_episode_number: int,
    sample_id: str,
    episode_title: str,
    status: str,
    attempts_payload: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    selected_segments_path: Path | None,
    selected_words: int,
    selected_duration_ms: int,
    error: str = "",
    combined_segments_path: Path | None = None,
) -> None:
    """Persist the current per-episode response document."""
    response_payload = {
        "run_id": run_id,
        "episode_order": episode_order,
        "manifest_episode_number": manifest_episode_number,
        "sample_id": sample_id,
        "episode_title": episode_title,
        "status": status,
        "attempts": attempts_payload,
        "selected_words": selected_words,
        "selected_duration_ms": selected_duration_ms,
        "selected_duration_seconds": round(selected_duration_ms / 1000.0, 3),
        "selected_samples": selected_rows,
        "selected_segments_path": (
            str(selected_segments_path)
            if selected_segments_path is not None and selected_rows
            else ""
        ),
        "combined_segments_path": (
            str(combined_segments_path)
            if combined_segments_path is not None and selected_rows
            else ""
        ),
    }
    if error:
        response_payload["error"] = error
    response_path.write_text(
        json.dumps(response_payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _assistant_tool_calls(assistant_message: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized tool-call payloads from one assistant message."""
    raw_tool_calls = assistant_message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []
    return [tool_call for tool_call in raw_tool_calls if isinstance(tool_call, dict)]


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    """Return the function name for one normalized tool call."""
    function_payload = tool_call.get("function", {})
    function = function_payload if isinstance(function_payload, dict) else {}
    return str(function.get("name") or "").strip()


def _find_tool_call_by_name(
    *,
    assistant_message: dict[str, Any],
    tool_name: str,
) -> dict[str, Any] | None:
    """Return the first tool call that matches the requested name."""
    for tool_call in _assistant_tool_calls(assistant_message):
        if _tool_call_name(tool_call) == tool_name:
            return tool_call
    return None


def _find_tool_calls_by_name(
    *,
    assistant_message: dict[str, Any],
    tool_name: str,
) -> list[dict[str, Any]]:
    """Return all tool calls that match the requested name."""
    return [
        tool_call
        for tool_call in _assistant_tool_calls(assistant_message)
        if _tool_call_name(tool_call) == tool_name
    ]


def _unsupported_tool_names(assistant_message: dict[str, Any]) -> set[str]:
    """Return tool names that are not part of the sampler's tool contract."""
    allowed_tool_names = {
        _MEASURE_SELECTION_DURATION_TOOL_NAME,
        _SUBMIT_SAMPLES_TOOL_NAME,
    }
    unsupported: set[str] = set()
    for tool_call in _assistant_tool_calls(assistant_message):
        tool_name = _tool_call_name(tool_call)
        if tool_name and tool_name not in allowed_tool_names:
            unsupported.add(tool_name)
    return unsupported


def _execute_measure_selection_duration_tool_calls(
    *,
    tool_calls: list[dict[str, Any]],
    transcript: Transcript,
    target_seconds: int,
    duration_tolerance_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute one or more local duration-measurement tool calls."""
    tool_results: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function_payload = tool_call.get("function", {})
        function = function_payload if isinstance(function_payload, dict) else {}
        raw_arguments = str(function.get("arguments") or "{}")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
        result = _measure_selection_duration(
            arguments=arguments if isinstance(arguments, dict) else {},
            transcript=transcript,
            target_seconds=target_seconds,
            duration_tolerance_ratio=duration_tolerance_ratio,
        )
        tool_results.append(
            {
                "tool_call_id": str(tool_call.get("id") or "").strip(),
                "tool_name": _tool_call_name(tool_call),
                "arguments": arguments if isinstance(arguments, dict) else {},
                "result": result,
            }
        )
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": str(tool_call.get("id") or "").strip(),
                "content": json.dumps(result, indent=2),
            }
        )
    return tool_results, tool_messages


def _measure_selection_duration(
    *,
    arguments: dict[str, Any],
    transcript: Transcript,
    target_seconds: int,
    duration_tolerance_ratio: float,
) -> dict[str, Any]:
    """Measure exact local durations for the model's current candidate spans."""
    raw_samples = arguments.get("samples")
    samples = raw_samples if isinstance(raw_samples, list) else []
    if not samples:
        return {
            "status": "invalid_request",
            "message": "measure_selection_duration requires a non-empty `samples` array.",
            "requested_sample_count": 0,
            "valid_sample_count": 0,
            "invalid_sample_count": 0,
            "total_duration_ms": 0,
            "total_duration_seconds": 0.0,
            "total_covered_source_segments": 0,
            "samples": [],
            "errors": [],
            "budget": _compute_duration_budget_stats(
                target_seconds=target_seconds,
                total_duration_ms=0,
                duration_tolerance_ratio=duration_tolerance_ratio,
            ),
        }

    measured_samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_duration_ms = 0
    total_covered_source_segments = 0
    for index, raw_sample in enumerate(samples, start=1):
        sample_payload = raw_sample if isinstance(raw_sample, dict) else {}
        try:
            source_start_index = int(sample_payload.get("source_start_index"))
        except (TypeError, ValueError):
            source_start_index = -1
        try:
            source_end_index = int(sample_payload.get("source_end_index"))
        except (TypeError, ValueError):
            source_end_index = -1
        order_raw = sample_payload.get("order")
        order = int(order_raw) if isinstance(order_raw, int) else index
        measured = _measure_source_span(
            transcript=transcript,
            source_start_index=source_start_index,
            source_end_index=source_end_index,
        )
        if measured is None:
            errors.append(
                {
                    "order": order,
                    "source_start_index": source_start_index,
                    "source_end_index": source_end_index,
                    "message": (
                        "Source span could not be measured locally. Indices must "
                        "be non-negative, in bounds, and ordered start <= end."
                    ),
                }
            )
            continue
        measured_samples.append(
            {
                "order": order,
                "source_start_index": source_start_index,
                "source_end_index": source_end_index,
                "start_time_ms": measured["start_time_ms"],
                "end_time_ms": measured["end_time_ms"],
                "start_timestamp": measured["start_timestamp"],
                "end_timestamp": measured["end_timestamp"],
                "duration_ms": measured["duration_ms"],
                "duration_seconds": round(measured["duration_ms"] / 1000.0, 3),
                "source_segment_count": measured["source_segment_count"],
            }
        )
        total_duration_ms += int(measured["duration_ms"])
        total_covered_source_segments += int(measured["source_segment_count"])

    return {
        "status": "ok",
        "requested_sample_count": len(samples),
        "valid_sample_count": len(measured_samples),
        "invalid_sample_count": len(errors),
        "total_duration_ms": total_duration_ms,
        "total_duration_seconds": round(total_duration_ms / 1000.0, 3),
        "total_covered_source_segments": total_covered_source_segments,
        "samples": measured_samples,
        "errors": errors,
        "budget": _compute_duration_budget_stats(
            target_seconds=target_seconds,
            total_duration_ms=total_duration_ms,
            duration_tolerance_ratio=duration_tolerance_ratio,
        ),
    }


def _compute_duration_budget_stats(
    *,
    target_seconds: int,
    total_duration_ms: int,
    duration_tolerance_ratio: float,
) -> dict[str, Any]:
    """Mirror the main pipeline's duration-budget telemetry for sampler spans."""
    total_duration_seconds = round(total_duration_ms / 1000.0, 3)
    tolerance_seconds = round(target_seconds * duration_tolerance_ratio, 3)
    min_seconds = round(target_seconds - tolerance_seconds, 3)
    max_seconds = round(target_seconds + tolerance_seconds, 3)
    delta_to_target_seconds = round(total_duration_seconds - target_seconds, 3)
    remaining_seconds_to_target = round(target_seconds - total_duration_seconds, 3)
    seconds_under_target = round(max(target_seconds - total_duration_seconds, 0.0), 3)
    seconds_over_target = round(max(total_duration_seconds - target_seconds, 0.0), 3)
    in_budget = min_seconds <= total_duration_seconds <= max_seconds
    if total_duration_seconds > max_seconds:
        recommended_action = "remove"
    elif total_duration_seconds < min_seconds:
        recommended_action = "add"
    else:
        recommended_action = "submit"
    return {
        "target_seconds": target_seconds,
        "duration_tolerance_ratio": duration_tolerance_ratio,
        "tolerance_seconds": tolerance_seconds,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "total_duration_seconds": total_duration_seconds,
        "delta_to_target_seconds": delta_to_target_seconds,
        "remaining_seconds_to_target": remaining_seconds_to_target,
        "seconds_under_target": seconds_under_target,
        "seconds_over_target": seconds_over_target,
        "in_budget": in_budget,
        "recommended_action": recommended_action,
    }


def _measure_source_span(
    *,
    transcript: Transcript,
    source_start_index: int,
    source_end_index: int,
) -> dict[str, int | str] | None:
    """Measure one contiguous source span against the loaded transcript."""
    if source_start_index < 0 or source_end_index < 0:
        return None
    if source_end_index < source_start_index:
        return None
    if source_end_index >= len(transcript.segments):
        return None
    start_segment = transcript.segments[source_start_index]
    end_segment = transcript.segments[source_end_index]
    start_time_ms = int(start_segment.start_time)
    end_time_ms = int(end_segment.end_time)
    return {
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
        "duration_ms": max(0, end_time_ms - start_time_ms),
        "start_timestamp": _format_timestamp(start_time_ms),
        "end_timestamp": _format_timestamp(end_time_ms),
        "source_segment_count": (source_end_index - source_start_index + 1),
    }


def _build_system_prompt(
    *,
    sampling_mode: str,
    target_seconds: int,
    target_words: int | None,
    requested_samples: int | None,
) -> str:
    """Build the fixed system prompt for transcript sampling."""
    if sampling_mode == "fixed_samples":
        assert requested_samples is not None
        goal_text = (
            f"select exactly {requested_samples} transcript-grounded reconstructed "
            "podcast snippets"
        )
        budget_text = "Return exactly the requested number of samples."
    else:
        goal_text = (
            "select transcript-grounded reconstructed podcast snippets that a "
            "researcher or downstream system should rely on for a duration-constrained "
            f"review sample of about {target_seconds} seconds"
        )
        budget_text = (
            "Use the measured source-duration budget as the primary stopping rule. "
            "Keep selecting or trimming spans until the measured total is in budget."
        )
    return (
        "You are an expert transcript sampler for summary research. "
        f"Your job is to {goal_text}. "
        "You are not writing the summary itself. "
        "Your output should give a human reviewer an accurate depiction of the "
        "episode without requiring them to listen to the full source audio. "
        "You are doing grounded reconstruction from a noisy transcript, not polished rewriting. "
        "You have access to a local timing tool named `measure_selection_duration`. "
        f"The per-episode human-evaluation sample budget is approximately {target_seconds} seconds. "
        "Whenever you need exact duration math for the spans you have already picked, "
        "call `measure_selection_duration` instead of estimating or doing arithmetic yourself. "
        "That tool returns the measured running total, the remaining seconds to target, "
        "whether the current selection is in budget, and whether you should add or remove material. "
        "After you are satisfied with the running total, finish with one final "
        "`submit_samples` tool call. "
        "Do not narrate your reasoning, draft alternatives, explain validation issues, "
        "or ask for another pass. "
        "Keep every sample grounded in one inclusive contiguous source span from the transcript. "
        "Preserve what the speaker said, including disfluencies, repetitions, and filler words, "
        "unless a repeated token is obviously transcript noise caused by fragmentation or overlap. "
        "You may repair only local transcript noise such as broken spacing, chopped words, "
        "fragmentation across adjacent lines, punctuation damage, or obvious diarization seams "
        "inside the cited source span so the snippet is understandable to a human reviewer. "
        "Do not silently compress, summarize, or add connective language that is not supported by "
        "the cited source span. If uncertain, cite a slightly wider source span instead of "
        "inventing smoother wording. "
        "Include `source_start_index`, `source_end_index`, `source_start_timestamp`, "
        "`source_end_timestamp`, `speaker_mode`, and `speaker_labels` as best-effort "
        "metadata for the snippet you selected. "
        "Prefer snippets that capture the episode arc, major decisions, conflicts, reveals, "
        "resolutions, and distinctive speaker phrasing while still remaining source-grounded. "
        f"{budget_text}"
    )


def _build_user_prompt(
    *,
    sample: PreparedBenchmarkSample,
    transcript_text: str,
    sampling_mode: str,
    target_seconds: int,
    target_words: int | None,
    requested_samples: int | None,
    transcript_word_count: int,
    source_description: str = "",
) -> str:
    """Build the per-episode user prompt."""
    episode_title = (
        str(sample.metadata.get("episode_title", "") or "").strip() or sample.sample_id
    )
    target_line = (
        f"Requested samples: {requested_samples}"
        if sampling_mode == "fixed_samples"
        else (
            "Primary stopping rule: finish when the measured selected source "
            f"duration is in budget around {target_seconds} seconds."
        )
    )
    return "\n".join(
        [
            f"Episode key: {sample.episode_key}",
            f"Episode title: {episode_title}",
            f"Transcript words: {transcript_word_count}",
            (
                "Target sampled source duration for human evaluation: "
                f"about {target_seconds} seconds"
            ),
            target_line,
            source_description,
            (
                "Select source-grounded spans that together give a human reviewer "
                "an accurate depiction of what matters in the episode without "
                "listening to the full podcast."
            ),
            (
                "Each transcript line is indexed. Return source indices and timestamps "
                "for every sample when you can. Stay grounded in one inclusive "
                "contiguous source span per sample. Preserve what the speakers said, "
                "including disfluencies, repetitions, and filler words. You may "
                "repair only local transcript noise such as broken spacing, chopped "
                "words, punctuation damage, adjacent-line fragmentation, and obvious "
                "diarization seams inside the cited span so the snippet is "
                "understandable to a human reviewer. Do not rewrite the exchange into "
                "cleaner podcast dialogue than the source supports. Use "
                "`measure_selection_duration` with the samples you have already chosen "
                "whenever you need exact total duration or exact per-sample timing. "
                "Read the returned `budget` object, especially `remaining_seconds_to_target`, "
                "`in_budget`, and `recommended_action`, before deciding whether to add, "
                "remove, or submit. Do not do duration arithmetic yourself. When the payload is ready, "
                "call `submit_samples` with no extra prose."
            ),
            (
                "When a transcript line includes `[overlap: ...]`, that means another "
                "speaker was active during the same time span. Treat that bracket as "
                "source metadata, not rewritten dialogue."
            ),
            "",
            "Transcript:",
            transcript_text,
        ]
    )


def _measure_selection_duration_tool_schema() -> dict[str, Any]:
    """Return the local duration-accounting tool schema for candidate samples."""
    return {
        "type": "function",
        "function": {
            "name": _MEASURE_SELECTION_DURATION_TOOL_NAME,
            "description": (
                "Measure the exact local duration of the candidate transcript spans "
                "you have already chosen so far."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "samples": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "order": {"type": "integer"},
                                "source_start_index": {"type": "integer"},
                                "source_end_index": {"type": "integer"},
                            },
                            "required": [
                                "source_start_index",
                                "source_end_index",
                            ],
                        },
                    }
                },
                "required": ["samples"],
            },
        },
    }


def _submit_samples_tool_schema() -> dict[str, Any]:
    """Return the strict tool schema used for structured sample submission."""
    return {
        "type": "function",
        "function": {
            "name": _SUBMIT_SAMPLES_TOOL_NAME,
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
                                "speaker_mode": {
                                    "type": "string",
                                    "enum": sorted(_VALID_SPEAKER_MODES),
                                },
                                "speaker_labels": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string"},
                                },
                                "source_start_index": {"type": "integer"},
                                "source_end_index": {"type": "integer"},
                                "source_start_timestamp": {"type": "string"},
                                "source_end_timestamp": {"type": "string"},
                                "segment_text": {"type": "string"},
                                "rationale": {"type": "string"},
                                "estimated_words": {"type": "integer"},
                            },
                            "required": [
                                "order",
                                "speaker_mode",
                                "speaker_labels",
                                "source_start_index",
                                "source_end_index",
                                "source_start_timestamp",
                                "source_end_timestamp",
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
    transcript: Transcript,
    episode_key: str,
    sampling_mode: str,
    target_seconds: int,
    duration_tolerance_ratio: float,
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
            "actual_selected_duration_ms": 0,
        }

    submit_call: dict[str, Any] | None = None
    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_payload = tool_call.get("function", {})
        function = function_payload if isinstance(function_payload, dict) else {}
        if str(function.get("name") or "") == _SUBMIT_SAMPLES_TOOL_NAME:
            submit_call = function
            break
    if submit_call is None:
        return [], {
            "status": "failed_validation",
            "message": "Model did not call the submit_samples tool.",
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }

    try:
        payload = json.loads(str(submit_call.get("arguments") or "{}"))
    except json.JSONDecodeError:
        return [], {
            "status": "failed_validation",
            "message": "submit_samples arguments were not valid JSON.",
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }
    if not isinstance(payload, dict):
        return [], {
            "status": "failed_validation",
            "message": "submit_samples arguments must be a JSON object.",
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        return [], {
            "status": "failed_validation",
            "message": "submit_samples returned an empty samples list.",
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }

    parsed_samples: list[_EpisodeSelection] = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            return [], {
                "status": "failed_validation",
                "message": "Each sample must be an object.",
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
        segment_text = str(raw_sample.get("segment_text") or "").strip()
        rationale = str(raw_sample.get("rationale") or "").strip()
        speaker_mode = str(raw_sample.get("speaker_mode") or "").strip()
        raw_speaker_labels = raw_sample.get("speaker_labels")
        speaker_labels = tuple(
            str(item or "").strip()
            for item in (raw_speaker_labels if isinstance(raw_speaker_labels, list) else [])
            if str(item or "").strip()
        )
        try:
            source_start_index = int(raw_sample.get("source_start_index"))
        except (TypeError, ValueError):
            source_start_index = -1
        try:
            source_end_index = int(raw_sample.get("source_end_index"))
        except (TypeError, ValueError):
            source_end_index = -1
        source_start_timestamp = str(raw_sample.get("source_start_timestamp") or "").strip()
        source_end_timestamp = str(raw_sample.get("source_end_timestamp") or "").strip()
        if not segment_text:
            return [], {
                "status": "failed_validation",
                "message": "Each sample requires non-empty segment_text.",
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
        if not rationale:
            rationale = "Selected by model."
        if not speaker_mode:
            speaker_mode = "single_speaker_repair"
        estimated_words = int(raw_sample.get("estimated_words") or 0)
        actual_words = count_words(segment_text)
        if actual_words <= 0:
            return [], {
                "status": "failed_validation",
                "message": "Each sample requires non-empty segment_text.",
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
        if estimated_words <= 0:
            estimated_words = actual_words
        measured = _measure_source_span(
            transcript=transcript,
            source_start_index=source_start_index,
            source_end_index=source_end_index,
        )
        parsed_samples.append(
            _EpisodeSelection(
                order=int(raw_sample.get("order") or len(parsed_samples) + 1),
                speaker_mode=speaker_mode,
                speaker_labels=speaker_labels,
                speaker_label=(
                    speaker_labels[0]
                    if speaker_labels
                    else None
                ),
                source_start_index=source_start_index,
                source_end_index=source_end_index,
                source_start_timestamp=source_start_timestamp,
                source_end_timestamp=source_end_timestamp,
                segment_text=segment_text,
                rationale=rationale,
                estimated_words=estimated_words,
                actual_words=actual_words,
                start_time_ms=(
                    int(measured["start_time_ms"]) if measured is not None else None
                ),
                end_time_ms=(
                    int(measured["end_time_ms"]) if measured is not None else None
                ),
                duration_ms=(
                    int(measured["duration_ms"]) if measured is not None else None
                ),
                source_segment_count=(
                    int(measured["source_segment_count"])
                    if measured is not None
                    else None
                ),
            )
        )

    parsed_samples.sort(key=lambda item: item.order)
    actual_selected_words = sum(item.actual_words for item in parsed_samples)
    actual_selected_duration_ms = sum(
        int(item.duration_ms or 0) for item in parsed_samples
    )
    if requested_samples is not None and len(parsed_samples) != requested_samples:
        return parsed_samples, {
            "status": "failed_validation",
            "message": (
                "Model returned the wrong number of samples. "
                f"Requested={requested_samples}, actual={len(parsed_samples)}."
            ),
            "actual_selected_words": actual_selected_words,
            "actual_selected_duration_ms": actual_selected_duration_ms,
        }

    if sampling_mode == "fixed_samples":
        return parsed_samples, {
            "status": "completed",
            "message": "Submission accepted.",
            "actual_selected_words": actual_selected_words,
            "actual_selected_duration_ms": actual_selected_duration_ms,
        }

    duration_budget = _compute_duration_budget_stats(
        target_seconds=target_seconds,
        total_duration_ms=actual_selected_duration_ms,
        duration_tolerance_ratio=duration_tolerance_ratio,
    )
    if not bool(duration_budget["in_budget"]):
        return parsed_samples, {
            "status": "failed_validation",
            "message": (
                "Selected source duration was outside the allowed tolerance band. "
                f"Target={target_seconds}s, actual={duration_budget['total_duration_seconds']}s, "
                f"allowed_range={duration_budget['min_seconds']}-{duration_budget['max_seconds']}s, "
                f"remaining_to_target={duration_budget['remaining_seconds_to_target']}s."
            ),
            "actual_selected_words": actual_selected_words,
            "actual_selected_duration_ms": actual_selected_duration_ms,
        }

    return parsed_samples, {
        "status": "completed",
        "message": "Submission accepted.",
        "actual_selected_words": actual_selected_words,
        "actual_selected_duration_ms": actual_selected_duration_ms,
    }


def _validate_last_measured_selection(
    *,
    arguments: dict[str, Any],
    transcript: Transcript,
    sampling_mode: str,
    target_seconds: int,
    duration_tolerance_ratio: float,
    requested_samples: int | None,
) -> tuple[list[_EpisodeSelection], dict[str, Any]]:
    """Accept the last measured span set when it already satisfies local checks."""
    raw_samples = arguments.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        return [], {
            "status": "failed_validation",
            "message": "Last measured selection did not contain any samples.",
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }

    parsed_samples: list[_EpisodeSelection] = []
    for index, raw_sample in enumerate(raw_samples, start=1):
        if not isinstance(raw_sample, dict):
            return [], {
                "status": "failed_validation",
                "message": "Each measured sample must be an object.",
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
        try:
            source_start_index = int(raw_sample.get("source_start_index"))
            source_end_index = int(raw_sample.get("source_end_index"))
        except (TypeError, ValueError):
            return [], {
                "status": "failed_validation",
                "message": (
                    "Each measured sample requires integer source indices."
                ),
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
        materialized = _materialize_transcript_span(
            transcript=transcript,
            source_start_index=source_start_index,
            source_end_index=source_end_index,
        )
        if materialized is None:
            return [], {
                "status": "failed_validation",
                "message": (
                    "Each measured sample must cite an in-bounds inclusive "
                    "transcript span with start <= end."
                ),
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }

        speaker_mode = str(
            raw_sample.get("speaker_mode") or materialized["speaker_mode"]
        ).strip() or str(materialized["speaker_mode"])
        raw_speaker_labels = raw_sample.get("speaker_labels")
        provided_speaker_labels = (
            tuple(
                str(item or "").strip()
                for item in raw_speaker_labels
                if str(item or "").strip()
            )
            if isinstance(raw_speaker_labels, list)
            else ()
        )
        speaker_labels = (
            provided_speaker_labels
            if provided_speaker_labels
            else tuple(materialized["speaker_labels"])
        )
        speaker_label = (
            " + ".join(speaker_labels)
            if len(speaker_labels) > 1
            else (speaker_labels[0] if speaker_labels else None)
        )
        rationale = str(raw_sample.get("rationale") or "").strip() or (
            "Accepted from the final locally measured selection."
        )
        parsed_samples.append(
            _EpisodeSelection(
                order=int(raw_sample.get("order") or index),
                segment_text=str(materialized["segment_text"]),
                rationale=rationale,
                estimated_words=int(materialized["actual_words"]),
                actual_words=int(materialized["actual_words"]),
                speaker_mode=speaker_mode,
                speaker_labels=speaker_labels,
                speaker_label=speaker_label,
                source_start_index=source_start_index,
                source_end_index=source_end_index,
                source_start_timestamp=str(
                    materialized["source_start_timestamp"]
                ),
                source_end_timestamp=str(materialized["source_end_timestamp"]),
                start_time_ms=int(materialized["start_time_ms"]),
                end_time_ms=int(materialized["end_time_ms"]),
                duration_ms=int(materialized["duration_ms"]),
                source_segment_count=int(materialized["source_segment_count"]),
            )
        )

    parsed_samples.sort(key=lambda item: item.order)
    actual_selected_words = sum(item.actual_words for item in parsed_samples)
    actual_selected_duration_ms = sum(
        int(item.duration_ms or 0) for item in parsed_samples
    )
    if requested_samples is not None and len(parsed_samples) != requested_samples:
        return parsed_samples, {
            "status": "failed_validation",
            "message": (
                "Last measured selection returned the wrong number of samples. "
                f"Requested={requested_samples}, actual={len(parsed_samples)}."
            ),
            "actual_selected_words": actual_selected_words,
            "actual_selected_duration_ms": actual_selected_duration_ms,
        }

    if sampling_mode == "fixed_samples":
        return parsed_samples, {
            "status": "completed",
            "message": "Accepted last measured selection.",
            "actual_selected_words": actual_selected_words,
            "actual_selected_duration_ms": actual_selected_duration_ms,
        }

    duration_budget = _compute_duration_budget_stats(
        target_seconds=target_seconds,
        total_duration_ms=actual_selected_duration_ms,
        duration_tolerance_ratio=duration_tolerance_ratio,
    )
    if not bool(duration_budget["in_budget"]):
        return parsed_samples, {
            "status": "failed_validation",
            "message": (
                "Last measured source duration was outside the allowed "
                "tolerance band. "
                f"Target={target_seconds}s, "
                f"actual={duration_budget['total_duration_seconds']}s, "
                f"allowed_range={duration_budget['min_seconds']}-"
                f"{duration_budget['max_seconds']}s."
            ),
            "actual_selected_words": actual_selected_words,
            "actual_selected_duration_ms": actual_selected_duration_ms,
        }

    return parsed_samples, {
        "status": "completed",
        "message": "Accepted last measured selection.",
        "actual_selected_words": actual_selected_words,
        "actual_selected_duration_ms": actual_selected_duration_ms,
    }


def _is_repairable_single_speaker_seam(
    *,
    span_segments: list[Any] | tuple[Any, ...],
    target_speaker: str,
) -> bool:
    """Allow bounded cross-speaker seam repair for obvious diarization noise."""
    text_segments = [
        segment
        for segment in span_segments
        if str(getattr(segment, "text", "") or "").strip()
    ]
    if not text_segments:
        return False
    distinct_speakers = tuple(
        dict.fromkeys(str(getattr(segment, "speaker", "") or "").strip() for segment in text_segments)
    )
    if len(distinct_speakers) <= 1:
        return True
    if target_speaker not in distinct_speakers:
        return False

    total_words = 0
    off_speaker_words = 0
    for segment in text_segments:
        text = str(getattr(segment, "text", "") or "").strip()
        speaker = str(getattr(segment, "speaker", "") or "").strip()
        word_count = count_words(text)
        total_words += word_count
        if speaker != target_speaker:
            off_speaker_words += word_count
    if total_words <= 0:
        return False
    if off_speaker_words <= 0:
        return True
    if off_speaker_words > _SEAM_MAX_OFF_SPEAKER_WORDS:
        return False
    if (off_speaker_words / total_words) > _SEAM_MAX_OFF_SPEAKER_RATIO:
        return False

    for index, segment in enumerate(text_segments):
        speaker = str(getattr(segment, "speaker", "") or "").strip()
        if speaker == target_speaker:
            continue
        text = str(getattr(segment, "text", "") or "").strip()
        if count_words(text) > _SEAM_MAX_OFF_SPEAKER_WORDS:
            return False
        if index == 0 or index == len(text_segments) - 1:
            return False
        previous_speaker = str(getattr(text_segments[index - 1], "speaker", "") or "").strip()
        next_speaker = str(getattr(text_segments[index + 1], "speaker", "") or "").strip()
        if previous_speaker != target_speaker or next_speaker != target_speaker:
            return False
    return True


def _classify_episode_error(message: str) -> str:
    """Classify one provider error into a stable episode status."""
    normalized = message.lower()
    if "context" in normalized or "too long" in normalized or "too many tokens" in normalized:
        return "failed_context_limit"
    return "failed_api"


def _build_selected_rows(
    *,
    run_id: str,
    text_only: bool,
    transcribe_enabled: bool,
    source_mode: str,
    duration_preset: str,
    target_seconds: int | None,
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
            "text_only": text_only,
            "transcribe_enabled": transcribe_enabled,
            "source_mode": source_mode,
            "duration_preset": duration_preset,
            "target_seconds": target_seconds,
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
            "speaker_mode": selection.speaker_mode,
            "speaker_labels": list(selection.speaker_labels),
            "source_start_index": selection.source_start_index,
            "source_end_index": selection.source_end_index,
            "source_start_timestamp": selection.source_start_timestamp,
            "source_end_timestamp": selection.source_end_timestamp,
            "start_time": selection.start_time_ms,
            "end_time": selection.end_time_ms,
            "duration_ms": selection.duration_ms,
            "source_segment_count": selection.source_segment_count,
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
        speaker_label = (
            selection.speaker_label
            if selection.speaker_label is not None
            else "UNKNOWN"
        )
        lines.extend(
            [
                f"{selection.order}. [{speaker_label}] {selection.segment_text}",
                (
                    "Source: "
                    f"{selection.source_start_index}-{selection.source_end_index} "
                    f"({selection.source_start_timestamp}-{selection.source_end_timestamp}) "
                    f"| {selection.speaker_mode}"
                    + (
                        f" | measured {selection.duration_ms / 1000.0:.3f}s"
                        if selection.duration_ms is not None
                        else ""
                    )
                ),
                f"Rationale: {selection.rationale}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_combined_segments_text(
    *,
    output_path: Path,
    parsed_samples: list[_EpisodeSelection],
) -> None:
    """Write one text-only combined transcript artifact from accepted selections."""
    lines: list[str] = []
    for index, selection in enumerate(parsed_samples, start=1):
        if index > 1:
            lines.append("")
        lines.append(selection.segment_text)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _render_raw_turns_for_prompt(raw_turns: tuple[_RawTranscriptTurn, ...]) -> str:
    """Render source speaker turns into one prompt-friendly transcript block."""
    lines = [
        (
            f"[#{turn.index} {_format_timestamp(turn.start_time_ms)}-{_format_timestamp(turn.end_time_ms)}] "
            f"[{turn.speaker_label}]"
            f"{_render_overlap_suffix(turn.overlap_speakers)} {turn.text}"
        )
        for turn in raw_turns
        if turn.text.strip()
    ]
    if not lines:
        raise RuntimeError("Source transcript contains no usable speaker-turn text.")
    return "\n".join(lines)


def _submit_text_only_segments_tool_schema() -> dict[str, Any]:
    """Return the strict submit schema for raw text-only span selection."""
    return {
        "type": "function",
        "function": {
            "name": _SUBMIT_SAMPLES_TOOL_NAME,
            "description": "Submit exact raw-transcript spans for the text-only sampling mode.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "episode_key": {"type": "string"},
                    "requested_samples": {"type": "integer"},
                    "samples": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "order": {"type": "integer"},
                                "source_start_index": {"type": "integer"},
                                "source_end_index": {"type": "integer"},
                                "rationale": {"type": "string"},
                            },
                            "required": [
                                "order",
                                "source_start_index",
                                "source_end_index",
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


def _build_text_only_system_prompt(*, requested_samples: int, source_mode: str) -> str:
    """Build the fixed system prompt for text-only span selection."""
    source_description = (
        "the stage-1 diarized transcript"
        if source_mode == _TRANSCRIBED_SOURCE_MODE
        else "the raw speaker-turn data"
    )
    return (
        "You are an expert transcript sampler for SPoRC episode review. "
        f"Select exactly {requested_samples} transcript-grounded segments from {source_description}. "
        "The source may be fragmented, so one selected segment may cite a contiguous span of source indices "
        "in order to reconstruct one coherent speaker line from adjacent fragments. "
        "Prefer segments that give a human reviewer an accurate depiction of the episode without listening "
        "to the full podcast. "
        "Do not rewrite, summarize, or polish the transcript. "
        "Do not invent timestamps, speaker labels, or text. "
        "Return only exact inclusive source spans from the provided transcript, then finish with one "
        "`submit_samples` tool call."
    )


def _build_text_only_user_prompt(
    *,
    sample: PreparedBenchmarkSample,
    raw_turns: tuple[_RawTranscriptTurn, ...],
    requested_samples: int,
    source_label: str,
    source_guidance: str,
) -> str:
    """Build the per-episode user prompt for text-only span selection."""
    episode_title = (
        str(sample.metadata.get("episode_title", "") or "").strip() or sample.sample_id
    )
    return "\n".join(
        [
            f"Episode key: {sample.episode_key}",
            f"Episode title: {episode_title}",
            f"{source_label} line count: {len(raw_turns)}",
            f"Requested segments: {requested_samples}",
            source_guidance,
            (
                "When a line includes `[overlap: ...]`, that means another speaker "
                "was active during the same time span. Keep the cited source span "
                "grounded in the provided line boundaries and overlap metadata."
            ),
            (
                "Return exact `source_start_index` and `source_end_index` pairs only. "
                "Each selected segment must map to one inclusive contiguous span from the provided transcript. "
                "Short rationale text is allowed, but do not include reconstructed transcript text in the tool payload."
            ),
            "",
            f"{source_label}:",
            _render_raw_turns_for_prompt(raw_turns),
        ]
    )


def _validate_text_only_submission(
    *,
    assistant_message: dict[str, Any],
    raw_turns: tuple[_RawTranscriptTurn, ...],
    episode_key: str,
    requested_samples: int,
) -> tuple[list[_EpisodeSelection], dict[str, Any]]:
    """Validate and locally materialize a raw text-only span selection."""
    raw_tool_calls = assistant_message.get("tool_calls")
    if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
        return [], {
            "status": "failed_validation",
            "message": "Model did not call submit_samples.",
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }

    submit_call: dict[str, Any] | None = None
    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_payload = tool_call.get("function", {})
        function = function_payload if isinstance(function_payload, dict) else {}
        if str(function.get("name") or "") == _SUBMIT_SAMPLES_TOOL_NAME:
            submit_call = function
            break
    if submit_call is None:
        return [], {
            "status": "failed_validation",
            "message": "Model did not call the submit_samples tool.",
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }

    try:
        payload = json.loads(str(submit_call.get("arguments") or "{}"))
    except json.JSONDecodeError:
        return [], {
            "status": "failed_validation",
            "message": "submit_samples arguments were not valid JSON.",
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }
    if not isinstance(payload, dict):
        return [], {
            "status": "failed_validation",
            "message": "submit_samples arguments must be a JSON object.",
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }
    payload_episode_key = str(payload.get("episode_key") or "").strip()
    if payload_episode_key and payload_episode_key != episode_key:
        return [], {
            "status": "failed_validation",
            "message": (
                "submit_samples returned the wrong episode_key. "
                f"Expected={episode_key}, actual={payload_episode_key}."
            ),
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        return [], {
            "status": "failed_validation",
            "message": "submit_samples returned an empty samples list.",
            "actual_selected_words": 0,
            "actual_selected_duration_ms": 0,
        }

    parsed_samples: list[_EpisodeSelection] = []
    for index, raw_sample in enumerate(raw_samples, start=1):
        if not isinstance(raw_sample, dict):
            return [], {
                "status": "failed_validation",
                "message": "Each sample must be an object.",
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
        try:
            source_start_index = int(raw_sample.get("source_start_index"))
            source_end_index = int(raw_sample.get("source_end_index"))
        except (TypeError, ValueError):
            return [], {
                "status": "failed_validation",
                "message": "Each text-only sample requires integer source indices.",
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
        materialized = _materialize_raw_turn_span(
            raw_turns=raw_turns,
            source_start_index=source_start_index,
            source_end_index=source_end_index,
        )
        if materialized is None:
            return [], {
                "status": "failed_validation",
                "message": (
                    "Each text-only sample must cite an in-bounds inclusive raw transcript span "
                    "with start <= end."
                ),
                "actual_selected_words": 0,
                "actual_selected_duration_ms": 0,
            }
        rationale = str(raw_sample.get("rationale") or "").strip() or "Selected by model."
        parsed_samples.append(
            _EpisodeSelection(
                order=int(raw_sample.get("order") or index),
                segment_text=materialized["segment_text"],
                rationale=rationale,
                estimated_words=int(materialized["actual_words"]),
                actual_words=int(materialized["actual_words"]),
                speaker_mode=_RAW_TRANSCRIPT_SPEAKER_MODE,
                speaker_labels=tuple(materialized["speaker_labels"]),
                speaker_label=materialized["speaker_label"],
                source_start_index=source_start_index,
                source_end_index=source_end_index,
                source_start_timestamp=str(materialized["source_start_timestamp"]),
                source_end_timestamp=str(materialized["source_end_timestamp"]),
                start_time_ms=int(materialized["start_time_ms"]),
                end_time_ms=int(materialized["end_time_ms"]),
                duration_ms=int(materialized["duration_ms"]),
                source_segment_count=int(materialized["source_segment_count"]),
            )
        )

    parsed_samples.sort(key=lambda item: item.order)
    actual_selected_words = sum(item.actual_words for item in parsed_samples)
    actual_selected_duration_ms = sum(int(item.duration_ms or 0) for item in parsed_samples)
    if len(parsed_samples) != requested_samples:
        return parsed_samples, {
            "status": "failed_validation",
            "message": (
                "Model returned the wrong number of text-only samples. "
                f"Requested={requested_samples}, actual={len(parsed_samples)}."
            ),
            "actual_selected_words": actual_selected_words,
            "actual_selected_duration_ms": actual_selected_duration_ms,
        }
    return parsed_samples, {
        "status": "completed",
        "message": "Submission accepted.",
        "actual_selected_words": actual_selected_words,
        "actual_selected_duration_ms": actual_selected_duration_ms,
    }


def _materialize_raw_turn_span(
    *,
    raw_turns: tuple[_RawTranscriptTurn, ...],
    source_start_index: int,
    source_end_index: int,
) -> dict[str, Any] | None:
    """Convert one raw-turn index span into a locally derived selected segment."""
    if source_start_index < 0 or source_end_index < source_start_index:
        return None
    if source_end_index >= len(raw_turns):
        return None

    span_turns = raw_turns[source_start_index : source_end_index + 1]
    if not span_turns:
        return None
    merged_lines: list[tuple[str, str]] = []
    for turn in span_turns:
        if merged_lines and merged_lines[-1][0] == turn.speaker_label:
            prior_speaker, prior_text = merged_lines[-1]
            merged_lines[-1] = (prior_speaker, f"{prior_text} {turn.text}".strip())
        else:
            merged_lines.append((turn.speaker_label, turn.text))

    speaker_labels = tuple(
        dict.fromkeys(
            speaker_label
            for speaker_label, _ in merged_lines
            if speaker_label.strip()
        )
    )
    rendered_lines = [
        f"[{speaker_label}] {text}".strip()
        for speaker_label, text in merged_lines
        if text.strip()
    ]
    first_turn = span_turns[0]
    last_turn = span_turns[-1]
    actual_words = sum(count_words(turn.text) for turn in span_turns)
    return {
        "segment_text": "\n".join(rendered_lines),
        "speaker_labels": speaker_labels,
        "speaker_label": (" + ".join(speaker_labels) if len(speaker_labels) > 1 else (speaker_labels[0] if speaker_labels else None)),
        "source_start_timestamp": _format_timestamp(first_turn.start_time_ms),
        "source_end_timestamp": _format_timestamp(last_turn.end_time_ms),
        "start_time_ms": first_turn.start_time_ms,
        "end_time_ms": last_turn.end_time_ms,
        "duration_ms": max(0, last_turn.end_time_ms - first_turn.start_time_ms),
        "source_segment_count": len(span_turns),
        "actual_words": actual_words,
    }


def _materialize_transcript_span(
    *,
    transcript: Transcript,
    source_start_index: int,
    source_end_index: int,
) -> dict[str, Any] | None:
    """Convert one transcript-segment span into a locally derived selection."""
    if source_start_index < 0 or source_end_index < source_start_index:
        return None
    if source_end_index >= len(transcript.segments):
        return None

    span_segments = transcript.segments[source_start_index : source_end_index + 1]
    if not span_segments:
        return None

    merged_lines: list[tuple[str, str]] = []
    speaker_labels = tuple(
        dict.fromkeys(
            (str(segment.speaker or "").strip() or "UNKNOWN")
            for segment in span_segments
        )
    )
    for segment in span_segments:
        speaker_label = str(segment.speaker or "").strip() or "UNKNOWN"
        text = str(segment.text or "").strip()
        if not text:
            continue
        if merged_lines and merged_lines[-1][0] == speaker_label:
            prior_speaker, prior_text = merged_lines[-1]
            merged_lines[-1] = (
                prior_speaker,
                f"{prior_text} {text}".strip(),
            )
        else:
            merged_lines.append((speaker_label, text))

    first_segment = span_segments[0]
    last_segment = span_segments[-1]
    actual_words = sum(
        count_words(str(segment.text or ""))
        for segment in span_segments
    )
    rendered_lines = [
        f"[{speaker_label}] {text}".strip()
        for speaker_label, text in merged_lines
        if text.strip()
    ]
    speaker_mode = (
        "single_speaker_repair"
        if len(speaker_labels) <= 1
        else "multi_speaker_exchange"
    )
    return {
        "segment_text": "\n".join(rendered_lines),
        "speaker_mode": speaker_mode,
        "speaker_labels": speaker_labels,
        "speaker_label": (
            " + ".join(speaker_labels)
            if len(speaker_labels) > 1
            else (speaker_labels[0] if speaker_labels else None)
        ),
        "source_start_timestamp": _format_timestamp(first_segment.start_time),
        "source_end_timestamp": _format_timestamp(last_segment.end_time),
        "start_time_ms": int(first_segment.start_time),
        "end_time_ms": int(last_segment.end_time),
        "duration_ms": max(
            0,
            int(last_segment.end_time) - int(first_segment.start_time),
        ),
        "source_segment_count": len(span_segments),
        "actual_words": actual_words,
    }


def _render_transcript_for_prompt(transcript: Transcript) -> str:
    """Render a transcript into one prompt-friendly transcript block."""
    lines: list[str] = []
    for index, segment in enumerate(transcript.segments):
        text = str(segment.text or "").strip()
        if not text:
            continue
        overlap_speakers = ()
        if isinstance(segment.extras, dict):
            raw_overlap = segment.extras.get("overlap_speakers", [])
            if isinstance(raw_overlap, list | tuple):
                overlap_speakers = tuple(
                    str(speaker).strip()
                    for speaker in raw_overlap
                    if str(speaker).strip()
                )
        lines.append(
            (
                f"[#{index} {_format_timestamp(segment.start_time)}-{_format_timestamp(segment.end_time)}] "
                f"[{segment.speaker}]"
                f"{_render_overlap_suffix(overlap_speakers)} {text}"
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


def _render_overlap_suffix(overlap_speakers: tuple[str, ...]) -> str:
    """Render one prompt suffix for overlap-aware transcript lines."""
    if not overlap_speakers:
        return ""
    return " [overlap: " + ", ".join(overlap_speakers) + "]"


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
                    f"[bold]Source:[/bold] {_render_source_mode_label(settings)}",
                    f"[bold]Provider:[/bold] {settings['provider']}",
                    f"[bold]Model:[/bold] {settings.get('resolved_model') or settings['model']}",
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
    if settings["sampling_mode"] == "text_only_segments":
        return (
            "[bold]Per-episode target:[/bold] "
            f"{settings['requested_samples']} source transcript segment(s)"
        )
    if settings["sampling_mode"] == "fixed_samples":
        return (
            "[bold]Per-episode target:[/bold] "
            f"{settings['requested_samples']} sample(s) within about "
            f"{settings['preset_seconds']} seconds"
        )
    return (
        "[bold]Per-episode target:[/bold] "
        f"about {settings['preset_seconds']} seconds of measured sampled source duration"
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
    if settings["sampling_mode"] == "text_only_segments":
        return f"{settings['requested_samples']} source segment(s)"
    if settings["sampling_mode"] == "fixed_samples":
        return f"{settings['requested_samples']} sample(s), about {settings['preset_seconds']}s"
    return f"about {settings['preset_seconds']}s measured source duration"


def _render_source_mode_label(settings: dict[str, Any]) -> str:
    """Render a human-readable source mode label for llm-sample banners."""
    if settings["source_mode"] == _TRANSCRIBED_SOURCE_MODE:
        return (
            "stage-1 transcript (source separation + Granite ASR + configured diarization)"
        )
    if settings["source_mode"] == _RAW_SOURCE_MODE:
        return "raw SPoRC speaker-turn transcript"
    return "prepared transcript"


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


def _render_episode_success(
    console: Console,
    *,
    selected_words: int,
    selected_duration_ms: int,
    selection_count: int,
) -> None:
    """Render the accepted selection summary for one episode."""
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Selected words: {selected_words}",
                    f"Measured duration: {selected_duration_ms / 1000.0:.3f}s",
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
