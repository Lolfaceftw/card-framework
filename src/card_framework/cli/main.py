from collections.abc import Callable, Sequence
import asyncio
import json
from itertools import permutations
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, cast

import hydra
import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import InMemoryQueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from omegaconf import DictConfig, OmegaConf

from card_framework.agents.client import AgentClient
from card_framework.agents.critic import CriticExecutor
from card_framework.agents.health import AgentHealthChecker
from card_framework.agents.retrieval import InfoRetrievalExecutor
from card_framework.agents.summarizer import SummarizerExecutor
from card_framework.agents.utils import load_transcript
from card_framework.audio_pipeline import (
    build_audio_to_script_orchestrator,
    build_interjector_orchestrator,
    build_speaker_sample_generator,
    build_voice_clone_orchestrator,
)
from card_framework.audio_pipeline.calibration import (
    DEFAULT_DURATION_TOLERANCE_RATIO,
    bootstrap_speaker_samples_from_audio,
    ensure_voice_clone_calibration,
)
from card_framework.audio_pipeline.contracts import TranscriptPayload
from card_framework.audio_pipeline.eta import (
    DynamicEtaTracker,
    EtaProfilePersistence,
    StageEtaStrategy,
    StageProgressCallback,
    StageProgressUpdate,
    UnitStageEtaLearner,
    UnitStageEtaHistory,
    UnitStageEtaStrategy,
    format_eta_seconds,
)
from card_framework.audio_pipeline.errors import NonRetryableAudioStageError
from card_framework.audio_pipeline.gpu_heartbeat import (
    VoiceCloneGpuHeartbeatService,
    WindowsNvidiaDedicatedGpuProbe,
    parse_voice_clone_gpu_heartbeat_config,
)
from card_framework.audio_pipeline.io import write_transcript_atomic
from card_framework.audio_pipeline.runtime import resolve_device, resolve_path
from card_framework.audio_pipeline.speaker_samples import resolve_sample_source_audio_path
from card_framework.retrieval.embeddings import TranscriptIndex
from card_framework.shared.events import event_bus
from card_framework.shared.llm_provider import EmbeddingProvider, LLMProvider
from card_framework.shared.logger_utils import configure_logger
from card_framework.orchestration import StageOrchestrator
from card_framework.runtime.loop_orchestrator import Orchestrator
from card_framework.runtime.pipeline_plan import build_pipeline_stage_plan
from card_framework.providers.logging_provider import LoggingLLMProvider
from card_framework.providers.response_callbacks import RichConsoleResponseCallback
from card_framework.orchestration.transcript import Transcript
from card_framework.shared.paths import DEFAULT_CONFIG_PATH


_STAGE_TWO_BOOTSTRAP_WINDOW_MS = 30_000
_STAGE_TWO_BOOTSTRAP_MAX_WINDOWS = 12
_STAGE_TWO_BOOTSTRAP_MIN_AVG_WINDOW_SIMILARITY = 0.18
_STAGE_TWO_BOOTSTRAP_MIN_STRONG_WINDOW_RATIO = 0.25
_STAGE_TWO_BOOTSTRAP_STRONG_WINDOW_SIMILARITY = 0.2
_STAGE_TWO_BOOTSTRAP_SHORT_RUN_MIN_AVG_WINDOW_SIMILARITY = 0.5
_STAGE_TWO_BOOTSTRAP_MIN_SPEAKER_OVERLAP_RATIO = 0.45
_STAGE_TWO_BOOTSTRAP_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def _suppress_chatty_logger_propagation() -> None:
    """Prevent noisy third-party logs from leaking to terminal handlers."""
    for logger_name in ["google", "google_genai", "httpx", "a2a", "uvicorn"]:
        logging.getLogger(logger_name).propagate = False


def _build_a2a_app(name: str, description: str, port: int, executor: AgentExecutor):
    """Build a Starlette A2A application from an executor."""
    agent_card = AgentCard(
        name=name,
        description=description,
        url=f"http://127.0.0.1:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="default",
                name=name,
                description=description,
                tags=["summarization"],
            )
        ],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        queue_manager=InMemoryQueueManager(),
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    return a2a_app.build()


def _run_server_in_thread(name: str, app, port: int) -> threading.Thread:
    """Run a uvicorn server in a daemon thread."""
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _serve() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_serve, name=name, daemon=True)
    thread.start()
    return thread


def _to_plain_dict(value: DictConfig | dict | None) -> dict:
    """Convert Hydra config nodes into plain dictionaries."""
    if isinstance(value, DictConfig):
        resolved = OmegaConf.to_container(value, resolve=True)
        if isinstance(resolved, dict):
            return resolved
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _has_target_config(value: dict | None) -> bool:
    """Return whether a config mapping includes a non-empty Hydra target."""
    if not isinstance(value, dict):
        return False
    target = value.get("_target_")
    return isinstance(target, str) and bool(target.strip())


def _instantiate_llm_provider(
    llm_cfg: DictConfig | dict,
    *,
    enable_logging: bool,
) -> LLMProvider:
    """Instantiate an LLM provider and apply logging wrapper when enabled."""
    llm: LLMProvider = hydra.utils.instantiate(llm_cfg)
    _attach_stream_callback(llm)
    if enable_logging:
        return LoggingLLMProvider(inner_provider=llm)
    return llm


def _attach_stream_callback(llm: LLMProvider) -> None:
    """
    Attach a UI stream callback when a provider supports callback injection.

    This keeps UI dependencies in the composition root rather than provider
    implementation modules.
    """
    callback_setter = getattr(llm, "set_response_callback", None)
    if callable(callback_setter):
        callback_setter(RichConsoleResponseCallback())
        return

    inner_provider = getattr(llm, "inner_provider", None)
    if isinstance(inner_provider, LLMProvider):
        _attach_stream_callback(inner_provider)


def _resolve_stage_llm(
    override_cfg: dict | None,
    *,
    shared_llm: LLMProvider,
    enable_logging: bool,
) -> tuple[LLMProvider, str]:
    """Resolve stage LLM provider, falling back to shared provider when unset."""
    if _has_target_config(override_cfg):
        return (
            _instantiate_llm_provider(override_cfg, enable_logging=enable_logging),
            "override",
        )
    return shared_llm, "shared"


def _should_defer_speaker_samples(audio_cfg_dict: dict[str, Any]) -> bool:
    """Return whether speaker-sample generation should move off the summary critical path."""
    speaker_samples_cfg = _to_plain_dict(audio_cfg_dict.get("speaker_samples", {}))
    return bool(speaker_samples_cfg.get("enabled", True)) and bool(
        speaker_samples_cfg.get("defer_until_voice_clone", False)
    )


def _is_live_draft_audio_requested(audio_cfg_dict: dict[str, Any]) -> bool:
    """Return whether live stage-2/stage-3 audio drafting is enabled in config."""
    voice_clone_cfg = _to_plain_dict(audio_cfg_dict.get("voice_clone", {}))
    live_drafting_cfg = _to_plain_dict(voice_clone_cfg.get("live_drafting", {}))
    return bool(voice_clone_cfg.get("enabled", False)) and bool(
        live_drafting_cfg.get("enabled", True)
    )


def _save_eta_profile_if_supported(
    *,
    eta_strategy: StageEtaStrategy | None,
    profile_path: Path | None,
    profile_context: dict[str, str] | None,
    failure_message: str,
) -> None:
    """Persist learned ETA state when the strategy supports profile storage."""
    if profile_path is None or not isinstance(eta_strategy, EtaProfilePersistence):
        return
    try:
        eta_strategy.save_profile(profile_path, context=profile_context or {})
    except Exception:
        event_bus.publish("system_message", failure_message)


def _build_speaker_sample_preparer(
    *,
    stage_start: str,
    audio_cfg_dict: dict[str, Any],
    project_root: Path,
    transcript_path: str,
    eta_strategy: StageEtaStrategy | None = None,
    eta_profile_path: Path | None = None,
    eta_profile_context: dict[str, str] | None = None,
    eta_update_interval_seconds: float = 10.0,
    eta_progress_smoothing: float = 0.25,
    eta_overrun_factor: float = 1.15,
    eta_headroom_seconds: float = 1.0,
) -> Callable[[Transcript], Transcript]:
    """Build a callback that can prepare speaker samples immediately before voice cloning."""

    def _prepare(transcript: Transcript) -> Transcript:
        return _run_post_transcript_speaker_sample_step(
            stage_start=stage_start,
            audio_cfg_dict=audio_cfg_dict,
            project_root=project_root,
            transcript_path=transcript_path,
            transcript=transcript,
            eta_strategy=eta_strategy,
            eta_profile_path=eta_profile_path,
            eta_profile_context=eta_profile_context,
            eta_update_interval_seconds=eta_update_interval_seconds,
            eta_progress_smoothing=eta_progress_smoothing,
            eta_overrun_factor=eta_overrun_factor,
            eta_headroom_seconds=eta_headroom_seconds,
        )

    return _prepare


def _run_post_transcript_speaker_sample_step(
    *,
    stage_start: str,
    audio_cfg_dict: dict[str, Any],
    project_root: Path,
    transcript_path: str,
    transcript: Transcript,
    eta_strategy: StageEtaStrategy | None = None,
    eta_profile_path: Path | None = None,
    eta_profile_context: dict[str, str] | None = None,
    eta_update_interval_seconds: float = 10.0,
    eta_progress_smoothing: float = 0.25,
    eta_overrun_factor: float = 1.15,
    eta_headroom_seconds: float = 1.0,
) -> Transcript:
    """
    Generate per-speaker voice samples after transcript availability.

    Args:
        stage_start: Active start stage from pipeline plan.
        audio_cfg_dict: Resolved audio config mapping.
        project_root: Repository root for relative path resolution.
        transcript_path: Path to transcript JSON to update.
        transcript: Loaded transcript domain DTO.

    Returns:
        Transcript with updated speaker-sample metadata when generation runs;
        otherwise the original transcript.
    """
    if stage_start not in {"stage-1", "stage-2"}:
        return transcript

    speaker_samples_cfg = _to_plain_dict(audio_cfg_dict.get("speaker_samples", {}))
    if not bool(speaker_samples_cfg.get("enabled", True)):
        return transcript

    existing_manifest_path = _resolve_existing_speaker_samples_manifest_path(
        transcript=transcript,
        project_root=project_root,
    )
    if stage_start == "stage-2" and existing_manifest_path is not None:
        event_bus.publish(
            "system_message",
            (
                "Reusing speaker sample manifest from transcript metadata: "
                f"{existing_manifest_path}"
            ),
        )
        return transcript

    work_dir = resolve_path(
        str(audio_cfg_dict.get("work_dir", "artifacts/audio_stage")),
        base_dir=project_root,
    )
    output_dir_name = str(speaker_samples_cfg.get("output_dir_name", "speaker_samples"))
    if not output_dir_name.strip():
        raise ValueError("audio.speaker_samples.output_dir_name must be non-empty.")
    samples_output_dir = resolve_path(output_dir_name, base_dir=work_dir)
    if stage_start == "stage-2":
        return _bootstrap_stage_two_speaker_samples_from_audio(
            audio_cfg_dict=audio_cfg_dict,
            project_root=project_root,
            transcript_path=transcript_path,
            transcript=transcript,
            samples_output_dir=samples_output_dir,
        )
    # Prefer the vocals stem when transcript metadata provides it, but fall back
    # to configured audio for reusable transcripts that do not carry stem paths.
    source_audio_path = resolve_sample_source_audio_path(
        source_mode="vocals",
        transcript_metadata=transcript.metadata,
        configured_audio_path=str(audio_cfg_dict.get("audio_path", "")),
        base_dir=project_root,
    )
    sample_generator = build_speaker_sample_generator(audio_cfg_dict)
    expected_sample_count = len(
        {
            segment.speaker
            for segment in transcript.segments
            if isinstance(segment.speaker, str) and segment.speaker.strip()
        }
    )
    eta_tracker: DynamicEtaTracker | None = None
    if (
        expected_sample_count > 0
        and isinstance(eta_strategy, UnitStageEtaStrategy)
        and (
            not isinstance(eta_strategy, UnitStageEtaHistory)
            or eta_strategy.has_unit_stage_history(stage="speaker_samples")
        )
    ):
        estimated_total_seconds = eta_strategy.estimate_unit_stage_total_seconds(
            stage="speaker_samples",
            total_units=expected_sample_count,
        )
        if estimated_total_seconds is not None:
            eta_tracker = DynamicEtaTracker(
                initial_total_seconds=estimated_total_seconds,
                progress_smoothing=eta_progress_smoothing,
                overrun_factor=eta_overrun_factor,
                headroom_seconds=eta_headroom_seconds,
            )

    if eta_tracker is None:
        event_bus.publish(
            "system_message",
            (
                "Generating speaker voice samples from "
                f"{source_audio_path} into {samples_output_dir}"
            ),
        )
    else:
        event_bus.publish(
            "system_message",
            (
                "Generating speaker voice samples from "
                f"{source_audio_path} into {samples_output_dir} "
                f"(estimated time left {format_eta_seconds(eta_tracker.initial_total_seconds)})"
            ),
        )

    started_at = time.monotonic()
    tracker_lock = threading.Lock()
    stop_event = threading.Event()
    ticker_thread: threading.Thread | None = None

    progress_callback: StageProgressCallback | None = None
    if eta_tracker is not None:
        def _on_progress(update: StageProgressUpdate) -> None:
            elapsed_seconds = max(0.0, time.monotonic() - started_at)
            with tracker_lock:
                eta_tracker.observe_progress(
                    elapsed_seconds=elapsed_seconds,
                    update=update,
                )

        progress_callback = _on_progress

        if eta_update_interval_seconds > 0:
            def _ticker() -> None:
                while not stop_event.wait(eta_update_interval_seconds):
                    elapsed_seconds = max(0.0, time.monotonic() - started_at)
                    with tracker_lock:
                        remaining_seconds = eta_tracker.estimate_signed_remaining_seconds(
                            elapsed_seconds=elapsed_seconds
                        )
                    if remaining_seconds >= 0:
                        event_bus.publish(
                            "status_message",
                            (
                                "Speaker sample stage: estimated time left "
                                f"{format_eta_seconds(remaining_seconds)}"
                            ),
                            inline=True,
                        )
                        continue
                    event_bus.publish(
                        "status_message",
                        (
                            "Speaker sample stage: running longer than estimate by "
                            f"{format_eta_seconds(abs(remaining_seconds))}"
                        ),
                        inline=True,
                    )

            ticker_thread = threading.Thread(target=_ticker, daemon=True)
            ticker_thread.start()

    try:
        sample_result = sample_generator.generate(
            transcript_payload=transcript.to_payload(),
            source_audio_path=source_audio_path,
            output_dir=samples_output_dir,
            progress_callback=progress_callback,
        )
    finally:
        stop_event.set()
        if ticker_thread is not None:
            ticker_thread.join(timeout=0.2)

    elapsed_seconds = max(0.0, time.monotonic() - started_at)
    if (
        isinstance(eta_strategy, UnitStageEtaLearner)
        and len(sample_result.artifacts) > 0
    ):
        try:
            eta_strategy.observe_unit_stage_duration(
                stage="speaker_samples",
                total_units=len(sample_result.artifacts),
                elapsed_seconds=elapsed_seconds,
            )
            _save_eta_profile_if_supported(
                eta_strategy=eta_strategy,
                profile_path=eta_profile_path,
                profile_context=eta_profile_context,
                failure_message="Speaker sample stage: ETA profile save failed.",
            )
        except Exception:
            event_bus.publish(
                "system_message",
                "Speaker sample stage: ETA learning update skipped.",
            )

    event_bus.publish(
        "status_message",
        (
            f"Generated {len(sample_result.artifacts)} speaker samples "
            f"at {sample_result.output_dir} in {format_eta_seconds(elapsed_seconds)}"
        ),
    )

    metadata = dict(transcript.metadata)
    metadata.update(
        {
            "speaker_samples_manifest_path": str(sample_result.manifest_path),
            "speaker_samples_dir": str(sample_result.output_dir),
            "speaker_sample_count": len(sample_result.artifacts),
            "speaker_samples_generated_at_utc": sample_result.generated_at_utc,
        }
    )
    updated_transcript = transcript.with_metadata(metadata)

    resolved_transcript_path = resolve_path(transcript_path, base_dir=project_root)
    write_transcript_atomic(
        cast(TranscriptPayload, updated_transcript.to_payload()),
        resolved_transcript_path,
    )
    event_bus.publish(
        "status_message",
        f"Speaker sample manifest written to {sample_result.manifest_path}",
    )
    return updated_transcript


def _resolve_existing_speaker_samples_manifest_path(
    *,
    transcript: Transcript,
    project_root: Path,
) -> Path | None:
    """Resolve an existing transcript-linked speaker-sample manifest when present."""
    manifest_value = str(
        transcript.metadata.get("speaker_samples_manifest_path", "")
    ).strip()
    if not manifest_value:
        return None
    candidate = Path(manifest_value).expanduser()
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    return candidate if candidate.is_file() else None


def _load_bootstrap_transcript(bootstrap_transcript_path: Path) -> Transcript:
    """Load the stage-2 bootstrap transcript artifact."""
    payload = json.loads(bootstrap_transcript_path.read_text(encoding="utf-8-sig"))
    return Transcript.from_mapping(payload)


def _ordered_transcript_speakers(transcript: Transcript) -> tuple[str, ...]:
    """Return speaker IDs in first-seen order from one transcript."""
    seen: set[str] = set()
    ordered: list[str] = []
    for segment in transcript.segments:
        speaker = segment.speaker.strip()
        if not speaker or speaker in seen:
            continue
        seen.add(speaker)
        ordered.append(speaker)
    return tuple(ordered)


def _tokenize_stage_two_bootstrap_text(text: str) -> set[str]:
    """Extract normalized lexical tokens for stage-2 transcript matching."""
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_'-]{1,}", text.lower())
    return {
        token
        for token in candidates
        if len(token) >= 4 and token not in _STAGE_TWO_BOOTSTRAP_STOPWORDS
    }


def _stage_two_text_similarity(text_a: str, text_b: str) -> float:
    """Return Jaccard similarity for two transcript text windows."""
    tokens_a = _tokenize_stage_two_bootstrap_text(text_a)
    tokens_b = _tokenize_stage_two_bootstrap_text(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / float(len(tokens_a | tokens_b))


def _build_stage_two_time_window_texts(
    transcript: Transcript,
) -> dict[int, str]:
    """Bucket transcript text into fixed windows for stage-2 compatibility checks."""
    windows: dict[int, list[str]] = {}
    max_time_ms = _STAGE_TWO_BOOTSTRAP_WINDOW_MS * _STAGE_TWO_BOOTSTRAP_MAX_WINDOWS
    for segment in transcript.segments:
        if segment.start_time >= max_time_ms:
            continue
        bucket = max(0, segment.start_time // _STAGE_TWO_BOOTSTRAP_WINDOW_MS)
        windows.setdefault(bucket, []).append(segment.text)
    return {
        bucket: " ".join(parts).strip()
        for bucket, parts in windows.items()
        if any(part.strip() for part in parts)
    }


def _stage_two_bootstrap_window_similarities(
    *,
    transcript: Transcript,
    bootstrap_transcript: Transcript,
) -> dict[int, float]:
    """Compare overlapping stage-2 transcript windows and return per-window scores."""
    transcript_windows = _build_stage_two_time_window_texts(transcript)
    bootstrap_windows = _build_stage_two_time_window_texts(bootstrap_transcript)
    similarities: dict[int, float] = {}
    for bucket in sorted(set(transcript_windows) & set(bootstrap_windows)):
        score = _stage_two_text_similarity(
            transcript_windows[bucket],
            bootstrap_windows[bucket],
        )
        similarities[bucket] = score
    return similarities


def _validate_stage_two_bootstrap_transcript_compatibility(
    *,
    transcript: Transcript,
    bootstrap_transcript: Transcript,
    bootstrap_audio_path: Path,
    bootstrap_transcript_path: Path,
) -> None:
    """Fail early when bootstrap audio does not appear to match the reusable transcript."""
    window_similarities = _stage_two_bootstrap_window_similarities(
        transcript=transcript,
        bootstrap_transcript=bootstrap_transcript,
    )
    if not window_similarities:
        raise NonRetryableAudioStageError(
            "Stage-2 speaker-sample bootstrap could not compare the reusable "
            "transcript against the inferred bootstrap transcript. "
            f"Bootstrap transcript: {bootstrap_transcript_path}. "
            "Provide matching source audio via audio.audio_path or reuse a "
            "transcript that already carries speaker_samples_manifest_path."
        )
    compared_window_count = len(window_similarities)
    average_similarity = sum(window_similarities.values()) / float(compared_window_count)
    strong_window_ratio = (
        sum(
            1
            for score in window_similarities.values()
            if score >= _STAGE_TWO_BOOTSTRAP_STRONG_WINDOW_SIMILARITY
        )
        / float(compared_window_count)
    )
    if compared_window_count < 3:
        is_compatible = (
            average_similarity
            >= _STAGE_TWO_BOOTSTRAP_SHORT_RUN_MIN_AVG_WINDOW_SIMILARITY
        )
    else:
        is_compatible = (
            average_similarity >= _STAGE_TWO_BOOTSTRAP_MIN_AVG_WINDOW_SIMILARITY
            and strong_window_ratio >= _STAGE_TWO_BOOTSTRAP_MIN_STRONG_WINDOW_RATIO
        )
    if is_compatible:
        return
    raise NonRetryableAudioStageError(
        "Stage-2 speaker-sample bootstrap audio does not appear to match the "
        "reusable transcript. "
        f"Compared {compared_window_count} overlapping "
        f"{_STAGE_TWO_BOOTSTRAP_WINDOW_MS // 1000}-second transcript windows "
        f"from {bootstrap_transcript_path} using audio {bootstrap_audio_path}; "
        f"average lexical similarity={average_similarity:.3f}, "
        f"strong_window_ratio={strong_window_ratio:.3f}. "
        "Provide the matching source audio via audio.audio_path or reuse a "
        "transcript that already carries speaker_samples_manifest_path."
    )


def _speaker_interval_overlap_ms(
    *,
    transcript: Transcript,
    bootstrap_transcript: Transcript,
) -> dict[str, dict[str, int]]:
    """Measure overlap milliseconds between bootstrap and reusable speaker labels."""
    overlap_by_bootstrap_speaker: dict[str, dict[str, int]] = {}
    transcript_speakers = _ordered_transcript_speakers(transcript)
    for bootstrap_speaker in _ordered_transcript_speakers(bootstrap_transcript):
        overlap_by_bootstrap_speaker[bootstrap_speaker] = {
            transcript_speaker: 0 for transcript_speaker in transcript_speakers
        }
    for bootstrap_segment in bootstrap_transcript.segments:
        bootstrap_speaker = bootstrap_segment.speaker.strip()
        if bootstrap_speaker not in overlap_by_bootstrap_speaker:
            continue
        for transcript_segment in transcript.segments:
            transcript_speaker = transcript_segment.speaker.strip()
            if transcript_speaker not in overlap_by_bootstrap_speaker[bootstrap_speaker]:
                continue
            overlap_ms = min(
                bootstrap_segment.end_time,
                transcript_segment.end_time,
            ) - max(
                bootstrap_segment.start_time,
                transcript_segment.start_time,
            )
            if overlap_ms <= 0:
                continue
            overlap_by_bootstrap_speaker[bootstrap_speaker][transcript_speaker] += (
                overlap_ms
            )
    return overlap_by_bootstrap_speaker


def _speaker_total_duration_ms(transcript: Transcript) -> dict[str, int]:
    """Return accumulated segment duration per speaker."""
    durations: dict[str, int] = {}
    for segment in transcript.segments:
        speaker = segment.speaker.strip()
        if not speaker:
            continue
        durations[speaker] = durations.get(speaker, 0) + max(
            0, segment.end_time - segment.start_time
        )
    return durations


def _resolve_stage_two_bootstrap_speaker_mapping(
    *,
    transcript: Transcript,
    bootstrap_transcript: Transcript,
) -> dict[str, str]:
    """Map bootstrap speaker labels back onto reusable transcript labels."""
    transcript_speakers = _ordered_transcript_speakers(transcript)
    bootstrap_speakers = _ordered_transcript_speakers(bootstrap_transcript)
    if not transcript_speakers:
        raise NonRetryableAudioStageError(
            "Reusable stage-2 transcript does not contain any speaker labels."
        )
    if not bootstrap_speakers:
        raise NonRetryableAudioStageError(
            "Bootstrap transcript does not contain any speaker labels."
        )
    if len(transcript_speakers) != len(bootstrap_speakers):
        raise NonRetryableAudioStageError(
            "Stage-2 speaker-sample bootstrap speaker coverage does not match the "
            "reusable transcript. "
            f"Reusable transcript speakers: {list(transcript_speakers)}. "
            f"Bootstrap transcript speakers: {list(bootstrap_speakers)}. "
            "Provide matching source audio or reuse a transcript that already "
            "includes a valid speaker_samples_manifest_path."
        )
    overlap_ms = _speaker_interval_overlap_ms(
        transcript=transcript,
        bootstrap_transcript=bootstrap_transcript,
    )
    bootstrap_durations = _speaker_total_duration_ms(bootstrap_transcript)
    if len(bootstrap_speakers) <= 6:
        best_mapping: dict[str, str] | None = None
        best_total_overlap = -1
        for transcript_permutation in permutations(
            transcript_speakers,
            len(bootstrap_speakers),
        ):
            candidate_mapping = {
                bootstrap_speaker: transcript_speaker
                for bootstrap_speaker, transcript_speaker in zip(
                    bootstrap_speakers,
                    transcript_permutation,
                    strict=True,
                )
            }
            total_overlap = sum(
                overlap_ms[bootstrap_speaker][candidate_mapping[bootstrap_speaker]]
                for bootstrap_speaker in bootstrap_speakers
            )
            if total_overlap > best_total_overlap:
                best_total_overlap = total_overlap
                best_mapping = candidate_mapping
        if best_mapping is None:
            raise NonRetryableAudioStageError(
                "Unable to determine bootstrap speaker mapping for stage-2 "
                "speaker-sample generation."
            )
        mapping = best_mapping
    else:
        scored_pairs = sorted(
            (
                (
                    overlap_ms[bootstrap_speaker][transcript_speaker],
                    bootstrap_speaker,
                    transcript_speaker,
                )
                for bootstrap_speaker in bootstrap_speakers
                for transcript_speaker in transcript_speakers
            ),
            reverse=True,
        )
        mapping: dict[str, str] = {}
        used_transcript_speakers: set[str] = set()
        for score, bootstrap_speaker, transcript_speaker in scored_pairs:
            if bootstrap_speaker in mapping or transcript_speaker in used_transcript_speakers:
                continue
            mapping[bootstrap_speaker] = transcript_speaker
            used_transcript_speakers.add(transcript_speaker)
            if len(mapping) == len(bootstrap_speakers):
                break
        if len(mapping) != len(bootstrap_speakers):
            raise NonRetryableAudioStageError(
                "Unable to determine bootstrap speaker mapping for stage-2 "
                "speaker-sample generation."
            )
    for bootstrap_speaker, transcript_speaker in mapping.items():
        duration_ms = bootstrap_durations.get(bootstrap_speaker, 0)
        speaker_overlap_ratio = (
            overlap_ms[bootstrap_speaker][transcript_speaker] / float(duration_ms)
            if duration_ms > 0
            else 0.0
        )
        if speaker_overlap_ratio < _STAGE_TWO_BOOTSTRAP_MIN_SPEAKER_OVERLAP_RATIO:
            raise NonRetryableAudioStageError(
                "Stage-2 bootstrap inferred speaker labels could not be aligned "
                "confidently back to the reusable transcript. "
                f"Speaker '{bootstrap_speaker}' best matched '{transcript_speaker}' "
                f"with overlap ratio={speaker_overlap_ratio:.3f}. "
                "Provide matching source audio or reuse a transcript that already "
                "includes a valid speaker_samples_manifest_path."
            )
    return mapping


def _rewrite_stage_two_speaker_sample_manifest(
    *,
    manifest_path: Path,
    speaker_mapping: dict[str, str],
) -> None:
    """Rewrite a bootstrap speaker-sample manifest using reusable transcript labels."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise NonRetryableAudioStageError(
            "Stage-2 speaker-sample manifest could not be read for speaker "
            f"label remapping: {manifest_path}"
        ) from exc
    samples = payload.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise NonRetryableAudioStageError(
            "Stage-2 speaker-sample manifest does not contain any sample entries "
            f"to remap: {manifest_path}"
        )
    seen_manifest_speakers: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        bootstrap_speaker = str(sample.get("speaker", "")).strip()
        if not bootstrap_speaker:
            continue
        seen_manifest_speakers.add(bootstrap_speaker)
        mapped_speaker = speaker_mapping.get(bootstrap_speaker)
        if mapped_speaker is None:
            raise NonRetryableAudioStageError(
                "Stage-2 bootstrap manifest contains a speaker that could not be "
                f"mapped back to the reusable transcript: {bootstrap_speaker}"
            )
        sample["bootstrap_speaker"] = bootstrap_speaker
        sample["speaker"] = mapped_speaker
    missing_manifest_speakers = sorted(set(speaker_mapping) - seen_manifest_speakers)
    if missing_manifest_speakers:
        raise NonRetryableAudioStageError(
            "Stage-2 bootstrap manifest is missing sample entries for bootstrap "
            f"speaker(s): {missing_manifest_speakers}"
        )
    payload["stage_two_bootstrap_speaker_mapping"] = dict(speaker_mapping)
    temp_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(manifest_path)
    except Exception as exc:
        raise NonRetryableAudioStageError(
            "Failed to rewrite the stage-2 speaker-sample manifest after "
            f"speaker-label remapping: {manifest_path}"
        ) from exc


def _bootstrap_stage_two_speaker_samples_from_audio(
    *,
    audio_cfg_dict: dict[str, Any],
    project_root: Path,
    transcript_path: str,
    transcript: Transcript,
    samples_output_dir: Path,
) -> Transcript:
    """Bootstrap stage-2 speaker samples from source audio when manifest is missing."""
    bootstrap_audio_path = _resolve_stage_two_speaker_sample_bootstrap_audio_path(
        transcript=transcript,
        audio_cfg_dict=audio_cfg_dict,
        project_root=project_root,
    )
    work_dir = resolve_path(
        str(audio_cfg_dict.get("work_dir", "artifacts/audio_stage")),
        base_dir=project_root,
    )
    bootstrap_transcript_path = work_dir / "speaker_sample_bootstrap.transcript.json"
    bootstrap_audio_work_dir = work_dir / "speaker_sample_bootstrap_audio"
    event_bus.publish(
        "system_message",
        (
            "Speaker sample manifest missing for stage-2 transcript. "
            "Bootstrapping fresh speaker samples from source audio via source "
            "separation, transcription, and diarization."
        ),
    )
    started_at = time.monotonic()
    bootstrap_result = bootstrap_speaker_samples_from_audio(
        project_root=project_root,
        audio_cfg=audio_cfg_dict,
        audio_path=bootstrap_audio_path,
        bootstrap_transcript_path=bootstrap_transcript_path,
        bootstrap_audio_work_dir=bootstrap_audio_work_dir,
        speaker_samples_output_dir=samples_output_dir,
    )
    elapsed_seconds = max(0.0, time.monotonic() - started_at)
    bootstrap_transcript = _load_bootstrap_transcript(
        bootstrap_result.transcript_path
    )
    _validate_stage_two_bootstrap_transcript_compatibility(
        transcript=transcript,
        bootstrap_transcript=bootstrap_transcript,
        bootstrap_audio_path=bootstrap_audio_path,
        bootstrap_transcript_path=bootstrap_result.transcript_path,
    )
    sample_result = bootstrap_result.generation_result
    speaker_mapping = _resolve_stage_two_bootstrap_speaker_mapping(
        transcript=transcript,
        bootstrap_transcript=bootstrap_transcript,
    )
    if any(
        bootstrap_speaker != transcript_speaker
        for bootstrap_speaker, transcript_speaker in speaker_mapping.items()
    ):
        _rewrite_stage_two_speaker_sample_manifest(
            manifest_path=sample_result.manifest_path,
            speaker_mapping=speaker_mapping,
        )
        event_bus.publish(
            "system_message",
            (
                "Remapped stage-2 bootstrap speaker labels onto the reusable "
                "transcript speaker IDs: "
                + ", ".join(
                    f"{bootstrap_speaker}->{transcript_speaker}"
                    for bootstrap_speaker, transcript_speaker in sorted(
                        speaker_mapping.items()
                    )
                )
            ),
        )
    event_bus.publish(
        "status_message",
        (
            f"Generated {len(sample_result.artifacts)} speaker samples "
            f"at {sample_result.output_dir} in {format_eta_seconds(elapsed_seconds)}"
        ),
    )
    metadata = dict(transcript.metadata)
    metadata.update(
        {
            "speaker_samples_manifest_path": str(sample_result.manifest_path),
            "speaker_samples_dir": str(sample_result.output_dir),
            "speaker_sample_count": len(sample_result.artifacts),
            "speaker_samples_generated_at_utc": sample_result.generated_at_utc,
        }
    )
    updated_transcript = transcript.with_metadata(metadata)
    resolved_transcript_path = resolve_path(transcript_path, base_dir=project_root)
    write_transcript_atomic(
        cast(TranscriptPayload, updated_transcript.to_payload()),
        resolved_transcript_path,
    )
    event_bus.publish(
        "status_message",
        f"Speaker sample manifest written to {sample_result.manifest_path}",
    )
    return updated_transcript


def _resolve_stage_two_speaker_sample_bootstrap_audio_path(
    *,
    transcript: Transcript,
    audio_cfg_dict: dict[str, Any],
    project_root: Path,
) -> Path:
    """Resolve source audio required to rebuild stage-2 speaker samples."""
    configured_audio_path = str(audio_cfg_dict.get("audio_path", "")).strip()
    if configured_audio_path:
        candidate = resolve_path(configured_audio_path, base_dir=project_root)
        if candidate.exists():
            return candidate
        raise NonRetryableAudioStageError(
            "Configured audio.audio_path for stage-2 speaker-sample bootstrap "
            f"does not exist: {candidate}"
        )
    source_audio_value = str(transcript.metadata.get("source_audio_path", "")).strip()
    if source_audio_value:
        candidate = Path(source_audio_value).expanduser()
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        if candidate.exists():
            return candidate
        raise NonRetryableAudioStageError(
            "Transcript metadata source_audio_path for stage-2 speaker-sample "
            f"bootstrap does not exist: {candidate}"
        )
    raise NonRetryableAudioStageError(
        "Stage-2 speaker-sample bootstrap requires source audio when the "
        "transcript does not already carry a reusable "
        "speaker_samples_manifest_path. Set audio.audio_path or provide "
        "metadata.source_audio_path."
    )


def _wait_for_agent_servers(
    *,
    checker: AgentHealthChecker,
    servers: Sequence[tuple[str, int]],
    overall_timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 0.2,
    request_timeout_seconds: float = 1.0,
) -> None:
    """Block until required A2A servers are healthy or terminate the process."""
    if not servers:
        return
    event_bus.publish("system_message", "Waiting for A2A servers to start...")
    if checker.wait_for_many(
        servers,
        overall_timeout_seconds=overall_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        request_timeout_seconds=request_timeout_seconds,
    ):
        return
    sys.exit(1)


def _resolve_duration_targets(
    orchestrator_cfg: dict[str, Any],
) -> tuple[int, float]:
    """Resolve target duration and tolerance from config."""
    raw_target_seconds = orchestrator_cfg.get("target_seconds")
    if raw_target_seconds in (None, ""):
        target_minutes = float(orchestrator_cfg.get("target_minutes", 5.0))
        if target_minutes <= 0:
            raise ValueError("orchestrator.target_minutes must be greater than zero.")
        target_seconds = max(1, int(round(target_minutes * 60.0)))
    else:
        target_seconds = int(round(float(raw_target_seconds)))
        if target_seconds <= 0:
            raise ValueError("orchestrator.target_seconds must be greater than zero.")
    duration_tolerance_ratio = float(
        orchestrator_cfg.get(
            "duration_tolerance_ratio",
            DEFAULT_DURATION_TOLERANCE_RATIO,
        )
    )
    if not 0.0 < duration_tolerance_ratio < 1.0:
        raise ValueError(
            "orchestrator.duration_tolerance_ratio must be within (0.0, 1.0)."
        )
    return target_seconds, duration_tolerance_ratio


@hydra.main(
    version_base=None,
    config_path=str(DEFAULT_CONFIG_PATH.parent),
    config_name=DEFAULT_CONFIG_PATH.stem,
)
def main(cfg: DictConfig) -> None:
    """Run the configured summarization pipeline."""
    logging_cfg = _to_plain_dict(cfg.get("logging", {}))
    configure_logger(logging_cfg)

    if not bool(logging_cfg.get("print_to_terminal", False)):
        _suppress_chatty_logger_propagation()

    project_root = Path(hydra.utils.get_original_cwd())
    audio_cfg_dict = _to_plain_dict(cfg.get("audio", {}))
    voice_clone_cfg_dict = _to_plain_dict(audio_cfg_dict.get("voice_clone", {}))
    live_draft_audio_requested = _is_live_draft_audio_requested(audio_cfg_dict)
    orchestrator_cfg_dict = _to_plain_dict(cfg.get("orchestrator", {}))
    target_seconds, duration_tolerance_ratio = _resolve_duration_targets(
        orchestrator_cfg_dict
    )
    pipeline_cfg_dict = _to_plain_dict(cfg.get("pipeline", {}))
    stage_plan = build_pipeline_stage_plan(pipeline_cfg_dict, project_root=project_root)
    runtime_device = resolve_device(str(audio_cfg_dict.get("device", "auto")))
    work_dir = resolve_path(
        str(audio_cfg_dict.get("work_dir", "artifacts/audio_stage")),
        base_dir=project_root,
    )
    audio_orchestrator = build_audio_to_script_orchestrator(audio_cfg_dict)
    eta_profile_path = work_dir / audio_orchestrator.eta_profile_filename
    eta_profile_context = audio_orchestrator._build_eta_profile_context(device=runtime_device)
    if not stage_plan.run_audio_stage:
        audio_orchestrator._load_eta_profile(
            profile_path=eta_profile_path,
            context=eta_profile_context,
        )

    event_bus.publish(
        "system_message",
        (
            "Pipeline plan: "
            f"start_stage={stage_plan.start_stage}"
        ),
    )

    transcript_path = str(cfg.transcript_path)
    input_audio_path: Path | None = None
    configured_audio_path_value = str(audio_cfg_dict.get("audio_path", "")).strip()
    if configured_audio_path_value:
        candidate_audio_path = resolve_path(
            configured_audio_path_value,
            base_dir=project_root,
        )
        if candidate_audio_path.exists():
            input_audio_path = candidate_audio_path

    if stage_plan.run_audio_stage:
        audio_path_value = str(audio_cfg_dict.get("audio_path", "")).strip()
        if not audio_path_value:
            raise ValueError(
                "audio.audio_path is required when pipeline.start_stage=stage-1. "
                "To skip stage-1, set pipeline.start_stage=stage-2."
            )

        input_audio_path = resolve_path(audio_path_value, base_dir=project_root)
        output_transcript_path = resolve_path(
            str(
                audio_cfg_dict.get(
                    "output_transcript_path",
                    "artifacts/transcripts/latest.transcript.json",
                )
            ),
            base_dir=project_root,
        )
        audio_stage_output = audio_orchestrator.run(
            input_audio_path=input_audio_path,
            output_transcript_path=output_transcript_path,
            work_dir=work_dir,
            device=runtime_device,
            metadata_overrides={
                "separator_model": str(
                    audio_cfg_dict.get("separation", {}).get("model", "htdemucs")
                ),
                "transcriber_model": str(
                    audio_cfg_dict.get("asr", {}).get("model", "large-v3")
                ),
                "diarizer_backend": str(
                    audio_cfg_dict.get("diarization", {}).get("provider", "nemo")
                ),
            },
        )
        transcript_path = str(audio_stage_output.transcript_path)
    else:
        event_bus.publish(
            "system_message",
            f"Skipping audio stage (start_stage={stage_plan.start_stage}).",
        )

    if stage_plan.start_stage == "stage-4":
        event_bus.publish(
            "system_message",
            (
                "Skipping transcript load for stage-4. Using the provided summary XML "
                "and voice-clone manifest only."
            ),
        )
        transcript = Transcript.from_mapping({"segments": [], "metadata": {}})
    else:
        resolved_transcript_path = resolve_path(transcript_path, base_dir=project_root)
        if not resolved_transcript_path.exists():
            if stage_plan.start_stage == "stage-3":
                raise ValueError(
                    "transcript_path must point to an existing transcript JSON with "
                    "metadata.speaker_samples_manifest_path when "
                    "pipeline.start_stage=stage-3."
                )
            raise FileNotFoundError(
                f"Transcript file not found: {resolved_transcript_path}"
            )

        transcript_path = str(resolved_transcript_path)
        event_bus.publish("system_message", f"Loading transcript from {transcript_path}...")
        transcript_payload = load_transcript(transcript_path)
        transcript = Transcript.from_mapping(transcript_payload)
        event_bus.publish(
            "system_message",
            f"Loaded {len(transcript.segments)} segments",
        )
    speaker_sample_preparer: Callable[[Transcript], Transcript] | None = None
    if stage_plan.start_stage == "stage-4":
        speaker_sample_preparer = None
    elif _should_defer_speaker_samples(audio_cfg_dict):
        event_bus.publish(
            "system_message",
            (
                "Speaker sample generation deferred until live draft voice-clone "
                "preparation."
                if live_draft_audio_requested
                else "Speaker sample generation deferred until voice clone stage."
            ),
        )
        speaker_sample_preparer = _build_speaker_sample_preparer(
            stage_start=stage_plan.start_stage,
            audio_cfg_dict=audio_cfg_dict,
            project_root=project_root,
            transcript_path=transcript_path,
            eta_strategy=audio_orchestrator.eta_strategy,
            eta_profile_path=eta_profile_path,
            eta_profile_context=eta_profile_context,
            eta_update_interval_seconds=audio_orchestrator.eta_update_interval_seconds,
            eta_progress_smoothing=audio_orchestrator.eta_progress_smoothing,
            eta_overrun_factor=audio_orchestrator.eta_overrun_factor,
            eta_headroom_seconds=audio_orchestrator.eta_headroom_seconds,
        )
    else:
        transcript = _run_post_transcript_speaker_sample_step(
            stage_start=stage_plan.start_stage,
            audio_cfg_dict=audio_cfg_dict,
            project_root=project_root,
            transcript_path=transcript_path,
            transcript=transcript,
            eta_strategy=audio_orchestrator.eta_strategy,
            eta_profile_path=eta_profile_path,
            eta_profile_context=eta_profile_context,
            eta_update_interval_seconds=audio_orchestrator.eta_update_interval_seconds,
            eta_progress_smoothing=audio_orchestrator.eta_progress_smoothing,
            eta_overrun_factor=audio_orchestrator.eta_overrun_factor,
            eta_headroom_seconds=audio_orchestrator.eta_headroom_seconds,
        )
    calibration = None
    if live_draft_audio_requested and (
        stage_plan.run_summarizer_stage
        or stage_plan.run_critic_stage
        or stage_plan.start_stage == "stage-3"
    ):
        event_bus.publish(
            "system_message",
            "Live draft voice-clone mode enabled. Skipping calibration and using rendered audio durations during stage-2.",
        )
    needs_calibration = (
        stage_plan.run_summarizer_stage or stage_plan.run_critic_stage
    ) and not live_draft_audio_requested
    if needs_calibration:
        speaker_manifest_path: Path | None = None
        if stage_plan.start_stage != "stage-4":
            manifest_value = str(
                transcript.metadata.get("speaker_samples_manifest_path", "")
            ).strip()
            if manifest_value:
                speaker_manifest_path = Path(manifest_value)
                if not speaker_manifest_path.is_absolute():
                    speaker_manifest_path = (
                        project_root / speaker_manifest_path
                    ).resolve()
        event_bus.publish(
            "system_message",
            "Ensuring voice-clone calibration artifact is available...",
        )
        calibration = ensure_voice_clone_calibration(
            project_root=project_root,
            audio_cfg=audio_cfg_dict,
            speaker_samples_manifest_path=speaker_manifest_path,
            transcript_path=(
                resolve_path(transcript_path, base_dir=project_root)
                if stage_plan.start_stage != "stage-4"
                else None
            ),
            audio_path=input_audio_path,
        )
        event_bus.publish(
            "status_message",
            (
                "Voice calibration ready at "
                f"{calibration.artifact_path}"
            ),
        )
        if stage_plan.start_stage != "stage-4":
            refreshed_transcript_path = resolve_path(
                transcript_path,
                base_dir=project_root,
            )
            if refreshed_transcript_path.exists():
                transcript_payload = load_transcript(str(refreshed_transcript_path))
                transcript = Transcript.from_mapping(transcript_payload)
    if stage_plan.run_summarizer_stage or stage_plan.run_critic_stage:
        stage_terminal_logging = bool(
            logging_cfg.get(
                "summarizer_critic_print_to_terminal",
                logging_cfg.get("print_to_terminal", False),
            )
        )
        current_terminal_logging = bool(logging_cfg.get("print_to_terminal", False))
        if stage_terminal_logging != current_terminal_logging:
            logging_cfg["print_to_terminal"] = stage_terminal_logging
            configure_logger(logging_cfg)
            if not stage_terminal_logging:
                _suppress_chatty_logger_propagation()

    event_bus.publish("system_message", "Instantiating LLM providers...")
    shared_agent_client = AgentClient(event_bus=event_bus)
    logging_enabled = bool(cfg.get("logging", {}).get("enabled", False))
    shared_llm = _instantiate_llm_provider(cfg.llm, enable_logging=logging_enabled)

    stage_llm_cfg = _to_plain_dict(cfg.get("stage_llm", {}))
    summarizer_llm, summarizer_llm_source = _resolve_stage_llm(
        stage_llm_cfg.get("summarizer"),
        shared_llm=shared_llm,
        enable_logging=logging_enabled,
    )
    critic_llm, critic_llm_source = _resolve_stage_llm(
        stage_llm_cfg.get("critic"),
        shared_llm=shared_llm,
        enable_logging=logging_enabled,
    )
    interjector_llm, interjector_llm_source = _resolve_stage_llm(
        stage_llm_cfg.get("interjector"),
        shared_llm=shared_llm,
        enable_logging=logging_enabled,
    )

    event_bus.publish(
        "system_message", f"Default LLM provider: {type(shared_llm).__name__}"
    )
    event_bus.publish(
        "system_message",
        (
            "Summarizer LLM provider: "
            f"{type(summarizer_llm).__name__} (source={summarizer_llm_source})"
        ),
    )
    event_bus.publish(
        "system_message",
        f"Critic LLM provider: {type(critic_llm).__name__} (source={critic_llm_source})",
    )
    event_bus.publish(
        "system_message",
        (
            "Interjector LLM provider: "
            f"{type(interjector_llm).__name__} (source={interjector_llm_source})"
        ),
    )

    voice_clone_orchestrator = build_voice_clone_orchestrator(
        audio_cfg_dict,
        project_root=project_root,
    )
    live_draft_audio_enabled = (
        live_draft_audio_requested and voice_clone_orchestrator is not None
    )
    interjector_orchestrator = build_interjector_orchestrator(
        audio_cfg_dict,
        llm=interjector_llm,
        project_root=project_root,
    )
    if stage_plan.start_stage == "stage-3" and voice_clone_orchestrator is None:
        raise ValueError(
            "audio.voice_clone.enabled must be true when "
            "pipeline.start_stage=stage-3."
        )
    if stage_plan.start_stage == "stage-4" and interjector_orchestrator is None:
        raise ValueError(
            "audio.interjector.enabled must be true when "
            "pipeline.start_stage=stage-4."
        )
    if (
        interjector_orchestrator is not None
        and voice_clone_orchestrator is None
        and stage_plan.start_stage != "stage-4"
    ):
        raise ValueError(
            "audio.interjector.enabled=true requires audio.voice_clone.enabled=true "
            "unless pipeline.start_stage=stage-4 with an existing "
            "pipeline.voice_clone_manifest_path."
        )

    voice_clone_gpu_heartbeat: VoiceCloneGpuHeartbeatService | None = None
    if voice_clone_orchestrator is None:
        event_bus.publish("system_message", "Voice clone stage disabled.")
    else:
        event_bus.publish(
            "system_message",
            (
                "Voice clone stage enabled with output dir "
                f"{voice_clone_orchestrator.output_dir}"
            ),
        )
        voice_clone_cfg = _to_plain_dict(audio_cfg_dict.get("voice_clone", {}))
        heartbeat_cfg = parse_voice_clone_gpu_heartbeat_config(
            _to_plain_dict(voice_clone_cfg.get("gpu_heartbeat", {}))
        )
        provider_device = str(
            getattr(voice_clone_orchestrator.provider, "device", "")
        ).strip().lower()
        if not heartbeat_cfg.enabled:
            event_bus.publish(
                "system_message",
                "Voice clone stage: dedicated GPU heartbeat disabled by config.",
            )
        elif os.name != "nt":
            event_bus.publish(
                "system_message",
                "Voice clone stage: dedicated GPU heartbeat is Windows-only; skipping.",
            )
        elif provider_device != "cuda":
            event_bus.publish(
                "system_message",
                (
                    "Voice clone stage: dedicated GPU heartbeat skipped because "
                    f"voice clone provider device is '{provider_device or 'unknown'}'."
                ),
            )
        else:
            voice_clone_gpu_heartbeat = VoiceCloneGpuHeartbeatService(
                probe=WindowsNvidiaDedicatedGpuProbe(
                    command_timeout_seconds=heartbeat_cfg.command_timeout_seconds,
                ),
                emit_status=lambda message: event_bus.publish("status_message", message),
                emit_system=lambda message: event_bus.publish("system_message", message),
                interval_seconds=heartbeat_cfg.interval_seconds,
                threshold_ratio=heartbeat_cfg.dedicated_usage_threshold_ratio,
                top_other_processes=heartbeat_cfg.top_other_processes,
            )
            event_bus.publish(
                "system_message",
                (
                    "Voice clone stage: dedicated GPU heartbeat configured "
                    f"(interval={heartbeat_cfg.interval_seconds}s, "
                    f"threshold={heartbeat_cfg.dedicated_usage_threshold_ratio:.0%})."
                ),
            )
    if interjector_orchestrator is None:
        event_bus.publish("system_message", "Interjector stage disabled.")
    else:
        event_bus.publish(
            "system_message",
            (
                "Interjector stage enabled with output dir "
                f"{interjector_orchestrator.output_dir}"
            ),
        )

    is_embedding_enabled = "NoOpEmbeddingProvider" not in cfg.embedding.get(
        "_target_", ""
    )

    if is_embedding_enabled and cfg.embedding.get("device", "") == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                event_bus.publish(
                    "error_message",
                    "CUDA is requested but not available on this device.",
                )
                print(
                    "\n[WARNING] CUDA is not available on this device, but it was requested for embeddings."
                )
                print("Options:")
                print("  [1] Fallback to CPU (slow)")
                print("  [2] Disable embeddings completely (use NoOpProvider)")

                choice = ""
                while choice not in ["1", "2"]:
                    choice = input("Please select an option [1/2]: ").strip()

                if choice == "1":
                    cfg.embedding.device = "cpu"
                    event_bus.publish(
                        "system_message", "Falling back to CPU for embeddings."
                    )
                elif choice == "2":
                    cfg.embedding._target_ = (
                        "card_framework.providers.null_provider.NoOpEmbeddingProvider"
                    )
                    is_embedding_enabled = False
                    event_bus.publish("system_message", "Embeddings disabled by user.")
        except ImportError:
            pass

    transcript_index = None
    if is_embedding_enabled and stage_plan.requires_retrieval_tools:
        event_bus.publish("system_message", "Instantiating Embedding provider...")
        embedding: EmbeddingProvider = hydra.utils.instantiate(cfg.embedding)
        event_bus.publish(
            "system_message", f"Embedding provider: {type(embedding).__name__}"
        )
        transcript_index = TranscriptIndex(embedding_provider=embedding)
    else:
        event_bus.publish(
            "system_message",
            "Embedding provider disabled or not required for selected pipeline plan.",
        )

    retrieval_port = cfg.ports.retrieval
    summarizer_port = cfg.ports.summarizer
    critic_port = cfg.ports.critic
    agents_cfg = _to_plain_dict(cfg.get("agents", {}))
    summarizer_cfg = _to_plain_dict(agents_cfg.get("summarizer", {}))
    critic_cfg = _to_plain_dict(agents_cfg.get("critic", {}))

    if transcript_index is not None:
        event_bus.publish(
            "system_message",
            f"Starting Info Retrieval A2A server on port {retrieval_port}...",
        )
        retrieval_app = _build_a2a_app(
            name="InfoRetrieval",
            description="Indexes transcript segments and retrieves relevant ones.",
            port=retrieval_port,
            executor=InfoRetrievalExecutor(transcript_index),
        )
        _run_server_in_thread("retrieval-a2a", retrieval_app, retrieval_port)
    else:
        event_bus.publish("system_message", "Info Retrieval Agent disabled.")

    if stage_plan.run_summarizer_stage:
        if calibration is None and not live_draft_audio_enabled:
            raise RuntimeError(
                "Summarizer stage requires a loaded voice-clone calibration artifact when live drafting is disabled."
            )
        event_bus.publish(
            "system_message",
            f"Starting Summarizer A2A server on port {summarizer_port}...",
        )
        summarizer_app = _build_a2a_app(
            name="Summarizer",
            description="Produces abstractive summaries.",
            port=summarizer_port,
            executor=SummarizerExecutor(
                llm=summarizer_llm,
                calibration=calibration,
                retrieval_port=retrieval_port,
                max_tool_turns=int(summarizer_cfg.get("max_tool_turns", 3)),
                is_embedding_enabled=is_embedding_enabled,
                loop_guardrails=_to_plain_dict(summarizer_cfg.get("loop_guardrails")),
                voice_clone_orchestrator=voice_clone_orchestrator,
                live_draft_audio_enabled=live_draft_audio_enabled,
                emo_preset_catalog={
                    str(name).strip(): str(emo_text).strip()
                    for name, emo_text in _to_plain_dict(
                        voice_clone_cfg_dict.get("emo_presets", {})
                    ).items()
                    if str(name).strip() and str(emo_text).strip()
                },
                agent_client=shared_agent_client,
                event_bus=event_bus,
            ),
        )
        _run_server_in_thread("summarizer-a2a", summarizer_app, summarizer_port)
    else:
        event_bus.publish("system_message", "Summarizer stage disabled by pipeline plan.")

    if stage_plan.run_critic_stage:
        if calibration is None and not live_draft_audio_enabled:
            raise RuntimeError(
                "Critic stage requires a loaded voice-clone calibration artifact when live drafting is disabled."
            )
        event_bus.publish(
            "system_message",
            f"Starting Critic A2A server on port {critic_port}...",
        )
        critic_app = _build_a2a_app(
            name="Critic",
            description="Evaluates summaries.",
            port=critic_port,
            executor=CriticExecutor(
                llm=critic_llm,
                calibration=calibration,
                max_tool_turns=int(critic_cfg.get("max_tool_turns", 5)),
                retrieval_port=retrieval_port,
                is_embedding_enabled=is_embedding_enabled,
                agent_client=shared_agent_client,
                event_bus=event_bus,
            ),
        )
        _run_server_in_thread("critic-a2a", critic_app, critic_port)
    else:
        event_bus.publish("system_message", "Critic stage disabled by pipeline plan.")

    servers_to_check: list[tuple[str, int]] = []
    if transcript_index is not None:
        servers_to_check.append(("InfoRetrieval", retrieval_port))
    if stage_plan.run_summarizer_stage:
        servers_to_check.append(("Summarizer", summarizer_port))
    if stage_plan.run_critic_stage:
        servers_to_check.append(("Critic", critic_port))

    checker = AgentHealthChecker(max_retries=5, base_delay=1.0)
    _wait_for_agent_servers(
        checker=checker,
        servers=servers_to_check,
        overall_timeout_seconds=10.0,
        poll_interval_seconds=0.2,
        request_timeout_seconds=1.0,
    )

    loop_memory_cfg = _to_plain_dict(orchestrator_cfg_dict.get("loop_memory", {}))
    loop_memory_artifact_path: Path | None = None
    loop_memory_artifact_value = str(loop_memory_cfg.get("artifact_path", "")).strip()
    if loop_memory_artifact_value:
        loop_memory_artifact_path = resolve_path(
            loop_memory_artifact_value,
            base_dir=project_root,
        )

    event_bus.publish("system_message", "Starting orchestration loop...")
    orchestrator = Orchestrator(
        retrieval_port=retrieval_port,
        summarizer_port=summarizer_port,
        critic_port=critic_port,
        timeouts=dict(cfg.orchestrator.get("timeouts", {})),
        agent_client=shared_agent_client,
        event_bus=event_bus,
    )
    stage_orchestrator = StageOrchestrator(
        orchestrator=orchestrator,
        stage_plan=stage_plan,
        project_root=project_root,
        target_seconds=target_seconds,
        duration_tolerance_ratio=duration_tolerance_ratio,
        max_iterations=int(orchestrator_cfg_dict.get("max_iterations", 60)),
        voice_clone_orchestrator=voice_clone_orchestrator,
        interjector_orchestrator=interjector_orchestrator,
        speaker_sample_preparer=speaker_sample_preparer,
        eta_strategy=audio_orchestrator.eta_strategy,
        eta_profile_path=eta_profile_path,
        eta_profile_context=eta_profile_context,
        eta_update_interval_seconds=audio_orchestrator.eta_update_interval_seconds,
        eta_progress_smoothing=audio_orchestrator.eta_progress_smoothing,
        eta_overrun_factor=audio_orchestrator.eta_overrun_factor,
        eta_headroom_seconds=audio_orchestrator.eta_headroom_seconds,
        voice_clone_gpu_heartbeat=voice_clone_gpu_heartbeat,
        loop_memory_artifact_path=loop_memory_artifact_path,
        live_draft_audio_enabled=live_draft_audio_enabled,
    )
    try:
        asyncio.run(
            stage_orchestrator.run(
                transcript=transcript,
                retrieval_enabled=transcript_index is not None,
            )
        )
    finally:
        audio_orchestrator._save_eta_profile(
            profile_path=eta_profile_path,
            context=eta_profile_context,
        )


if __name__ == "__main__":
    main()

