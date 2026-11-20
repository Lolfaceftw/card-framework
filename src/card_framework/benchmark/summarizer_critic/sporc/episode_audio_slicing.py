"""Build clipped episode-audio montages from Anthropic-selected transcript spans."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel

from card_framework.audio_pipeline.gateways.speaker_sample_gateway import (
    FfmpegSpeakerSampleExporter,
)
from card_framework.audio_pipeline.runtime import (
    ensure_command_available,
    probe_audio_duration_ms,
    utc_now_iso,
)
from card_framework.audio_pipeline.speaker_samples import AudioSlice
from card_framework.benchmark.summarizer_critic.sporc import llm_sampler
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_ARTIFACTS_ROOT,
    PreparedBenchmarkSample,
)
from card_framework.shared.env import load_repo_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_EPISODE_ASSETS_MANIFEST = (
    _SCRIPT_DIR / "sampled_segments" / "episode_assets" / "prepared_episodes_manifest.json"
).resolve()
_DEFAULT_OUTPUT_ROOT = (DEFAULT_ARTIFACTS_ROOT / "sporc_episode_audio_slices").resolve()
_DEFAULT_DURATION_PRESET = "10m"
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_OUTPUT_TOKENS = 16_000
_DEFAULT_MAX_TOOL_ROUNDS = 8
_DEFAULT_SAMPLE_RATE_HZ = 44_100
_DEFAULT_CHANNELS = 1
_DEFAULT_EDGE_FADE_MS = 20
_DEFAULT_MP3_BITRATE = "192k"
_DEFAULT_TIMEOUT_SECONDS = 900
_AUDIO_FORMAT_CHOICES = ("mp3", "wav")
_DURATION_PRESET_CHOICES = ("5m", "10m", "15m")
_MODEL_INPUT_OUTPUT_PRICING_USD_PER_MTOK = {
    # Snapshot from Anthropic's Models overview on 2026-03-21:
    # https://platform.claude.com/docs/en/about-claude/models/overview
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
}


@dataclass(slots=True, frozen=True)
class _SliceRunPaths:
    """Store resolved filesystem paths for one episode-audio slicing run."""

    run_root: Path
    summary_path: Path
    selected_samples_jsonl_path: Path
    selected_samples_csv_path: Path
    episodes_root: Path
    experiments_path: Path


@dataclass(slots=True, frozen=True)
class _RenderResult:
    """Describe one rendered episode-audio artifact."""

    output_path: Path
    output_format: str
    source_audio_path: Path
    slice_count: int
    selected_duration_ms: int
    rendered_duration_ms: int | None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for episode-audio slicing runs."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample full episode transcripts from the prepared SPoRC "
            "episode-assets manifest and cut one final audio montage per episode."
        )
    )
    parser.add_argument(
        "--dataset-manifest",
        default=str(_DEFAULT_EPISODE_ASSETS_MANIFEST),
        help=(
            "Episode-assets summary manifest. Defaults to the prepared local "
            "episode_assets/prepared_episodes_manifest.json file."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(_DEFAULT_OUTPUT_ROOT),
        help=(
            "Directory where run artifacts, selected samples, and sliced audio "
            "files will be written."
        ),
    )
    parser.add_argument(
        "--duration-preset",
        default=_DEFAULT_DURATION_PRESET,
        choices=_DURATION_PRESET_CHOICES,
        help="Target audio duration preset per episode.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help=(
            "How many manifest episodes to process. Use 0 to process every "
            "episode in the episode-assets summary."
        ),
    )
    parser.add_argument(
        "--select-episode",
        default="",
        help="Optional inclusive manifest range such as 2-4 or [2,4].",
    )
    parser.add_argument(
        "--provider",
        default=_DEFAULT_PROVIDER,
        choices=("anthropic", "vllm"),
        help="LLM provider used for transcript sampling.",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="Provider model name or alias.",
    )
    parser.add_argument(
        "--thinking-effort",
        default="medium",
        choices=sorted(llm_sampler._VALID_THINKING_EFFORTS),
        help="Anthropic thinking effort when thinking stays enabled.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Anthropic visible thinking to reduce token spend.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=_DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum model output tokens per tool round.",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=_DEFAULT_MAX_TOOL_ROUNDS,
        help="Maximum sampler tool rounds per episode.",
    )
    parser.add_argument(
        "--audio-format",
        default="mp3",
        choices=_AUDIO_FORMAT_CHOICES,
        help="Final rendered audio format.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=_DEFAULT_SAMPLE_RATE_HZ,
        help="Rendered audio sample rate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=_DEFAULT_CHANNELS,
        help="Rendered audio channel count.",
    )
    parser.add_argument(
        "--edge-fade-ms",
        type=int,
        default=_DEFAULT_EDGE_FADE_MS,
        help="Per-slice fade-in/out applied before concatenation.",
    )
    parser.add_argument(
        "--mp3-bitrate",
        default=_DEFAULT_MP3_BITRATE,
        help="MP3 bitrate used when --audio-format mp3.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="FFmpeg timeout used for render and transcode steps.",
    )
    parser.add_argument(
        "--max-estimated-cost-usd",
        type=float,
        default=0.0,
        help=(
            "Optional cumulative estimated-cost ceiling based on reported "
            "provider token usage. Use 0 to disable the guard."
        ),
    )
    parser.add_argument(
        "--words-per-minute",
        type=float,
        default=170.0,
        help="Forwarded sampler words-per-minute setting for metadata only.",
    )
    parser.add_argument(
        "--sample-factor",
        type=float,
        default=5.0,
        help="Forwarded sampler sample-factor setting for metadata only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the episode-audio slicing workflow from parsed CLI args."""
    args = build_parser().parse_args(argv)
    return run_episode_audio_slicing_session(args)


def run_episode_audio_slicing_session(
    args: argparse.Namespace,
    *,
    console: Console | None = None,
) -> int:
    """Run transcript sampling plus source-audio slicing for episode assets."""
    load_repo_dotenv(search_cwd=True)
    active_console = console if console is not None else Console()
    provider: Any | None = None
    try:
        settings = _resolve_settings(args)
        llm_sampler._validate_startup(settings=settings)
        prepared_samples = llm_sampler._load_episode_selection(
            manifest_path=settings["dataset_manifest"],
            episodes_requested=settings["episodes"],
            episode_range=settings["episode_range"],
        )
        run_paths = _create_run_paths(Path(settings["output_root"]))
        run_paths.run_root.mkdir(parents=True, exist_ok=True)
        run_paths.episodes_root.mkdir(parents=True, exist_ok=True)
        _initialize_experiments_log(run_paths=run_paths, settings=settings)

        provider = llm_sampler._create_llm_sampler_provider(settings=settings)
        settings["resolved_model"] = llm_sampler._resolve_provider_model_label(
            provider=provider,
            fallback_model=str(settings["model"]),
        )
        pricing = _resolve_pricing_for_model(str(settings["resolved_model"]))
        summary_payload = llm_sampler._build_initial_summary_payload(
            run_id=run_paths.run_root.name,
            settings=settings,
            prepared_samples=prepared_samples,
        )
        summary_payload.update(
            {
                "workflow": "episode_audio_slicing",
                "output_root": str(run_paths.run_root),
                "audio_format": str(settings["audio_format"]),
                "sample_rate_hz": int(settings["sample_rate_hz"]),
                "channels": int(settings["channels"]),
                "edge_fade_ms": int(settings["edge_fade_ms"]),
                "mp3_bitrate": str(settings["mp3_bitrate"]),
                "experiments_path": str(run_paths.experiments_path),
                "budget_polling_mode": (
                    "estimated_usage_only"
                    if settings["provider"] == "anthropic"
                    else "not_applicable"
                ),
                "anthropic_balance_polling_supported": False,
                "anthropic_balance_polling_note": (
                    "Anthropic does not expose remaining prepaid credits through "
                    "the standard API; this run tracks estimated spend from "
                    "reported token usage instead."
                    if settings["provider"] == "anthropic"
                    else ""
                ),
                "max_estimated_cost_usd": (
                    float(settings["max_estimated_cost_usd"])
                    if settings["max_estimated_cost_usd"] is not None
                    else None
                ),
                "estimated_cost_pricing": pricing,
                "estimated_cost_usd": 0.0,
                "usage_totals": {},
                "rendered_episode_count": 0,
                "rendered_audio_count": 0,
                "rendered_duration_ms_total": 0,
            }
        )
        selected_rows: list[dict[str, Any]] = []
        llm_sampler._write_run_exports(
            run_paths=run_paths,
            summary_payload=summary_payload,
            selected_rows=selected_rows,
        )
        _render_run_banner(
            console=active_console,
            run_paths=run_paths,
            settings=settings,
            episode_count=len(prepared_samples),
        )

        exit_code = 0
        budget_exhausted = False
        for episode_order, sample in enumerate(prepared_samples, start=1):
            manifest_episode_number = llm_sampler._resolve_manifest_episode_number(
                settings=settings,
                episode_order=episode_order,
            )
            episode_result = llm_sampler._run_episode_sampling(
                provider=provider,
                sample=sample,
                episode_order=episode_order,
                manifest_episode_number=manifest_episode_number,
                settings=settings,
                run_paths=run_paths,
                console=active_console,
            )
            episode_summary = dict(episode_result["summary"])
            response_path = Path(str(episode_summary["response_path"])).resolve()
            usage_totals = _extract_usage_totals_from_response(response_path)
            estimated_cost_usd = _estimate_cost_usd(
                usage_totals=usage_totals,
                pricing=pricing,
            )
            if episode_result["selected_rows"]:
                selected_rows.extend(episode_result["selected_rows"])
            episode_summary["usage_totals"] = usage_totals
            episode_summary["estimated_cost_usd"] = estimated_cost_usd
            episode_summary["rendered_audio_path"] = ""
            episode_summary["rendered_audio_duration_ms"] = 0
            episode_summary["rendered_audio_duration_seconds"] = 0.0
            episode_summary["source_audio_path"] = ""

            if episode_summary["status"] == "completed" and episode_result["selected_rows"]:
                episode_dir = (run_paths.episodes_root / sample.sample_id).resolve()
                try:
                    source_audio_path = _resolve_source_audio_path(
                        sample=sample,
                        episode_dir=episode_dir,
                    )
                    render_result = _render_episode_audio(
                        output_dir=episode_dir,
                        sample=sample,
                        selected_rows=episode_result["selected_rows"],
                        source_audio_path=source_audio_path,
                        audio_format=str(settings["audio_format"]),
                        sample_rate_hz=int(settings["sample_rate_hz"]),
                        channels=int(settings["channels"]),
                        edge_fade_ms=int(settings["edge_fade_ms"]),
                        mp3_bitrate=str(settings["mp3_bitrate"]),
                        timeout_seconds=int(settings["timeout_seconds"]),
                    )
                except Exception as exc:
                    exit_code = 1
                    episode_summary["status"] = "failed_render"
                    episode_summary["render_error"] = str(exc)
                    _append_experiment_note(
                        run_paths=run_paths,
                        heading=f"Episode {episode_order}: {sample.sample_id}",
                        lines=[
                            "- Status: failed_render",
                            f"- Render error: {exc}",
                            f"- Estimated cost (USD): {estimated_cost_usd:.6f}",
                        ],
                    )
                else:
                    episode_summary["rendered_audio_path"] = str(render_result.output_path)
                    episode_summary["rendered_audio_duration_ms"] = int(
                        render_result.rendered_duration_ms or 0
                    )
                    episode_summary["rendered_audio_duration_seconds"] = round(
                        (render_result.rendered_duration_ms or 0) / 1000.0,
                        3,
                    )
                    episode_summary["source_audio_path"] = str(render_result.source_audio_path)
                    episode_summary["rendered_audio_format"] = render_result.output_format
                    episode_summary["rendered_audio_slice_count"] = render_result.slice_count
                    summary_payload["rendered_episode_count"] = int(
                        summary_payload["rendered_episode_count"]
                    ) + 1
                    summary_payload["rendered_audio_count"] = int(
                        summary_payload["rendered_audio_count"]
                    ) + 1
                    summary_payload["rendered_duration_ms_total"] = int(
                        summary_payload["rendered_duration_ms_total"]
                    ) + int(render_result.rendered_duration_ms or 0)
                    _append_experiment_note(
                        run_paths=run_paths,
                        heading=f"Episode {episode_order}: {sample.sample_id}",
                        lines=[
                            "- Status: rendered",
                            f"- Source audio: {render_result.source_audio_path}",
                            f"- Output audio: {render_result.output_path}",
                            f"- Selected duration (ms): {render_result.selected_duration_ms}",
                            f"- Rendered duration (ms): {render_result.rendered_duration_ms or 0}",
                            f"- Slice count: {render_result.slice_count}",
                            f"- Estimated cost (USD): {estimated_cost_usd:.6f}",
                        ],
                    )
            else:
                exit_code = 1
                _append_experiment_note(
                    run_paths=run_paths,
                    heading=f"Episode {episode_order}: {sample.sample_id}",
                    lines=[
                        f"- Status: {episode_summary['status']}",
                        "- Output audio: not rendered",
                        f"- Estimated cost (USD): {estimated_cost_usd:.6f}",
                    ],
                )

            summary_payload["episodes"].append(episode_summary)
            summary_payload["updated_at_utc"] = utc_now_iso()
            summary_payload["selected_sample_count"] = len(selected_rows)
            summary_payload["completed_episode_count"] = sum(
                1
                for episode in summary_payload["episodes"]
                if episode["status"] == "completed"
            )
            summary_payload["usage_totals"] = _merge_usage_totals(
                base=summary_payload.get("usage_totals", {}),
                delta=usage_totals,
            )
            summary_payload["estimated_cost_usd"] = round(
                float(summary_payload.get("estimated_cost_usd") or 0.0)
                + estimated_cost_usd,
                6,
            )
            budget_limit = settings["max_estimated_cost_usd"]
            if budget_limit is not None and summary_payload["estimated_cost_usd"] >= budget_limit:
                budget_exhausted = True
                exit_code = 1
                summary_payload["status"] = "stopped_budget_guard"
                _append_experiment_note(
                    run_paths=run_paths,
                    heading="Budget Guard",
                    lines=[
                        f"- Estimated cumulative cost reached {summary_payload['estimated_cost_usd']:.6f} USD.",
                        f"- Configured ceiling: {budget_limit:.6f} USD.",
                        "- Remaining episodes were skipped to avoid overspending.",
                    ],
                )
                llm_sampler._write_run_exports(
                    run_paths=run_paths,
                    summary_payload=summary_payload,
                    selected_rows=selected_rows,
                )
                break

            summary_payload["status"] = (
                "completed"
                if exit_code == 0
                and len(summary_payload["episodes"]) == len(prepared_samples)
                else "completed_with_failures"
            )
            llm_sampler._write_run_exports(
                run_paths=run_paths,
                summary_payload=summary_payload,
                selected_rows=selected_rows,
            )

        if budget_exhausted:
            _render_budget_guard_banner(active_console, summary_payload=summary_payload)
        else:
            _render_completion_banner(active_console, summary_payload=summary_payload)
        return exit_code
    except KeyboardInterrupt:
        active_console.print(
            Panel.fit(
                "Episode audio slicing cancelled.",
                title="Cancelled",
                border_style="yellow",
            )
        )
        return 130
    except Exception as exc:
        active_console.print(
            Panel.fit(
                str(exc),
                title="Episode Audio Slicing Failed",
                border_style="red",
            )
        )
        return 1
    finally:
        closeable_client = getattr(provider, "_client", None)
        close_fn = getattr(closeable_client, "close", None)
        if callable(close_fn):
            close_fn()


def _resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve audio-slicing settings and forward sampler-compatible fields."""
    manifest_path = Path(str(getattr(args, "dataset_manifest", "") or "")).expanduser().resolve()
    if not manifest_path.exists():
        raise RuntimeError(
            "Episode-audio slicing requires an existing episode-assets manifest. "
            f"Expected: {manifest_path}"
        )
    if not llm_sampler._manifest_uses_episode_assets_summary(manifest_path):
        raise RuntimeError(
            "Episode-audio slicing requires "
            "`episode_assets/prepared_episodes_manifest.json`-style input."
        )
    requested_episodes = int(getattr(args, "episodes", 0) or 0)
    if requested_episodes <= 0 and not str(getattr(args, "select_episode", "") or "").strip():
        payload = llm_sampler._load_manifest_payload(manifest_path)
        raw_episodes = payload.get("episodes")
        requested_episodes = len(raw_episodes) if isinstance(raw_episodes, list) else 0
    forwarded_args = argparse.Namespace(
        dataset_manifest=str(manifest_path),
        provider=str(getattr(args, "provider", _DEFAULT_PROVIDER) or _DEFAULT_PROVIDER),
        sampling_root=str(Path(getattr(args, "output_root", _DEFAULT_OUTPUT_ROOT)).expanduser().resolve()),
        duration_preset=str(getattr(args, "duration_preset", _DEFAULT_DURATION_PRESET) or _DEFAULT_DURATION_PRESET),
        samples=0,
        text_only=False,
        transcribe_enabled=False,
        audio_config="",
        audio_device="",
        episodes=requested_episodes,
        select_episode=str(getattr(args, "select_episode", "") or ""),
        words_per_minute=float(getattr(args, "words_per_minute", 170.0) or 170.0),
        sample_factor=float(getattr(args, "sample_factor", 5.0) or 5.0),
        model=str(getattr(args, "model", _DEFAULT_MODEL) or _DEFAULT_MODEL),
        thinking_effort=str(getattr(args, "thinking_effort", "medium") or "medium"),
        disable_thinking=bool(getattr(args, "disable_thinking", False)),
        max_output_tokens=int(
            getattr(args, "max_output_tokens", _DEFAULT_MAX_OUTPUT_TOKENS)
            or _DEFAULT_MAX_OUTPUT_TOKENS
        ),
        max_tool_rounds=int(
            getattr(args, "max_tool_rounds", _DEFAULT_MAX_TOOL_ROUNDS)
            or _DEFAULT_MAX_TOOL_ROUNDS
        ),
    )
    settings = llm_sampler._resolve_settings(forwarded_args)
    settings["output_root"] = str(
        Path(str(getattr(args, "output_root", _DEFAULT_OUTPUT_ROOT))).expanduser().resolve()
    )
    settings["audio_format"] = str(getattr(args, "audio_format", "mp3") or "mp3").strip().lower()
    settings["sample_rate_hz"] = int(
        getattr(args, "sample_rate_hz", _DEFAULT_SAMPLE_RATE_HZ) or _DEFAULT_SAMPLE_RATE_HZ
    )
    settings["channels"] = int(getattr(args, "channels", _DEFAULT_CHANNELS) or _DEFAULT_CHANNELS)
    settings["edge_fade_ms"] = int(
        getattr(args, "edge_fade_ms", _DEFAULT_EDGE_FADE_MS) or _DEFAULT_EDGE_FADE_MS
    )
    settings["mp3_bitrate"] = str(getattr(args, "mp3_bitrate", _DEFAULT_MP3_BITRATE) or _DEFAULT_MP3_BITRATE)
    settings["timeout_seconds"] = int(
        getattr(args, "timeout_seconds", _DEFAULT_TIMEOUT_SECONDS) or _DEFAULT_TIMEOUT_SECONDS
    )
    max_estimated_cost_usd = float(
        getattr(args, "max_estimated_cost_usd", 0.0) or 0.0
    )
    settings["max_estimated_cost_usd"] = (
        max_estimated_cost_usd if max_estimated_cost_usd > 0 else None
    )
    if settings["audio_format"] not in _AUDIO_FORMAT_CHOICES:
        raise RuntimeError(
            "Audio format must be one of: " + ", ".join(_AUDIO_FORMAT_CHOICES)
        )
    if settings["sample_rate_hz"] <= 0:
        raise RuntimeError("Sample rate must be greater than zero.")
    if settings["channels"] <= 0:
        raise RuntimeError("Channel count must be greater than zero.")
    if settings["edge_fade_ms"] < 0:
        raise RuntimeError("Edge fade must be zero or greater.")
    if settings["timeout_seconds"] <= 0:
        raise RuntimeError("Timeout seconds must be greater than zero.")
    return settings


def _create_run_paths(output_root: Path) -> _SliceRunPaths:
    """Create resolved artifact paths for one episode-audio slicing run."""
    run_id = (
        datetime.now(timezone.utc).strftime("episode_audio_slice_%Y%m%dT%H%M%SZ_")
        + uuid4().hex[:8]
    )
    run_root = (output_root / run_id).resolve()
    return _SliceRunPaths(
        run_root=run_root,
        summary_path=(run_root / "run_summary.json").resolve(),
        selected_samples_jsonl_path=(run_root / "selected_samples.jsonl").resolve(),
        selected_samples_csv_path=(run_root / "selected_samples.csv").resolve(),
        episodes_root=(run_root / "episodes").resolve(),
        experiments_path=(run_root / "experiments.md").resolve(),
    )


def _render_run_banner(
    *,
    console: Console,
    run_paths: _SliceRunPaths,
    settings: dict[str, Any],
    episode_count: int,
) -> None:
    """Render the opening banner for an episode-audio slicing run."""
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Run ID: {run_paths.run_root.name}",
                    f"Episode-assets manifest: {settings['dataset_manifest']}",
                    f"Episodes: {episode_count}",
                    f"Target duration: {settings['duration_preset']} ({settings['preset_seconds']}s)",
                    f"Provider: {settings['provider']}",
                    f"Model: {settings.get('resolved_model') or settings['model']}",
                    f"Audio format: {settings['audio_format']}",
                    f"Run root: {run_paths.run_root}",
                    f"Experiments log: {run_paths.experiments_path}",
                ]
            ),
            title="Episode Audio Slicing",
            border_style="cyan",
        )
    )


def _initialize_experiments_log(
    *,
    run_paths: _SliceRunPaths,
    settings: dict[str, Any],
) -> None:
    """Create the markdown experiment log for this run."""
    run_paths.experiments_path.write_text(
        "\n".join(
            [
                "# Episode Audio Slicing Experiments",
                "",
                f"- Run ID: `{run_paths.run_root.name}`",
                f"- Created At (UTC): `{utc_now_iso()}`",
                f"- Manifest: `{settings['dataset_manifest']}`",
                f"- Duration Preset: `{settings['duration_preset']}`",
                f"- Provider: `{settings['provider']}`",
                f"- Model: `{settings['model']}`",
                f"- Output Root: `{run_paths.run_root}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _append_experiment_note(
    *,
    run_paths: _SliceRunPaths,
    heading: str,
    lines: list[str],
) -> None:
    """Append one markdown note into the experiment log."""
    with run_paths.experiments_path.open("a", encoding="utf-8") as handle:
        handle.write(f"## {heading}\n\n")
        for line in lines:
            handle.write(f"{line}\n")
        handle.write("\n")


def _resolve_source_audio_path(
    *,
    sample: PreparedBenchmarkSample,
    episode_dir: Path,
) -> Path:
    """Resolve or download the source audio needed for final slicing."""
    _, source_audio_path = llm_sampler._ensure_local_sampler_source_audio(
        sample=sample,
        episode_dir=episode_dir,
    )
    return source_audio_path.resolve()


def _render_episode_audio(
    *,
    output_dir: Path,
    sample: PreparedBenchmarkSample,
    selected_rows: list[dict[str, Any]],
    source_audio_path: Path,
    audio_format: str,
    sample_rate_hz: int,
    channels: int,
    edge_fade_ms: int,
    mp3_bitrate: str,
    timeout_seconds: int,
) -> _RenderResult:
    """Render one final clipped-audio montage from selected transcript spans."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slices = _selected_rows_to_audio_slices(selected_rows)
    output_path = (output_dir / f"{sample.sample_id}_selected_audio.{audio_format}").resolve()
    selected_duration_ms = sum(audio_slice.duration_ms for audio_slice in slices)
    exporter = FfmpegSpeakerSampleExporter(timeout_seconds=timeout_seconds)
    if audio_format == "wav":
        exporter.export(
            source_audio_path=source_audio_path,
            slices=slices,
            output_path=output_path,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            edge_fade_ms=edge_fade_ms,
        )
    else:
        temp_wav_path = output_path.with_suffix(".tmp.wav")
        exporter.export(
            source_audio_path=source_audio_path,
            slices=slices,
            output_path=temp_wav_path,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            edge_fade_ms=edge_fade_ms,
        )
        try:
            _transcode_wav_to_mp3(
                input_path=temp_wav_path,
                output_path=output_path,
                bitrate=mp3_bitrate,
                timeout_seconds=timeout_seconds,
            )
        finally:
            temp_wav_path.unlink(missing_ok=True)
    rendered_duration_ms = probe_audio_duration_ms(output_path)
    _write_episode_audio_manifest(
        output_dir=output_dir,
        source_audio_path=source_audio_path,
        output_path=output_path,
        selected_rows=selected_rows,
        selected_duration_ms=selected_duration_ms,
        rendered_duration_ms=rendered_duration_ms,
    )
    return _RenderResult(
        output_path=output_path,
        output_format=audio_format,
        source_audio_path=source_audio_path,
        slice_count=len(slices),
        selected_duration_ms=selected_duration_ms,
        rendered_duration_ms=rendered_duration_ms,
    )


def _selected_rows_to_audio_slices(
    selected_rows: list[dict[str, Any]],
) -> tuple[AudioSlice, ...]:
    """Convert selected-row export payloads into ordered audio slices."""
    ordered_rows = sorted(
        selected_rows,
        key=lambda item: int(item.get("selection_order") or 0),
    )
    slices: list[AudioSlice] = []
    for row in ordered_rows:
        start_time = row.get("start_time")
        end_time = row.get("end_time")
        if start_time is None or end_time is None:
            raise RuntimeError(
                "Selected rows must include measured start_time and end_time "
                "before audio slicing can run."
            )
        audio_slice = AudioSlice(
            start_time_ms=int(start_time),
            end_time_ms=int(end_time),
        )
        if audio_slice.duration_ms <= 0:
            raise RuntimeError(
                "Selected rows produced a zero-length or negative audio slice."
            )
        slices.append(audio_slice)
    if not slices:
        raise RuntimeError("Audio slicing requires at least one selected row.")
    return tuple(slices)


def _transcode_wav_to_mp3(
    *,
    input_path: Path,
    output_path: Path,
    bitrate: str,
    timeout_seconds: int,
) -> None:
    """Transcode one intermediate WAV montage into the final MP3 artifact."""
    ensure_command_available("ffmpeg")
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_output_path.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-vn",
        "-f",
        "mp3",
        str(temp_output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        temp_output_path.replace(output_path)
    except subprocess.TimeoutExpired as exc:
        temp_output_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Failed to transcode selected audio to MP3 due to timeout. "
            f"Command: {' '.join(command)}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        temp_output_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Failed to transcode selected audio to MP3 with ffmpeg. "
            f"Command: {' '.join(command)}. "
            f"Stderr: {(exc.stderr or '').strip()[:500]}"
        ) from exc


def _write_episode_audio_manifest(
    *,
    output_dir: Path,
    source_audio_path: Path,
    output_path: Path,
    selected_rows: list[dict[str, Any]],
    selected_duration_ms: int,
    rendered_duration_ms: int | None,
) -> None:
    """Persist one render sidecar with the selection and output metadata."""
    manifest_path = (output_dir / "selected_audio_manifest.json").resolve()
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": utc_now_iso(),
                "source_audio_path": str(source_audio_path),
                "output_audio_path": str(output_path),
                "selected_duration_ms": selected_duration_ms,
                "selected_duration_seconds": round(selected_duration_ms / 1000.0, 3),
                "rendered_duration_ms": int(rendered_duration_ms or 0),
                "rendered_duration_seconds": round(
                    (rendered_duration_ms or 0) / 1000.0,
                    3,
                ),
                "slice_count": len(selected_rows),
                "selected_rows": selected_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _extract_usage_totals_from_response(response_path: Path) -> dict[str, int]:
    """Aggregate Anthropic usage fields from one episode response artifact."""
    if not response_path.exists():
        return {}
    response_payload = json.loads(response_path.read_text(encoding="utf-8"))
    attempts = response_payload.get("attempts")
    if not isinstance(attempts, list):
        return {}
    totals: dict[str, int] = {}
    for attempt in attempts:
        attempt_payload = attempt if isinstance(attempt, dict) else {}
        provider_trace = attempt_payload.get("provider_trace")
        provider_payload = provider_trace if isinstance(provider_trace, dict) else {}
        response_payload_item = provider_payload.get("response")
        response_item = (
            response_payload_item
            if isinstance(response_payload_item, dict)
            else {}
        )
        usage_payload = response_item.get("usage")
        usage = usage_payload if isinstance(usage_payload, dict) else {}
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def _resolve_pricing_for_model(model_name: str) -> dict[str, float] | None:
    """Return the best-effort pricing snapshot for one supported Anthropic model."""
    normalized = model_name.strip().lower()
    return _MODEL_INPUT_OUTPUT_PRICING_USD_PER_MTOK.get(normalized)


def _estimate_cost_usd(
    *,
    usage_totals: dict[str, int],
    pricing: dict[str, float] | None,
) -> float:
    """Estimate provider cost from reported usage and the active pricing snapshot."""
    if not pricing:
        return 0.0
    input_tokens = int(usage_totals.get("input_tokens") or 0)
    output_tokens = int(usage_totals.get("output_tokens") or 0)
    estimated_cost = (
        (input_tokens / 1_000_000.0) * float(pricing["input"])
        + (output_tokens / 1_000_000.0) * float(pricing["output"])
    )
    return round(estimated_cost, 6)


def _merge_usage_totals(
    *,
    base: dict[str, Any],
    delta: dict[str, int],
) -> dict[str, int]:
    """Merge one episode's usage totals into the cumulative run totals."""
    merged = {
        str(key): int(value)
        for key, value in base.items()
        if isinstance(value, int)
    }
    for key, value in delta.items():
        merged[str(key)] = merged.get(str(key), 0) + int(value)
    return merged


def _render_completion_banner(
    console: Console,
    *,
    summary_payload: dict[str, Any],
) -> None:
    """Render the successful completion banner."""
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Status: {summary_payload['status']}",
                    f"Rendered audio count: {summary_payload['rendered_audio_count']}",
                    f"Estimated cost (USD): {summary_payload['estimated_cost_usd']:.6f}",
                    f"Run root: {summary_payload['output_root']}",
                ]
            ),
            title="Episode Audio Slicing Completed",
            border_style="green",
        )
    )


def _render_budget_guard_banner(
    console: Console,
    *,
    summary_payload: dict[str, Any],
) -> None:
    """Render the budget-guard stop banner."""
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Status: {summary_payload['status']}",
                    f"Estimated cost (USD): {summary_payload['estimated_cost_usd']:.6f}",
                    "The run stopped because the configured estimated-cost guard was reached.",
                    f"Run root: {summary_payload['output_root']}",
                ]
            ),
            title="Budget Guard Triggered",
            border_style="yellow",
        )
    )
