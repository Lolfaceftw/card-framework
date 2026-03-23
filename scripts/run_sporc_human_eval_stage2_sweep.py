"""Prepare cleaned SPoRC stage-2 inputs and run the Docker GPU sweep."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any
from uuid import uuid4

from card_framework.audio_pipeline.io import write_transcript_atomic
from card_framework.orchestration.transcript import Transcript, TranscriptSegment
from card_framework.shared.paths import REPO_ROOT
from card_framework.shared.summary_xml import parse_summary_xml

CONTAINER_REPO_ROOT = PurePosixPath("/workspace/card-framework")
DEFAULT_PACKAGE_ROOT = (REPO_ROOT / "sporc_human_eval_assets_10m").resolve()
DEFAULT_NOTES_PATH = (
    REPO_ROOT / "artifacts" / "run_notes" / "2026-03-21_sporc_gpu_gemini_sweep.md"
).resolve()
DEFAULT_OUTPUT_PARENT = (
    REPO_ROOT / "artifacts" / "sporc_human_eval_stage2_sweeps"
).resolve()
DEFAULT_DURATION_PRESETS = ("30s", "1m", "5m")
DEFAULT_LLM_PROFILE = "google_gemini"
DEFAULT_LLM_MODEL = "gemini-3.1-pro-preview"

SOURCE_PATTERN = re.compile(
    r"^Source:\s*(\d+)-(\d+)\s*\((\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2})\)\s*\|\s*([^|]+?)\s*\|\s*measured\s*([0-9.]+)s\s*$"
)
SECTION_SPLIT_PATTERN = re.compile(r"\n\s*\n(?=\d+\.\s)")
UTTERANCE_LINE_PATTERN = re.compile(r"^(SPEAKER_[A-Za-z0-9]+):\s*(.*)$")
LEADING_BRACKET_SPEAKER_PATTERN = re.compile(r"^\[(SPEAKER_[A-Za-z0-9]+)\]\s*(.*)$")
INVALID_FILENAME_PATTERN = re.compile(r'[<>:"/\\|?*]+')
CONSOLE_SAFE_TRANSLATIONS = str.maketrans(
    {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
    }
)

MOJIBAKE_REPLACEMENTS = {
    "\u00a0": " ",
    "â€”": "-",
    "â€“": "-",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€¦": "...",
    "â€": '"',
    "Â": "",
}


@dataclass(slots=True, frozen=True)
class DurationPreset:
    """One summary target preset."""

    label: str
    target_seconds: int


@dataclass(slots=True, frozen=True)
class ParsedTranscriptBlock:
    """One cleaned block from the transcript-slice text file."""

    index: int
    source_range_start: int
    source_range_end: int
    source_start_ms: int
    source_end_ms: int
    source_mode: str
    measured_seconds: float
    rationale: str
    utterances: tuple[tuple[str, str], ...]


@dataclass(slots=True, frozen=True)
class ParsedEpisode:
    """Prepared episode data ready for per-preset materialization."""

    order: int
    title: str
    sample_id: str
    audio_slice_path: Path
    transcript_blocks: tuple[ParsedTranscriptBlock, ...]
    speakers: tuple[str, ...]
    voice_samples_dir: Path
    original_voice_manifest: dict[str, Any]
    bootstrap_mapping: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Skip stage-1, clean SPoRC human-eval transcript slices into stage-2 "
            "transcript JSON inputs, rewrite the provided voice samples into "
            "transcript-linked manifests, and run setup_and_run through the "
            "Docker GPU profile for 30s/1m/5m outputs."
        )
    )
    parser.add_argument(
        "--package-root",
        default=str(DEFAULT_PACKAGE_ROOT),
        help="Root of the sporc_human_eval_assets_10m package.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Sweep output root. Defaults to artifacts/sporc_human_eval_stage2_sweeps/"
            "<run-id>."
        ),
    )
    parser.add_argument(
        "--notes-path",
        default=str(DEFAULT_NOTES_PATH),
        help="Markdown notes file to append major events to.",
    )
    parser.add_argument(
        "--duration-presets",
        default=",".join(DEFAULT_DURATION_PRESETS),
        help="Comma-separated preset labels. Supported: 30s, 1m, 5m.",
    )
    parser.add_argument(
        "--episode-indices",
        default="",
        help="Optional comma-separated 1-based episode indices, for example 1,3,5.",
    )
    parser.add_argument(
        "--llm-profile",
        default=DEFAULT_LLM_PROFILE,
        help=(
            "Hydra llm profile id to use for summarizer, critic, and interjector, "
            "for example google_gemini or deepseek_chat."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Model id to pin on the selected llm profile.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare cleaned stage-2 inputs only and skip Docker runs.",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip the one-time Docker setup_only warm-up step.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later runs after one preset fails.",
    )
    parser.add_argument(
        "--orchestrator-max-iterations",
        type=int,
        default=0,
        help=(
            "Optional override for orchestrator.max_iterations. Values <= 0 keep "
            "the repo default."
        ),
    )
    parser.add_argument(
        "--summarizer-max-tool-turns",
        type=int,
        default=0,
        help=(
            "Optional override for agents.summarizer.max_tool_turns. Values <= 0 "
            "keep the repo default."
        ),
    )
    parser.add_argument(
        "--critic-max-tool-turns",
        type=int,
        default=0,
        help=(
            "Optional override for agents.critic.max_tool_turns. Values <= 0 keep "
            "the repo default."
        ),
    )
    parser.add_argument(
        "--disable-interjector-tool-call-planning",
        action="store_true",
        help=(
            "Use the single-response interjector planner instead of one tool-call "
            "Gemini chat per eligible host turn."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the host-side stage-2 sweep."""
    args = build_parser().parse_args(argv)
    package_root = Path(str(args.package_root)).expanduser().resolve()
    notes_path = Path(str(args.notes_path)).expanduser().resolve()
    output_root = _resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    presets = _parse_duration_presets(args.duration_presets)
    selected_indices = _parse_episode_indices(args.episode_indices)
    package_summary = _load_package_summary(package_root)
    episodes = _load_prepared_episodes(
        package_root=package_root,
        package_summary=package_summary,
        selected_indices=selected_indices,
    )

    _append_note(
        notes_path,
        "Stage-2 human-eval sweep driver started. Stage-1 remains skipped. "
        f"Output root: `{output_root}`. Presets: {', '.join(preset.label for preset in presets)}.",
    )

    logs_root = (output_root / "logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)
    final_exports_root = (output_root / "final_exports").resolve()
    final_exports_root.mkdir(parents=True, exist_ok=True)

    summary_payload: dict[str, Any] = {
        "run_id": output_root.name,
        "created_at_utc": _utc_now_iso(),
        "package_root": str(package_root),
        "notes_path": str(notes_path),
        "duration_presets": [preset.label for preset in presets],
        "llm_profile": str(args.llm_profile),
        "llm_model": str(args.llm_model),
        "prepare_only": bool(args.prepare_only),
        "runs": [],
    }
    summary_path = (output_root / "run_summary.json").resolve()
    _write_json(summary_payload, summary_path)

    if not args.prepare_only and not args.skip_setup:
        setup_log_path = (logs_root / "setup_only.log").resolve()
        _append_note(
            notes_path,
            "Running one Docker GPU `setup_and_run --setup-only` warm-up before the "
            "stage-2 sweep.",
        )
        setup_code = _run_setup_only(
            llm_profile=str(args.llm_profile),
            llm_model=str(args.llm_model),
            log_path=setup_log_path,
        )
        if setup_code != 0:
            _append_note(
                notes_path,
                "Docker setup-only warm-up failed. See "
                f"`{setup_log_path}`.",
            )
            return setup_code

    exit_code = 0
    for episode in episodes:
        episode_label = _machine_episode_label(episode.order, episode.title)
        for preset in presets:
            preset_root = (
                output_root / "runs" / episode_label / preset.label
            ).resolve()
            preset_root.mkdir(parents=True, exist_ok=True)
            transcript_path = _materialize_preset_inputs(
                preset_root=preset_root,
                episode=episode,
            )
            run_record: dict[str, Any] = {
                "episode_index": episode.order,
                "episode_title": episode.title,
                "sample_id": episode.sample_id,
                "duration_preset": preset.label,
                "target_seconds": preset.target_seconds,
                "output_root": str(preset_root),
                "transcript_path": str(transcript_path),
                "status": "prepared" if args.prepare_only else "pending",
            }
            summary_payload["runs"].append(run_record)
            _write_json(summary_payload, summary_path)

            if args.prepare_only:
                continue

            log_path = (
                logs_root / f"{episode.order:02d}_{_slugify(episode.title)}_{preset.label}.log"
            ).resolve()
            _append_note(
                notes_path,
                "Launching Docker GPU stage-2 run for "
                f"`{episode.title}` at `{preset.label}`. "
                f"Output root: `{preset_root}`.",
            )
            command = _build_run_command(
                output_root=preset_root,
                transcript_path=transcript_path,
                target_seconds=preset.target_seconds,
                llm_profile=str(args.llm_profile),
                llm_model=str(args.llm_model),
                orchestrator_max_iterations=int(args.orchestrator_max_iterations),
                summarizer_max_tool_turns=int(args.summarizer_max_tool_turns),
                critic_max_tool_turns=int(args.critic_max_tool_turns),
                disable_interjector_tool_call_planning=bool(
                    args.disable_interjector_tool_call_planning
                ),
            )
            return_code = _run_with_tee(command=command, log_path=log_path)
            run_record["log_path"] = str(log_path)
            run_record["return_code"] = return_code

            if return_code != 0:
                run_record["status"] = "failed"
                exit_code = 1
                _append_note(
                    notes_path,
                    "Docker GPU stage-2 run failed for "
                    f"`{episode.title}` at `{preset.label}`. "
                    f"Log: `{log_path}`.",
                )
                _write_json(summary_payload, summary_path)
                if not args.continue_on_error:
                    return exit_code
                continue

            run_record["status"] = "completed"
            run_record["artifacts"] = _collect_artifact_paths(preset_root)
            export_paths = _export_human_readable_artifacts(
                episode=episode,
                preset=preset,
                preset_root=preset_root,
                final_exports_root=final_exports_root,
            )
            run_record["human_readable_exports"] = export_paths
            review_payload = _observe_completed_output(preset_root)
            run_record["llm_output_review"] = review_payload
            _append_note(
                notes_path,
                "Completed Docker GPU stage-2 run for "
                f"`{episode.title}` at `{preset.label}`. "
                f"Artifacts exported with human-readable names under `{final_exports_root}`.",
            )
            _append_note(
                notes_path,
                "Observed LLM output for "
                f"`{episode.title}` at `{preset.label}`: "
                + _render_output_review_note(review_payload),
            )
            _write_json(summary_payload, summary_path)

    _append_note(
        notes_path,
        "Stage-2 human-eval sweep driver finished with "
        f"exit_code={exit_code}. Summary: `{summary_path}`.",
    )
    return exit_code


def _resolve_output_root(raw_value: str) -> Path:
    if raw_value.strip():
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        return candidate.resolve()
    run_id = (
        "sporc_human_eval_stage2_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        + uuid4().hex[:8]
    )
    return (DEFAULT_OUTPUT_PARENT / run_id).resolve()


def _parse_duration_presets(raw_value: str) -> tuple[DurationPreset, ...]:
    mapping = {
        "30s": DurationPreset(label="30s", target_seconds=30),
        "1m": DurationPreset(label="1m", target_seconds=60),
        "5m": DurationPreset(label="5m", target_seconds=300),
    }
    labels = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one duration preset is required.")
    unsupported = [label for label in labels if label not in mapping]
    if unsupported:
        raise ValueError(
            "Unsupported duration preset(s): " + ", ".join(sorted(set(unsupported)))
        )
    return tuple(mapping[label] for label in labels)


def _parse_episode_indices(raw_value: str) -> set[int] | None:
    if not raw_value.strip():
        return None
    values: set[int] = set()
    for token in raw_value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.add(int(stripped))
    if not values:
        return None
    return values


def _load_package_summary(package_root: Path) -> dict[str, Any]:
    summary_path = (package_root / "package_summary.json").resolve()
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_prepared_episodes(
    *,
    package_root: Path,
    package_summary: dict[str, Any],
    selected_indices: set[int] | None,
) -> tuple[ParsedEpisode, ...]:
    raw_episodes = package_summary.get("episodes", [])
    if not isinstance(raw_episodes, list):
        raise RuntimeError("package_summary.json does not contain an episodes list.")

    prepared: list[ParsedEpisode] = []
    for payload in raw_episodes:
        if not isinstance(payload, dict):
            continue
        order = int(payload.get("order") or 0)
        if order <= 0:
            continue
        if selected_indices is not None and order not in selected_indices:
            continue

        title = str(payload.get("title") or "").strip()
        sample_id = str(payload.get("sample_id") or "").strip()
        transcript_slice_path = _resolve_summary_relative_path(
            package_root=package_root,
            raw_value=str(payload.get("transcript_slice") or ""),
        )
        voice_samples_dir = _resolve_summary_relative_path(
            package_root=package_root,
            raw_value=str(payload.get("voice_samples_folder") or ""),
        )
        audio_slice_path = _resolve_summary_relative_path(
            package_root=package_root,
            raw_value=str(payload.get("audio_slice") or ""),
        )

        parsed_blocks = _parse_transcript_slice(transcript_slice_path)
        speakers = tuple(
            sorted(
                {
                    speaker
                    for block in parsed_blocks
                    for speaker, _ in block.utterances
                }
            )
        )
        original_manifest = json.loads(
            (voice_samples_dir / "manifest.json").read_text(encoding="utf-8")
        )
        raw_mapping = payload.get("bootstrap_mapping") or {}
        bootstrap_mapping = (
            {
                str(key): str(value)
                for key, value in raw_mapping.items()
            }
            if isinstance(raw_mapping, dict)
            else {}
        )
        prepared.append(
            ParsedEpisode(
                order=order,
                title=title,
                sample_id=sample_id,
                audio_slice_path=audio_slice_path,
                transcript_blocks=parsed_blocks,
                speakers=speakers,
                voice_samples_dir=voice_samples_dir,
                original_voice_manifest=original_manifest,
                bootstrap_mapping=bootstrap_mapping,
            )
        )
    if not prepared:
        raise RuntimeError("No episodes matched the selected filters.")
    return tuple(prepared)


def _resolve_summary_relative_path(*, package_root: Path, raw_value: str) -> Path:
    candidate = Path(raw_value.replace("\\", "/"))
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve() if candidate.parts else package_root


def _parse_transcript_slice(transcript_slice_path: Path) -> tuple[ParsedTranscriptBlock, ...]:
    raw_text = transcript_slice_path.read_text(encoding="utf-8")
    normalized_text = raw_text.replace("\r\n", "\n")
    lines = normalized_text.splitlines()
    body_lines: list[str] = []
    in_body = False
    for line in lines:
        if in_body:
            body_lines.append(line)
            continue
        if not line.strip():
            in_body = True
    body_text = "\n".join(body_lines).strip()
    blocks: list[ParsedTranscriptBlock] = []
    for raw_block in SECTION_SPLIT_PATTERN.split(body_text):
        block = raw_block.strip()
        if not block:
            continue
        parsed_block = _parse_transcript_block(block)
        if parsed_block is not None:
            blocks.append(parsed_block)
    if not blocks:
        raise RuntimeError(
            f"No transcript blocks were parsed from {transcript_slice_path}."
        )
    return tuple(blocks)


def _parse_transcript_block(raw_block: str) -> ParsedTranscriptBlock | None:
    lines = [line.rstrip() for line in raw_block.splitlines() if line.strip()]
    if not lines:
        return None
    first_line = lines[0].lstrip()
    index_match = re.match(r"^(\d+)\.\s*(.*)$", first_line)
    if index_match is None:
        return None
    index = int(index_match.group(1))
    transcript_lines = [index_match.group(2).strip()]
    source_line = ""
    rationale = ""
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("Source:"):
            source_line = stripped
            continue
        if stripped.startswith("Rationale:"):
            rationale = _clean_text(stripped[len("Rationale:") :].strip())
            continue
        transcript_lines.append(stripped)

    source_match = SOURCE_PATTERN.match(source_line)
    if source_match is None:
        raise RuntimeError(f"Failed to parse Source line: {source_line}")
    source_range_start = int(source_match.group(1))
    source_range_end = int(source_match.group(2))
    source_start_ms = _parse_hms_to_ms(source_match.group(3))
    source_end_ms = _parse_hms_to_ms(source_match.group(4))
    source_mode = source_match.group(5).strip()
    measured_seconds = float(source_match.group(6))
    if source_end_ms <= source_start_ms:
        source_end_ms = source_start_ms + max(1, int(round(measured_seconds * 1000.0)))

    utterances: list[tuple[str, str]] = []
    current_speaker = ""
    current_text_parts: list[str] = []
    for raw_line in transcript_lines:
        parsed_line = _parse_utterance_line(raw_line)
        if parsed_line is None:
            continuation = _clean_text(raw_line)
            if continuation and current_speaker:
                current_text_parts.append(continuation)
            continue
        speaker, text = parsed_line
        if current_speaker:
            utterances.append((current_speaker, " ".join(current_text_parts).strip()))
        current_speaker = speaker
        current_text_parts = [text]
    if current_speaker:
        utterances.append((current_speaker, " ".join(current_text_parts).strip()))

    if not utterances:
        raise RuntimeError(f"Transcript block {index} did not yield utterances.")

    return ParsedTranscriptBlock(
        index=index,
        source_range_start=source_range_start,
        source_range_end=source_range_end,
        source_start_ms=source_start_ms,
        source_end_ms=source_end_ms,
        source_mode=source_mode,
        measured_seconds=measured_seconds,
        rationale=rationale,
        utterances=tuple(utterances),
    )


def _parse_utterance_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line:
        return None
    bracket_speakers: list[str] = []
    while True:
        match = LEADING_BRACKET_SPEAKER_PATTERN.match(line)
        if match is None:
            break
        bracket_speakers.append(match.group(1))
        line = match.group(2).lstrip()
    direct_match = UTTERANCE_LINE_PATTERN.match(line)
    if direct_match is not None:
        speaker = direct_match.group(1)
        text = _clean_text(direct_match.group(2))
        return (speaker, text) if text else None
    if bracket_speakers and line:
        return bracket_speakers[-1], _clean_text(line)
    return None


def _clean_text(text: str) -> str:
    cleaned = str(text)
    if any(marker in cleaned for marker in ("â", "Ã", "Â")):
        try:
            repaired = cleaned.encode("latin-1").decode("utf-8")
            if repaired.count("â") + repaired.count("Ã") < cleaned.count("â") + cleaned.count("Ã"):
                cleaned = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    for old, new in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(old, new)
    cleaned = cleaned.replace("\u2014", "-")
    cleaned = cleaned.replace("\u2013", "-")
    cleaned = cleaned.replace("\u2019", "'")
    cleaned = cleaned.replace("\u2018", "'")
    cleaned = cleaned.replace("\u201c", '"')
    cleaned = cleaned.replace("\u201d", '"')
    cleaned = cleaned.replace("\u2026", "...")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _parse_hms_to_ms(value: str) -> int:
    hours, minutes, seconds = [int(part) for part in value.split(":")]
    return ((hours * 60 + minutes) * 60 + seconds) * 1000


def _materialize_preset_inputs(*, preset_root: Path, episode: ParsedEpisode) -> Path:
    speaker_samples_dir = (preset_root / "speaker_samples").resolve()
    speaker_samples_dir.mkdir(parents=True, exist_ok=True)
    speaker_manifest_path = _rewrite_voice_samples_manifest(
        episode=episode,
        speaker_samples_dir=speaker_samples_dir,
    )

    segments: list[TranscriptSegment] = []
    for block in episode.transcript_blocks:
        block_duration_ms = max(1, block.source_end_ms - block.source_start_ms)
        weights = [max(1, len(text.split())) for _, text in block.utterances]
        total_weight = max(1, sum(weights))
        previous_end_ms = block.source_start_ms
        cumulative_weight = 0
        for index, (speaker, text) in enumerate(block.utterances, start=1):
            start_ms = previous_end_ms
            if index == len(block.utterances):
                end_ms = block.source_end_ms
            else:
                cumulative_weight += weights[index - 1]
                end_ms = block.source_start_ms + int(
                    round(block_duration_ms * (cumulative_weight / total_weight))
                )
                end_ms = max(end_ms, start_ms + 1)
                end_ms = min(end_ms, block.source_end_ms)
            previous_end_ms = end_ms
            segments.append(
                TranscriptSegment(
                    speaker=speaker,
                    text=text,
                    start_time=start_ms,
                    end_time=end_ms,
                    extras={
                        "selection_index": block.index,
                        "source_range_start": block.source_range_start,
                        "source_range_end": block.source_range_end,
                        "source_mode": block.source_mode,
                        "measured_duration_seconds": block.measured_seconds,
                        "selection_rationale": block.rationale,
                    },
                )
            )

    transcript_payload = Transcript(
        segments=tuple(segments),
        metadata={
            "dataset": "sporc_human_eval_assets_10m",
            "sample_id": episode.sample_id,
            "episode_title": episode.title,
            "speaker_samples_manifest_path": _to_container_path(speaker_manifest_path),
            "source_audio_path": str(_to_container_path(episode.audio_slice_path)),
            "transcript_cleaned_from_segments": True,
            "transcript_cleaned_at_utc": _utc_now_iso(),
            "transcript_cleaning_mode": "sporc_segment_slice_cleanup",
            "speaker_normalization": {
                "mode": "transcript_only_benchmark",
                "speaker_assignment_mode": "raw_with_uncertainty",
            },
        },
    ).to_payload()

    transcript_path = (preset_root / "transcript.json").resolve()
    write_transcript_atomic(transcript_payload, transcript_path)
    return transcript_path


def _rewrite_voice_samples_manifest(
    *,
    episode: ParsedEpisode,
    speaker_samples_dir: Path,
) -> Path:
    original_manifest = dict(episode.original_voice_manifest)
    raw_samples = original_manifest.get("samples", [])
    samples_by_speaker: dict[str, dict[str, Any]] = {}
    if isinstance(raw_samples, list):
        for item in raw_samples:
            if isinstance(item, dict):
                speaker = str(item.get("speaker") or "").strip()
                if speaker:
                    samples_by_speaker[speaker] = dict(item)

    rewritten_samples: list[dict[str, Any]] = []
    for transcript_speaker in episode.speakers:
        source_speaker = episode.bootstrap_mapping.get(
            transcript_speaker,
            transcript_speaker,
        )
        source_audio_path = (episode.voice_samples_dir / f"{source_speaker}.wav").resolve()
        if not source_audio_path.exists():
            raise FileNotFoundError(
                "Missing source voice sample for transcript speaker "
                f"{transcript_speaker}: {source_audio_path}"
            )
        dest_audio_path = (speaker_samples_dir / f"{transcript_speaker}.wav").resolve()
        shutil.copyfile(source_audio_path, dest_audio_path)

        source_entry = dict(samples_by_speaker.get(source_speaker, {}))
        sample_entry = {
            "speaker": transcript_speaker,
            "path": dest_audio_path.name,
            "requested_duration_ms": int(source_entry.get("requested_duration_ms", 30000)),
            "duration_ms": int(source_entry.get("duration_ms", 30000)),
            "status": str(source_entry.get("status", "ok")),
            "selected_slice_count": int(source_entry.get("selected_slice_count", 1)),
            "dropped_slice_count": int(source_entry.get("dropped_slice_count", 0)),
            "ranges_ms": list(source_entry.get("ranges_ms", [])),
        }
        rewritten_samples.append(sample_entry)

    rewritten_manifest = {
        "generated_at_utc": _utc_now_iso(),
        "source_audio_path": str(_to_container_path(episode.audio_slice_path)),
        "target_duration_ms": int(original_manifest.get("target_duration_ms", 30000)),
        "clip_method": str(original_manifest.get("clip_method", "concat_turns")),
        "short_speaker_policy": str(
            original_manifest.get("short_speaker_policy", "export_shorter")
        ),
        "sample_rate_hz": int(original_manifest.get("sample_rate_hz", 44100)),
        "channels": int(original_manifest.get("channels", 1)),
        "codec": str(original_manifest.get("codec", "pcm_s24le")),
        "edge_fade_ms": int(original_manifest.get("edge_fade_ms", 20)),
        "min_slice_duration_ms": int(
            original_manifest.get("min_slice_duration_ms", 1200)
        ),
        "max_slices": int(original_manifest.get("max_slices", 6)),
        "selection_order": str(
            original_manifest.get("selection_order", "chronological")
        ),
        "low_data_policy": str(
            original_manifest.get("low_data_policy", "quality_first_shorter")
        ),
        "samples": rewritten_samples,
    }
    manifest_path = (speaker_samples_dir / "manifest.json").resolve()
    _write_json(rewritten_manifest, manifest_path)
    return manifest_path


def _run_setup_only(*, llm_profile: str, llm_model: str, log_path: Path) -> int:
    command = [
        "docker",
        "compose",
        "--profile",
        "gpu",
        "run",
        "--rm",
        "setup-and-run-gpu",
        "--setup-only",
        "--override",
        "audio.device=cuda",
        "--override",
        "audio.voice_clone.device=cuda",
        "--override",
        "audio.voice_clone.enabled=true",
        "--override",
        "audio.interjector.enabled=true",
        "--override",
        f"selection.llm={llm_profile}",
        "--override",
        f"selection.stage_llm.summarizer={llm_profile}",
        "--override",
        f"selection.stage_llm.critic={llm_profile}",
        "--override",
        f"selection.stage_llm.interjector={llm_profile}",
        "--override",
        f"profiles.llm.{llm_profile}.model={llm_model}",
    ]
    return _run_with_tee(command=command, log_path=log_path)


def _build_run_command(
    *,
    output_root: Path,
    transcript_path: Path,
    target_seconds: int,
    llm_profile: str,
    llm_model: str,
    orchestrator_max_iterations: int = 0,
    summarizer_max_tool_turns: int = 0,
    critic_max_tool_turns: int = 0,
    disable_interjector_tool_call_planning: bool = False,
) -> list[str]:
    command = [
        "docker",
        "compose",
        "--profile",
        "gpu",
        "run",
        "--rm",
        "setup-and-run-gpu",
        "--no-menu",
        "--output-root",
        str(_to_container_path(output_root)),
        "--override",
        "pipeline.start_stage=stage-2",
        "--override",
        f"transcript_path={_to_container_path(transcript_path)}",
        "--override",
        "selection.embedding=disabled",
        "--override",
        f"selection.llm={llm_profile}",
        "--override",
        f"selection.stage_llm.summarizer={llm_profile}",
        "--override",
        f"selection.stage_llm.critic={llm_profile}",
        "--override",
        f"selection.stage_llm.interjector={llm_profile}",
        "--override",
        f"profiles.llm.{llm_profile}.model={llm_model}",
        "--override",
        f"orchestrator.target_seconds={target_seconds}",
        "--override",
        "audio.device=cuda",
        "--override",
        "audio.voice_clone.device=cuda",
        "--override",
        "audio.voice_clone.enabled=true",
        "--override",
        "audio.interjector.enabled=true",
        "--override",
        "logging.print_to_terminal=true",
        "--override",
        "logging.summarizer_critic_print_to_terminal=true",
        "--override",
        "orchestrator.timeouts.summarizer=900",
        "--override",
        "orchestrator.timeouts.critic=900",
        "--override",
        "orchestrator.timeouts.retrieval=300",
        "--override",
        "orchestrator.timeouts.index=300",
    ]
    if orchestrator_max_iterations > 0:
        command.extend(
            [
                "--override",
                f"orchestrator.max_iterations={orchestrator_max_iterations}",
            ]
        )
    if summarizer_max_tool_turns > 0:
        command.extend(
            [
                "--override",
                f"agents.summarizer.max_tool_turns={summarizer_max_tool_turns}",
            ]
        )
    if critic_max_tool_turns > 0:
        command.extend(
            [
                "--override",
                f"agents.critic.max_tool_turns={critic_max_tool_turns}",
            ]
        )
    if disable_interjector_tool_call_planning:
        command.extend(
            [
                "--override",
                "audio.interjector.tool_call_planning_enabled=false",
            ]
        )
    return command


def _run_with_tee(*, command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(_sanitize_console_text(line))
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        return process.wait()


def _collect_artifact_paths(output_root: Path) -> dict[str, str]:
    return {
        "summary_xml_path": str((output_root / "summary.xml").resolve()),
        "transcript_path": str((output_root / "transcript.json").resolve()),
        "speaker_manifest_path": str(
            (output_root / "speaker_samples" / "manifest.json").resolve()
        ),
        "voice_clone_manifest_path": str(
            (output_root / "audio_stage" / "voice_clone" / "manifest.json").resolve()
        ),
        "interjector_manifest_path": str(
            (
                output_root
                / "audio_stage"
                / "interjector"
                / "interjector_manifest.json"
            ).resolve()
        ),
        "final_audio_path": str(
            (
                output_root
                / "audio_stage"
                / "interjector"
                / "voice_cloned_interjected.wav"
            ).resolve()
        ),
    }


def _observe_completed_output(output_root: Path) -> dict[str, Any]:
    """Collect lightweight review notes from the completed LLM output artifacts."""
    summary_xml_path = (output_root / "summary.xml").resolve()
    interjector_manifest_path = (
        output_root / "audio_stage" / "interjector" / "interjector_manifest.json"
    ).resolve()
    review: dict[str, Any] = {
        "summary_xml_path": str(summary_xml_path),
        "interjector_manifest_path": str(interjector_manifest_path),
        "turn_count": 0,
        "word_count": 0,
        "speakers": [],
        "preview_lines": [],
        "interjection_artifact_count": 0,
        "concerns": [],
    }

    if summary_xml_path.exists():
        summary_xml = summary_xml_path.read_text(encoding="utf-8")
        turns = parse_summary_xml(summary_xml)
        review["turn_count"] = len(turns)
        review["word_count"] = sum(len(turn.text.split()) for turn in turns)
        review["speakers"] = [turn.speaker for turn in turns]
        review["preview_lines"] = [
            f"{turn.speaker}: {_truncate_text(turn.text, 120)}" for turn in turns[:3]
        ]
        if len(turns) <= 1:
            review["concerns"].append("summary_has_one_or_fewer_turns")
        if review["word_count"] <= 20:
            review["concerns"].append("summary_word_count_suspiciously_low")
        adjacent_duplicates = 0
        for previous, current in zip(turns, turns[1:], strict=False):
            if (
                previous.speaker == current.speaker
                and previous.text.strip() == current.text.strip()
            ):
                adjacent_duplicates += 1
        if adjacent_duplicates > 0:
            review["concerns"].append("adjacent_duplicate_turn_text_detected")

    if interjector_manifest_path.exists():
        try:
            manifest_payload = json.loads(
                interjector_manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            review["concerns"].append("interjector_manifest_unreadable")
        else:
            artifacts = manifest_payload.get("artifacts", [])
            if isinstance(artifacts, list):
                review["interjection_artifact_count"] = len(artifacts)
            else:
                review["concerns"].append("interjector_manifest_missing_artifacts_list")

    return review


def _render_output_review_note(review: dict[str, Any]) -> str:
    """Render one concise human-readable note line for the experimentation log."""
    preview_lines = review.get("preview_lines", [])
    preview = " | ".join(str(item) for item in preview_lines[:2]) if preview_lines else "n/a"
    concerns = review.get("concerns", [])
    concerns_text = ", ".join(str(item) for item in concerns) if concerns else "none"
    return (
        f"turns={int(review.get('turn_count', 0))}, "
        f"words={int(review.get('word_count', 0))}, "
        f"interjections={int(review.get('interjection_artifact_count', 0))}, "
        f"preview={_truncate_text(preview, 220)}, "
        f"concerns={concerns_text}."
    )


def _truncate_text(value: str, limit: int) -> str:
    """Return one single-line preview capped to a reasonable note length."""
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _export_human_readable_artifacts(
    *,
    episode: ParsedEpisode,
    preset: DurationPreset,
    preset_root: Path,
    final_exports_root: Path,
) -> dict[str, str]:
    readable_stem = _sanitize_filename(
        f"{episode.order:02d} - {episode.title} - {preset.label}"
    )
    exports: dict[str, str] = {}

    summary_source = (preset_root / "summary.xml").resolve()
    if summary_source.is_file():
        summary_dest = (final_exports_root / f"{readable_stem} - summary.xml").resolve()
        shutil.copyfile(summary_source, summary_dest)
        exports["summary_xml"] = str(summary_dest)

    audio_source = (
        preset_root
        / "audio_stage"
        / "interjector"
        / "voice_cloned_interjected.wav"
    ).resolve()
    if audio_source.is_file():
        audio_dest = (
            final_exports_root / f"{readable_stem} - voice_cloned_interjected.wav"
        ).resolve()
        shutil.copyfile(audio_source, audio_dest)
        exports["final_audio"] = str(audio_dest)
    return exports


def _to_container_path(host_path: Path) -> str:
    resolved = host_path.resolve()
    relative = resolved.relative_to(REPO_ROOT)
    return str(CONTAINER_REPO_ROOT.joinpath(*relative.parts))


def _append_note(notes_path: Path, message: str) -> None:
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    with notes_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n### {_iso_with_timezone()}\n\n- {message}\n")


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _human_episode_label(order: int, title: str) -> str:
    return _sanitize_filename(f"{order:02d} - {title}")


def _machine_episode_label(order: int, title: str) -> str:
    return f"{order:02d}_{_slugify(title)}"


def _sanitize_filename(value: str) -> str:
    collapsed = INVALID_FILENAME_PATTERN.sub(" - ", str(value))
    collapsed = re.sub(r"\s+", " ", collapsed).strip().rstrip(".")
    return collapsed or "artifact"


def _slugify(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return collapsed.lower() or "item"


def _sanitize_console_text(text: str) -> str:
    sanitized = str(text).translate(CONSOLE_SAFE_TRANSLATIONS)
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return sanitized
    return sanitized.encode(encoding, errors="replace").decode(encoding)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iso_with_timezone() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
