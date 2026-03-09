"""Use-case orchestrator for post-summary voice cloning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from card_framework.audio_pipeline.errors import ArtifactWriteError, NonRetryableAudioStageError
from card_framework.audio_pipeline.runtime import ensure_command_available
from card_framework.audio_pipeline.voice_clone_contracts import (
    VoiceCloneArtifact,
    VoiceCloneProvider,
    VoiceCloneTurn,
    VoiceSampleReference,
)
from card_framework.shared.summary_xml import DEFAULT_EMO_PRESET, parse_summary_xml


@dataclass(slots=True, frozen=True)
class VoiceCloneRunResult:
    """Result bundle for one voice-cloning run."""

    output_dir: Path
    manifest_path: Path
    generated_at_utc: str
    artifacts: tuple[VoiceCloneArtifact, ...]
    merged_output_audio_path: Path | None = None


@dataclass(slots=True)
class VoiceCloneOrchestrator:
    """
    Generate turn-level cloned-audio artifacts from summary XML.

    Args:
        provider: Concrete voice-cloning strategy adapter.
        output_dir: Destination directory for synthesized WAV artifacts.
        fail_on_error: Whether synthesis errors should abort the run.
        manifest_filename: Output manifest filename.
        merge_segments: Whether to merge all turn-level outputs into one WAV.
        merged_output_filename: Filename for merged output audio.
        merge_audio_codec: FFmpeg audio codec for merged WAV output.
        merge_timeout_seconds: Timeout applied to FFmpeg merge command.
    """

    provider: VoiceCloneProvider
    output_dir: Path
    fail_on_error: bool = True
    manifest_filename: str = "manifest.json"
    merge_segments: bool = True
    merged_output_filename: str = "voice_cloned.wav"
    merge_audio_codec: str = "pcm_s24le"
    merge_timeout_seconds: int = 300
    emo_preset_catalog: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate orchestrator configuration."""
        if not self.manifest_filename.strip():
            raise ValueError("manifest_filename must be non-empty.")
        if self.merge_segments:
            if not self.merged_output_filename.strip():
                raise ValueError("merged_output_filename must be non-empty.")
            if Path(self.merged_output_filename).suffix.lower() != ".wav":
                raise ValueError("merged_output_filename must end with '.wav'.")
        if not self.merge_audio_codec.strip():
            raise ValueError("merge_audio_codec must be non-empty.")
        if self.merge_timeout_seconds <= 0:
            raise ValueError("merge_timeout_seconds must be > 0.")

    def run(
        self,
        *,
        summary_xml: str,
        speaker_samples_manifest_path: Path,
        progress_callback: StageProgressCallback | None = None,
    ) -> VoiceCloneRunResult:
        """
        Run voice cloning for each speaker turn in summary XML.

        Args:
            summary_xml: Final summary XML containing speaker-tagged turns.
            speaker_samples_manifest_path: Path to speaker-sample manifest JSON.
            progress_callback: Optional callback invoked after progress updates.

        Returns:
            Structured run result including synthesized artifacts and manifest path.

        Raises:
            NonRetryableAudioStageError: If required inputs are missing/invalid.
            ArtifactWriteError: If manifest persistence fails.
        """
        turns = parse_summary_turns(summary_xml)
        sample_refs = load_speaker_sample_references(speaker_samples_manifest_path)
        total_turns = len(turns)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[VoiceCloneArtifact] = []
        if progress_callback is not None:
            try:
                progress_callback(
                    StageProgressUpdate(
                        completed_units=0,
                        total_units=total_turns,
                        note="voice clone stage started",
                    )
                )
            except Exception:
                pass
        for index, turn in enumerate(turns, start=1):
            reference = sample_refs.get(turn.speaker)
            if reference is None:
                message = (
                    "Missing speaker sample for summary speaker "
                    f"'{turn.speaker}' in {speaker_samples_manifest_path}."
                )
                if self.fail_on_error:
                    raise NonRetryableAudioStageError(message)
                continue

            output_audio_path = self.output_dir / _build_output_filename(
                turn_index=index,
                speaker=turn.speaker,
            )
            try:
                emo_text = _resolve_emo_text(
                    emo_preset=turn.emo_preset,
                    emo_preset_catalog=self.emo_preset_catalog,
                )
                rendered_path = self.provider.synthesize(
                    reference_audio_path=reference.path,
                    text=turn.text,
                    output_audio_path=output_audio_path,
                    emo_text=emo_text,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                if self.fail_on_error:
                    raise NonRetryableAudioStageError(
                        "Voice cloning failed for turn "
                        f"{index} speaker={turn.speaker}."
                    ) from exc
                continue

            artifacts.append(
                VoiceCloneArtifact(
                    turn_index=index,
                    speaker=turn.speaker,
                    text=turn.text,
                    emo_preset=turn.emo_preset,
                    reference_audio_path=reference.path,
                    output_audio_path=rendered_path,
                )
            )
            if progress_callback is not None:
                try:
                    progress_callback(
                        StageProgressUpdate(
                            completed_units=len(artifacts),
                            total_units=total_turns,
                            note=f"voice clone artifact generated for turn {index}",
                        )
                    )
                except Exception:
                    pass

        if not artifacts:
            raise NonRetryableAudioStageError(
                "Voice cloning produced no artifacts from summary turns."
            )

        merged_output_audio_path: Path | None = None
        if self.merge_segments:
            merged_output_audio_path = self.output_dir / self.merged_output_filename
            try:
                merge_audio_artifacts_to_wav(
                    artifact_paths=[artifact.output_audio_path for artifact in artifacts],
                    output_path=merged_output_audio_path,
                    audio_codec=self.merge_audio_codec,
                    timeout_seconds=self.merge_timeout_seconds,
                )
            except Exception as exc:
                if self.fail_on_error:
                    raise NonRetryableAudioStageError(
                        "Failed to merge voice clone artifacts into one WAV output."
                    ) from exc
                merged_output_audio_path = None

        generated_at_utc = _utc_now_iso()
        manifest_payload = {
            "generated_at_utc": generated_at_utc,
            "speaker_samples_manifest_path": str(speaker_samples_manifest_path),
            "artifact_count": len(artifacts),
            "merge_segments": self.merge_segments,
            "merged_output_audio_path": (
                str(merged_output_audio_path)
                if merged_output_audio_path is not None
                else None
            ),
            "artifacts": [
                {
                    "turn_index": artifact.turn_index,
                    "speaker": artifact.speaker,
                    "text": artifact.text,
                    "emo_preset": artifact.emo_preset,
                    "reference_audio_path": str(artifact.reference_audio_path),
                    "output_audio_path": str(artifact.output_audio_path),
                }
                for artifact in artifacts
            ],
        }
        manifest_path = self.output_dir / self.manifest_filename
        _write_json_atomic(manifest_payload, manifest_path)
        return VoiceCloneRunResult(
            output_dir=self.output_dir,
            manifest_path=manifest_path,
            generated_at_utc=generated_at_utc,
            artifacts=tuple(artifacts),
            merged_output_audio_path=merged_output_audio_path,
        )


def parse_summary_turns(summary_xml: str) -> list[VoiceCloneTurn]:
    """Parse summary speaker turns from simple XML tag pairs."""
    normalized = summary_xml.strip()
    if not normalized:
        raise NonRetryableAudioStageError("Summary XML is empty; cannot clone voice.")
    try:
        raw_turns = parse_summary_xml(normalized)
    except ValueError as exc:
        raise NonRetryableAudioStageError(
            "No speaker turns were found in summary XML for voice cloning."
        ) from exc
    turns = [
        VoiceCloneTurn(
            speaker=turn.speaker,
            text=turn.text,
            emo_preset=turn.emo_preset,
        )
        for turn in raw_turns
    ]
    if not turns:
        raise NonRetryableAudioStageError(
            "No speaker turns were found in summary XML for voice cloning."
        )
    return turns


def load_speaker_sample_references(
    manifest_path: Path,
) -> dict[str, VoiceSampleReference]:
    """Load speaker sample references, selecting longest sample per speaker."""
    if not manifest_path.exists():
        raise NonRetryableAudioStageError(
            f"Speaker sample manifest does not exist: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise NonRetryableAudioStageError(
            f"Speaker sample manifest is not valid JSON: {manifest_path}"
        ) from exc

    raw_samples = payload.get("samples", [])
    if not isinstance(raw_samples, list) or not raw_samples:
        raise NonRetryableAudioStageError(
            "Speaker sample manifest must include non-empty 'samples' list."
        )

    refs: dict[str, VoiceSampleReference] = {}
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            continue
        speaker = str(raw_sample.get("speaker", "")).strip()
        path_value = str(raw_sample.get("path", "")).strip()
        duration_value = raw_sample.get("duration_ms", 0)
        if not speaker or not path_value:
            continue
        try:
            duration_ms = int(duration_value)
        except (TypeError, ValueError):
            continue
        if duration_ms <= 0:
            continue
        candidate_path = _resolve_manifest_audio_path(
            manifest_path=manifest_path,
            audio_path_value=path_value,
        )
        if not candidate_path.exists():
            continue

        current = refs.get(speaker)
        if current is None or duration_ms > current.duration_ms:
            refs[speaker] = VoiceSampleReference(
                speaker=speaker,
                path=candidate_path,
                duration_ms=duration_ms,
            )

    if not refs:
        raise NonRetryableAudioStageError(
            "No usable speaker sample entries were found in speaker-sample manifest."
        )
    return refs


def _resolve_manifest_audio_path(*, manifest_path: Path, audio_path_value: str) -> Path:
    """Resolve one audio path listed in speaker-sample manifest."""
    candidate = Path(audio_path_value)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def _build_output_filename(*, turn_index: int, speaker: str) -> str:
    """Build deterministic artifact filename for one synthesized turn."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", speaker.strip()).strip("._")
    if not sanitized:
        sanitized = "speaker"
    return f"{turn_index:03d}_{sanitized}.wav"


def _resolve_emo_text(
    *,
    emo_preset: str,
    emo_preset_catalog: dict[str, str] | None,
) -> str | None:
    """Resolve emotion-guidance text for one turn preset."""
    if not emo_preset_catalog:
        return None
    normalized_preset = emo_preset.strip() or DEFAULT_EMO_PRESET
    emo_text = emo_preset_catalog.get(normalized_preset)
    if emo_text is not None:
        return emo_text
    return emo_preset_catalog.get(DEFAULT_EMO_PRESET)


def merge_audio_artifacts_to_wav(
    *,
    artifact_paths: Sequence[Path],
    output_path: Path,
    audio_codec: str,
    timeout_seconds: int,
) -> None:
    """Merge turn-level voice-clone WAV artifacts into one WAV output."""
    if not artifact_paths:
        raise NonRetryableAudioStageError(
            "Voice clone merge requires at least one artifact."
        )
    for artifact_path in artifact_paths:
        if not artifact_path.exists():
            raise NonRetryableAudioStageError(
                f"Voice clone artifact missing for merge: {artifact_path}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = _build_temp_output_path(output_path)

    if len(artifact_paths) == 1:
        source = artifact_paths[0]
        if source.resolve() == output_path.resolve():
            return
        try:
            shutil.copyfile(source, temp_output_path)
            temp_output_path.replace(output_path)
            return
        except Exception as exc:
            _remove_temp_output(temp_output_path)
            raise NonRetryableAudioStageError(
                f"Failed to persist merged voice clone artifact: {output_path}"
            ) from exc

    ensure_command_available("ffmpeg")
    filter_inputs = "".join(f"[{index}:a]" for index in range(len(artifact_paths)))
    filter_complex = f"{filter_inputs}concat=n={len(artifact_paths)}:v=0:a=1[outa]"
    command: list[str] = ["ffmpeg", "-y"]
    for artifact_path in artifact_paths:
        command.extend(["-i", str(artifact_path)])
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-c:a",
            audio_codec,
            "-vn",
            "-f",
            "wav",
            str(temp_output_path),
        ]
    )
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
        temp_output_path.replace(output_path)
    except subprocess.TimeoutExpired as exc:
        _remove_temp_output(temp_output_path)
        raise NonRetryableAudioStageError(
            "Failed to merge voice clone artifacts due to timeout. "
            f"Command: {' '.join(command)}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        _remove_temp_output(temp_output_path)
        raise NonRetryableAudioStageError(
            "Failed to merge voice clone artifacts with ffmpeg. "
            f"Command: {' '.join(command)}. "
            f"Stderr: {(exc.stderr or '').strip()[:500]}"
        ) from exc
    except Exception as exc:
        _remove_temp_output(temp_output_path)
        raise NonRetryableAudioStageError(
            f"Failed to persist merged voice clone artifact: {output_path}"
        ) from exc


def _build_temp_output_path(output_path: Path) -> Path:
    """Build temp output path preserving the final file extension."""
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.tmp")


def _remove_temp_output(temp_path: Path) -> None:
    """Remove stale temporary output on best-effort basis."""
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass


def _utc_now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    """Persist JSON payload atomically via temporary sidecar path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(output_path)
    except Exception as exc:
        raise ArtifactWriteError(
            f"Failed to write voice clone manifest to '{output_path}'."
        ) from exc

