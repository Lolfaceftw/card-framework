"""Use-case orchestrator for post-summary voice cloning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from audio_pipeline.errors import ArtifactWriteError, NonRetryableAudioStageError
from audio_pipeline.voice_clone_contracts import (
    VoiceCloneArtifact,
    VoiceCloneProvider,
    VoiceCloneTurn,
    VoiceSampleReference,
)


@dataclass(slots=True, frozen=True)
class VoiceCloneRunResult:
    """Result bundle for one voice-cloning run."""

    output_dir: Path
    manifest_path: Path
    generated_at_utc: str
    artifacts: tuple[VoiceCloneArtifact, ...]


@dataclass(slots=True)
class VoiceCloneOrchestrator:
    """
    Generate turn-level cloned-audio artifacts from summary XML.

    Args:
        provider: Concrete voice-cloning strategy adapter.
        output_dir: Destination directory for synthesized WAV artifacts.
        fail_on_error: Whether synthesis errors should abort the run.
        manifest_filename: Output manifest filename.
    """

    provider: VoiceCloneProvider
    output_dir: Path
    fail_on_error: bool = True
    manifest_filename: str = "manifest.json"

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
        turns = _parse_summary_turns(summary_xml)
        sample_refs = _load_speaker_sample_references(speaker_samples_manifest_path)
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
                rendered_path = self.provider.synthesize(
                    reference_audio_path=reference.path,
                    text=turn.text,
                    output_audio_path=output_audio_path,
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

        generated_at_utc = _utc_now_iso()
        manifest_payload = {
            "generated_at_utc": generated_at_utc,
            "speaker_samples_manifest_path": str(speaker_samples_manifest_path),
            "artifact_count": len(artifacts),
            "artifacts": [
                {
                    "turn_index": artifact.turn_index,
                    "speaker": artifact.speaker,
                    "text": artifact.text,
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
        )


def _parse_summary_turns(summary_xml: str) -> list[VoiceCloneTurn]:
    """Parse summary speaker turns from simple XML tag pairs."""
    normalized = summary_xml.strip()
    if not normalized:
        raise NonRetryableAudioStageError("Summary XML is empty; cannot clone voice.")

    pattern = re.compile(
        r"<(?P<speaker>[A-Za-z0-9_.-]+)>(?P<text>.*?)</(?P=speaker)>",
        re.DOTALL,
    )
    turns: list[VoiceCloneTurn] = []
    for match in pattern.finditer(normalized):
        speaker = match.group("speaker").strip()
        text = match.group("text").strip()
        if not text:
            continue
        turns.append(VoiceCloneTurn(speaker=speaker, text=text))

    if not turns:
        raise NonRetryableAudioStageError(
            "No speaker turns were found in summary XML for voice cloning."
        )
    return turns


def _load_speaker_sample_references(
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
