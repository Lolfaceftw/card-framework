"""Voice-clone calibration and duration-estimation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any

from agents.utils import count_words
from audio_pipeline.factory import (
    build_audio_to_script_orchestrator,
    build_speaker_sample_generator,
    build_voice_clone_provider,
)
from audio_pipeline.runtime import probe_audio_duration_ms, resolve_device, resolve_path
from audio_pipeline.speaker_samples import resolve_sample_source_audio_path
from orchestration.transcript import Transcript
from summary_xml import DEFAULT_EMO_PRESET, SummaryTurn

DEFAULT_DURATION_TOLERANCE_RATIO = 0.05
DEFAULT_CALIBRATION_ARTIFACT = "artifacts/calibration/voice_clone_calibration.json"
DEFAULT_CALIBRATION_PHRASES: tuple[str, ...] = (
    "So, here is the quick setup: we slow down a little, pause at the comma, and land cleanly at the end.",
    "Right, the point is simple. We keep the pacing conversational, with brief pauses, steady emphasis, and a natural finish.",
    "Yeah, if the delivery feels too rushed, add a comma, take a breath, and let the sentence settle before the next thought.",
)


@dataclass(slots=True, frozen=True)
class VoiceCloneCalibration:
    """Represent persisted calibration data for duration estimation."""

    artifact_path: Path
    generated_at_utc: str
    speaker_samples_manifest_path: Path
    preset_emo_texts: dict[str, str]
    calibration_phrases: tuple[str, ...]
    speaker_preset_wpm: dict[str, dict[str, float]]
    preset_default_wpm: dict[str, float]

    @classmethod
    def from_payload(cls, *, artifact_path: Path, payload: Mapping[str, Any]) -> "VoiceCloneCalibration":
        """Build a calibration object from persisted JSON payload."""
        raw_presets = payload.get("preset_emo_texts", {})
        preset_emo_texts = {
            str(name): str(emo_text)
            for name, emo_text in dict(raw_presets).items()
            if str(name).strip() and str(emo_text).strip()
        }
        raw_speaker_preset_wpm = payload.get("speaker_preset_wpm", {})
        speaker_preset_wpm: dict[str, dict[str, float]] = {}
        for speaker, preset_map in dict(raw_speaker_preset_wpm).items():
            normalized_preset_map: dict[str, float] = {}
            for preset_name, raw_value in dict(preset_map).items():
                try:
                    normalized_preset_map[str(preset_name)] = float(raw_value)
                except (TypeError, ValueError):
                    continue
            if normalized_preset_map:
                speaker_preset_wpm[str(speaker)] = normalized_preset_map
        raw_preset_default_wpm = payload.get("preset_default_wpm", {})
        preset_default_wpm = {
            str(name): float(value)
            for name, value in dict(raw_preset_default_wpm).items()
            if _is_positive_float(value)
        }
        manifest_path = Path(str(payload.get("speaker_samples_manifest_path", ""))).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(
                "Calibration artifact references a missing speaker-sample manifest: "
                f"{manifest_path}"
            )
        return cls(
            artifact_path=artifact_path.resolve(),
            generated_at_utc=str(payload.get("generated_at_utc", "")).strip(),
            speaker_samples_manifest_path=manifest_path,
            preset_emo_texts=preset_emo_texts,
            calibration_phrases=tuple(
                str(item)
                for item in payload.get("calibration_phrases", DEFAULT_CALIBRATION_PHRASES)
                if str(item).strip()
            )
            or DEFAULT_CALIBRATION_PHRASES,
            speaker_preset_wpm=speaker_preset_wpm,
            preset_default_wpm=preset_default_wpm,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize calibration object into JSON-compatible payload."""
        return {
            "generated_at_utc": self.generated_at_utc,
            "speaker_samples_manifest_path": str(self.speaker_samples_manifest_path),
            "preset_emo_texts": dict(self.preset_emo_texts),
            "calibration_phrases": list(self.calibration_phrases),
            "speaker_preset_wpm": {
                speaker: {preset: round(value, 6) for preset, value in preset_map.items()}
                for speaker, preset_map in self.speaker_preset_wpm.items()
            },
            "preset_default_wpm": {
                preset: round(value, 6) for preset, value in self.preset_default_wpm.items()
            },
        }

    def emo_text_for_preset(self, emo_preset: str) -> str:
        """Return the configured emotion text for one preset name."""
        normalized = emo_preset.strip() or DEFAULT_EMO_PRESET
        try:
            return self.preset_emo_texts[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown emo preset: {normalized}") from exc

    def wpm_for(self, *, speaker: str, emo_preset: str) -> float:
        """Resolve calibrated WPM for one speaker and preset with fallback."""
        normalized_preset = emo_preset.strip() or DEFAULT_EMO_PRESET
        speaker_map = self.speaker_preset_wpm.get(speaker.strip(), {})
        speaker_wpm = speaker_map.get(normalized_preset)
        if speaker_wpm is not None and speaker_wpm > 0:
            return speaker_wpm
        default_wpm = self.preset_default_wpm.get(normalized_preset)
        if default_wpm is not None and default_wpm > 0:
            return default_wpm
        neutral_wpm = self.preset_default_wpm.get(DEFAULT_EMO_PRESET)
        if neutral_wpm is not None and neutral_wpm > 0:
            return neutral_wpm
        for fallback_value in self.preset_default_wpm.values():
            if fallback_value > 0:
                return fallback_value
        raise ValueError("Calibration artifact does not contain any usable WPM values.")

    def estimate_text_seconds(
        self,
        *,
        speaker: str,
        emo_preset: str,
        text: str,
    ) -> float:
        """Estimate spoken duration for text using calibrated WPM."""
        words = count_words(text)
        if words <= 0:
            return 0.0
        wpm = self.wpm_for(speaker=speaker, emo_preset=emo_preset)
        return round((words / wpm) * 60.0, 3)

    def estimate_turn_seconds(self, turn: SummaryTurn) -> float:
        """Estimate spoken duration for one summary turn."""
        return self.estimate_text_seconds(
            speaker=turn.speaker,
            emo_preset=turn.emo_preset,
            text=turn.text,
        )

    def estimate_turns_seconds(self, turns: list[SummaryTurn]) -> float:
        """Estimate total spoken duration for many summary turns."""
        return round(sum(self.estimate_turn_seconds(turn) for turn in turns), 3)


def resolve_calibration_artifact_path(
    *,
    project_root: Path,
    audio_cfg: Mapping[str, Any],
) -> Path:
    """Resolve the calibration artifact path from audio config."""
    voice_clone_cfg = _as_mapping(audio_cfg.get("voice_clone", {}))
    calibration_cfg = _as_mapping(voice_clone_cfg.get("calibration", {}))
    artifact_value = str(
        calibration_cfg.get("artifact_path", DEFAULT_CALIBRATION_ARTIFACT)
    ).strip()
    if not artifact_value:
        artifact_value = DEFAULT_CALIBRATION_ARTIFACT
    return resolve_path(artifact_value, base_dir=project_root)


def resolve_emo_preset_catalog(audio_cfg: Mapping[str, Any]) -> dict[str, str]:
    """Resolve the allowed emo preset catalog from config."""
    voice_clone_cfg = _as_mapping(audio_cfg.get("voice_clone", {}))
    raw_catalog = _as_mapping(voice_clone_cfg.get("emo_presets", {}))
    presets = {
        str(name).strip(): str(emo_text).strip()
        for name, emo_text in raw_catalog.items()
        if str(name).strip() and str(emo_text).strip()
    }
    if not presets:
        raise ValueError("audio.voice_clone.emo_presets must define at least one preset.")
    if DEFAULT_EMO_PRESET not in presets:
        raise ValueError(
            "audio.voice_clone.emo_presets must include a "
            f"'{DEFAULT_EMO_PRESET}' preset."
        )
    return presets


def load_voice_clone_calibration(
    artifact_path: Path,
) -> VoiceCloneCalibration:
    """Load one persisted calibration artifact from disk."""
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    return VoiceCloneCalibration.from_payload(
        artifact_path=artifact_path,
        payload=payload,
    )


def ensure_voice_clone_calibration(
    *,
    project_root: Path,
    audio_cfg: Mapping[str, Any],
    speaker_samples_manifest_path: Path | None = None,
    transcript_path: Path | None = None,
    audio_path: Path | None = None,
    force: bool = False,
) -> VoiceCloneCalibration:
    """Ensure the project calibration artifact exists and return it."""
    artifact_path = resolve_calibration_artifact_path(
        project_root=project_root,
        audio_cfg=audio_cfg,
    )
    presets = resolve_emo_preset_catalog(audio_cfg)
    resolved_manifest = _resolve_or_prepare_speaker_samples_manifest(
        project_root=project_root,
        audio_cfg=audio_cfg,
        explicit_manifest_path=speaker_samples_manifest_path,
        explicit_transcript_path=transcript_path,
        explicit_audio_path=audio_path,
    )
    if artifact_path.exists() and not force:
        try:
            calibration = load_voice_clone_calibration(artifact_path)
        except Exception:
            calibration = None
        if calibration is not None and _calibration_matches_current_inputs(
            calibration=calibration,
            speaker_samples_manifest_path=resolved_manifest,
            preset_emo_texts=presets,
        ):
            return calibration

    provider = build_voice_clone_provider(audio_cfg, project_root=project_root)
    sample_refs = _load_speaker_sample_paths(resolved_manifest)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_preset_wpm: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory(
        prefix="voice_clone_calibration_",
        dir=str(artifact_path.parent),
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for speaker, reference_audio_path in sample_refs.items():
            preset_map: dict[str, float] = {}
            for preset_name, emo_text in presets.items():
                total_words = 0
                total_duration_seconds = 0.0
                for index, phrase in enumerate(DEFAULT_CALIBRATION_PHRASES, start=1):
                    output_audio_path = (
                        temp_dir / speaker / preset_name / f"{index:02d}.wav"
                    )
                    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
                    rendered_path = provider.synthesize(
                        reference_audio_path=reference_audio_path,
                        text=phrase,
                        output_audio_path=output_audio_path,
                        emo_text=emo_text,
                    )
                    duration_ms = probe_audio_duration_ms(rendered_path)
                    if duration_ms is None or duration_ms <= 0:
                        raise RuntimeError(
                            "Unable to determine calibration audio duration for "
                            f"{rendered_path}"
                        )
                    total_words += count_words(phrase)
                    total_duration_seconds += duration_ms / 1000.0
                preset_map[preset_name] = round(
                    (total_words / total_duration_seconds) * 60.0,
                    6,
                )
            speaker_preset_wpm[speaker] = preset_map

    preset_default_wpm = _build_preset_default_wpm(
        speaker_preset_wpm=speaker_preset_wpm,
        preset_names=tuple(presets.keys()),
    )
    calibration = VoiceCloneCalibration(
        artifact_path=artifact_path.resolve(),
        generated_at_utc=_utc_now_iso(),
        speaker_samples_manifest_path=resolved_manifest.resolve(),
        preset_emo_texts=presets,
        calibration_phrases=DEFAULT_CALIBRATION_PHRASES,
        speaker_preset_wpm=speaker_preset_wpm,
        preset_default_wpm=preset_default_wpm,
    )
    artifact_path.write_text(
        json.dumps(calibration.to_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return calibration


def _calibration_matches_current_inputs(
    *,
    calibration: VoiceCloneCalibration,
    speaker_samples_manifest_path: Path,
    preset_emo_texts: Mapping[str, str],
) -> bool:
    """Return whether a persisted calibration still matches current runtime inputs."""
    if (
        calibration.speaker_samples_manifest_path.resolve()
        != speaker_samples_manifest_path.resolve()
    ):
        return False
    if dict(calibration.preset_emo_texts) != dict(preset_emo_texts):
        return False
    try:
        manifest_speakers = set(
            _load_speaker_sample_paths(speaker_samples_manifest_path).keys()
        )
    except FileNotFoundError:
        return False
    calibrated_speakers = {
        speaker
        for speaker, preset_map in calibration.speaker_preset_wpm.items()
        if preset_map
    }
    return manifest_speakers.issubset(calibrated_speakers)


def _resolve_or_prepare_speaker_samples_manifest(
    *,
    project_root: Path,
    audio_cfg: Mapping[str, Any],
    explicit_manifest_path: Path | None,
    explicit_transcript_path: Path | None,
    explicit_audio_path: Path | None,
) -> Path:
    """Resolve speaker samples manifest or generate one from transcript context."""
    if explicit_manifest_path is not None:
        resolved = explicit_manifest_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Speaker-sample manifest not found: {resolved}"
            )
        return resolved

    latest_manifest = _find_latest_speaker_samples_manifest(project_root)
    if latest_manifest is not None:
        return latest_manifest

    transcript_candidate = explicit_transcript_path or _find_latest_transcript_path(project_root)
    if transcript_candidate is not None:
        return _generate_speaker_samples_from_transcript(
            project_root=project_root,
            audio_cfg=audio_cfg,
            transcript_path=transcript_candidate,
            audio_path=explicit_audio_path,
        )

    audio_candidate = _resolve_audio_path_candidate(
        project_root=project_root,
        audio_cfg=audio_cfg,
        explicit_audio_path=explicit_audio_path,
    )
    if audio_candidate is not None:
        return _generate_speaker_samples_from_audio(
            project_root=project_root,
            audio_cfg=audio_cfg,
            audio_path=audio_candidate,
        )

    raise FileNotFoundError(
        "No speaker-sample manifest, transcript artifact, or usable audio path was found. "
        "Provide --speaker-samples-manifest, --transcript-path, or --audio-path."
    )


def _generate_speaker_samples_from_transcript(
    *,
    project_root: Path,
    audio_cfg: Mapping[str, Any],
    transcript_path: Path,
    audio_path: Path | None,
) -> Path:
    """Generate speaker samples from an existing transcript artifact."""
    transcript_payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript = Transcript.from_mapping(transcript_payload)
    manifest_path_value = str(
        transcript.metadata.get("speaker_samples_manifest_path", "")
    ).strip()
    if manifest_path_value:
        manifest_path = Path(manifest_path_value)
        if not manifest_path.is_absolute():
            manifest_path = (project_root / manifest_path).resolve()
        if manifest_path.exists():
            return manifest_path

    work_dir = resolve_path(
        str(audio_cfg.get("work_dir", "artifacts/audio_stage")),
        base_dir=project_root,
    )
    speaker_samples_cfg = _as_mapping(audio_cfg.get("speaker_samples", {}))
    output_dir_name = str(
        speaker_samples_cfg.get("output_dir_name", "speaker_samples")
    ).strip() or "speaker_samples"
    output_dir = resolve_path(output_dir_name, base_dir=work_dir)

    configured_audio_path = str(audio_cfg.get("audio_path", "")).strip()
    if audio_path is not None:
        configured_audio_path = str(audio_path)
    source_audio_path = resolve_sample_source_audio_path(
        source_mode="vocals",
        transcript_metadata=transcript.metadata,
        configured_audio_path=configured_audio_path,
        base_dir=project_root,
    )
    generator = build_speaker_sample_generator(audio_cfg)
    result = generator.generate(
        transcript_payload=transcript.to_payload(),
        source_audio_path=source_audio_path,
        output_dir=output_dir,
        progress_callback=None,
    )

    metadata = dict(transcript.metadata)
    metadata.update(
        {
            "speaker_samples_manifest_path": str(result.manifest_path),
            "speaker_samples_dir": str(result.output_dir),
            "speaker_sample_count": len(result.artifacts),
            "speaker_samples_generated_at_utc": result.generated_at_utc,
        }
    )
    updated_transcript = transcript.with_metadata(metadata)
    transcript_path.write_text(
        json.dumps(updated_transcript.to_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result.manifest_path.resolve()


def _generate_speaker_samples_from_audio(
    *,
    project_root: Path,
    audio_cfg: Mapping[str, Any],
    audio_path: Path,
) -> Path:
    """Run the audio stage to create transcript context before speaker extraction."""
    calibration_root = project_root / "artifacts" / "calibration"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    transcript_path = (
        calibration_root / "generated_transcripts" / f"{run_id}.transcript.json"
    ).resolve()
    work_dir = (calibration_root / "bootstrap_audio_stage" / run_id).resolve()
    audio_orchestrator = build_audio_to_script_orchestrator(audio_cfg)
    audio_orchestrator.run(
        input_audio_path=audio_path.resolve(),
        output_transcript_path=transcript_path,
        work_dir=work_dir,
        device=resolve_device(str(audio_cfg.get("device", "auto"))),
        metadata_overrides={
            "separator_model": str(
                _as_mapping(audio_cfg.get("separation", {})).get("model", "htdemucs")
            ),
            "transcriber_model": str(
                _as_mapping(audio_cfg.get("asr", {})).get("model", "large-v3")
            ),
            "diarizer_backend": str(
                _as_mapping(audio_cfg.get("diarization", {})).get("provider", "nemo")
            ),
        },
    )
    return _generate_speaker_samples_from_transcript(
        project_root=project_root,
        audio_cfg=audio_cfg,
        transcript_path=transcript_path,
        audio_path=audio_path,
    )


def _find_latest_speaker_samples_manifest(project_root: Path) -> Path | None:
    """Return the most recent speaker-sample manifest when available."""
    candidates = sorted(
        (project_root / "artifacts" / "audio_stage" / "runs").glob(
            "*/speaker_samples/manifest.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


def _find_latest_transcript_path(project_root: Path) -> Path | None:
    """Return the most recent transcript artifact when available."""
    candidates = sorted(
        (project_root / "artifacts" / "transcripts").glob("*.transcript.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


def _resolve_audio_path_candidate(
    *,
    project_root: Path,
    audio_cfg: Mapping[str, Any],
    explicit_audio_path: Path | None,
) -> Path | None:
    """Resolve an available source-audio path for calibration bootstrap."""
    if explicit_audio_path is not None:
        candidate = explicit_audio_path.resolve()
        return candidate if candidate.exists() else None

    configured_audio_path = str(audio_cfg.get("audio_path", "")).strip()
    if not configured_audio_path:
        return None
    candidate = resolve_path(configured_audio_path, base_dir=project_root)
    return candidate if candidate.exists() else None


def _load_speaker_sample_paths(manifest_path: Path) -> dict[str, Path]:
    """Load one reference sample path per speaker from the manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    refs: dict[str, Path] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        speaker = str(sample.get("speaker", "")).strip()
        path_value = str(sample.get("path", "")).strip()
        duration_value = sample.get("duration_ms", 0)
        if not speaker or not path_value:
            continue
        try:
            duration_ms = int(duration_value)
        except (TypeError, ValueError):
            continue
        if duration_ms <= 0:
            continue
        candidate_path = Path(path_value)
        if not candidate_path.is_absolute():
            candidate_path = (manifest_path.parent / candidate_path).resolve()
        if not candidate_path.exists():
            continue
        current = refs.get(speaker)
        if current is None:
            refs[speaker] = candidate_path
    if not refs:
        raise FileNotFoundError(
            "Speaker-sample manifest does not contain any usable sample paths."
        )
    return refs


def _build_preset_default_wpm(
    *,
    speaker_preset_wpm: Mapping[str, Mapping[str, float]],
    preset_names: tuple[str, ...],
) -> dict[str, float]:
    """Build fallback preset WPM values averaged across speakers."""
    defaults: dict[str, float] = {}
    for preset_name in preset_names:
        values = [
            float(preset_map[preset_name])
            for preset_map in speaker_preset_wpm.values()
            if preset_name in preset_map and float(preset_map[preset_name]) > 0
        ]
        if values:
            defaults[preset_name] = round(sum(values) / len(values), 6)
    return defaults


def _utc_now_iso() -> str:
    """Return a compact UTC timestamp for persisted calibration artifacts."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_mapping(raw_value: Any) -> Mapping[str, Any]:
    """Normalize dynamic config values into a plain mapping."""
    return raw_value if isinstance(raw_value, Mapping) else {}


def _is_positive_float(raw_value: Any) -> bool:
    """Return whether a dynamic value can be parsed as a positive float."""
    try:
        return float(raw_value) > 0.0
    except (TypeError, ValueError):
        return False
