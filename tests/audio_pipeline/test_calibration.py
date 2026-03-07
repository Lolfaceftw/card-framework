"""Regression tests for voice-clone calibration reuse behavior."""

import json
from pathlib import Path

import audio_pipeline.calibration as calibration


class _StubVoiceCloneProvider:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None = None,
        progress_callback=None,
    ) -> Path:
        del text, emo_text, progress_callback
        self.calls.append(reference_audio_path)
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        output_audio_path.write_bytes(b"wav")
        return output_audio_path


def _write_manifest(
    tmp_path: Path,
    *,
    directory_name: str,
    speakers: list[str],
) -> Path:
    manifest_dir = tmp_path / directory_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    for speaker in speakers:
        sample_path = manifest_dir / f"{speaker}.wav"
        sample_path.write_bytes(f"sample-{speaker}".encode("utf-8"))
        samples.append(
            {
                "speaker": speaker,
                "path": str(sample_path),
                "duration_ms": 30_000,
            }
        )
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"samples": samples}, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _write_calibration_artifact(
    tmp_path: Path,
    *,
    manifest_path: Path,
    speaker_preset_wpm: dict[str, dict[str, float]],
) -> Path:
    artifact_path = (
        tmp_path / "artifacts" / "calibration" / "voice_clone_calibration.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-03-07T00:00:00+00:00",
                "speaker_samples_manifest_path": str(manifest_path),
                "preset_emo_texts": {"neutral": "steady"},
                "calibration_phrases": list(calibration.DEFAULT_CALIBRATION_PHRASES),
                "speaker_preset_wpm": speaker_preset_wpm,
                "preset_default_wpm": {"neutral": 180.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_path


def _build_audio_cfg() -> dict[str, object]:
    return {
        "voice_clone": {
            "calibration": {
                "artifact_path": "artifacts/calibration/voice_clone_calibration.json",
            },
            "emo_presets": {"neutral": "steady"},
        }
    }


def test_ensure_voice_clone_calibration_reuses_matching_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        directory_name="speaker_samples_current",
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )
    _write_calibration_artifact(
        tmp_path,
        manifest_path=manifest_path,
        speaker_preset_wpm={
            "SPEAKER_00": {"neutral": 180.0},
            "SPEAKER_01": {"neutral": 190.0},
        },
    )

    monkeypatch.setattr(
        calibration,
        "build_voice_clone_provider",
        lambda audio_cfg, project_root: (_ for _ in ()).throw(
            AssertionError("matching calibration should be reused")
        ),
    )

    result = calibration.ensure_voice_clone_calibration(
        project_root=tmp_path,
        audio_cfg=_build_audio_cfg(),
        speaker_samples_manifest_path=manifest_path,
    )

    assert result.speaker_samples_manifest_path == manifest_path.resolve()
    assert set(result.speaker_preset_wpm) == {"SPEAKER_00", "SPEAKER_01"}


def test_ensure_voice_clone_calibration_rebuilds_when_manifest_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    old_manifest_path = _write_manifest(
        tmp_path,
        directory_name="speaker_samples_old",
        speakers=["SPEAKER_00"],
    )
    new_manifest_path = _write_manifest(
        tmp_path,
        directory_name="speaker_samples_new",
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )
    _write_calibration_artifact(
        tmp_path,
        manifest_path=old_manifest_path,
        speaker_preset_wpm={"SPEAKER_00": {"neutral": 180.0}},
    )

    provider = _StubVoiceCloneProvider()
    monkeypatch.setattr(
        calibration,
        "build_voice_clone_provider",
        lambda audio_cfg, project_root: provider,
    )
    monkeypatch.setattr(
        calibration,
        "probe_audio_duration_ms",
        lambda path: 1_000,
    )

    result = calibration.ensure_voice_clone_calibration(
        project_root=tmp_path,
        audio_cfg=_build_audio_cfg(),
        speaker_samples_manifest_path=new_manifest_path,
    )

    assert result.speaker_samples_manifest_path == new_manifest_path.resolve()
    assert set(result.speaker_preset_wpm) == {"SPEAKER_00", "SPEAKER_01"}
    assert len(provider.calls) == 6


def test_ensure_voice_clone_calibration_rebuilds_when_artifact_misses_speaker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        directory_name="speaker_samples_current",
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )
    _write_calibration_artifact(
        tmp_path,
        manifest_path=manifest_path,
        speaker_preset_wpm={"SPEAKER_00": {"neutral": 180.0}},
    )

    provider = _StubVoiceCloneProvider()
    monkeypatch.setattr(
        calibration,
        "build_voice_clone_provider",
        lambda audio_cfg, project_root: provider,
    )
    monkeypatch.setattr(
        calibration,
        "probe_audio_duration_ms",
        lambda path: 1_000,
    )

    result = calibration.ensure_voice_clone_calibration(
        project_root=tmp_path,
        audio_cfg=_build_audio_cfg(),
        speaker_samples_manifest_path=manifest_path,
    )

    assert result.speaker_samples_manifest_path == manifest_path.resolve()
    assert set(result.speaker_preset_wpm) == {"SPEAKER_00", "SPEAKER_01"}
    assert len(provider.calls) == 6
