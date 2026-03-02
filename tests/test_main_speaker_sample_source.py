from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from orchestration.transcript import Transcript

pytest.importorskip("a2a")
import main as app_main

@dataclass(slots=True, frozen=True)
class _SampleResult:
    output_dir: Path
    manifest_path: Path
    generated_at_utc: str
    artifacts: tuple[object, ...]


class _StubSampleGenerator:
    def generate(
        self,
        *,
        transcript_payload: dict[str, object],
        source_audio_path: Path,
        output_dir: Path,
    ) -> _SampleResult:
        del transcript_payload, source_audio_path
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return _SampleResult(
            output_dir=output_dir,
            manifest_path=manifest_path,
            generated_at_utc="2026-01-01T00:00:00+00:00",
            artifacts=(),
        )


def test_post_transcript_step_enforces_vocals_source_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_source_modes: list[str] = []

    def _capture_source_mode(
        *,
        source_mode: str,
        transcript_metadata: dict[str, object] | None,
        configured_audio_path: str,
        base_dir: Path,
    ) -> Path:
        del transcript_metadata, configured_audio_path, base_dir
        captured_source_modes.append(source_mode)
        vocals_path = tmp_path / "vocals.wav"
        vocals_path.write_bytes(b"vocals")
        return vocals_path

    monkeypatch.setattr(app_main, "resolve_sample_source_audio_path", _capture_source_mode)
    monkeypatch.setattr(
        app_main,
        "build_speaker_sample_generator",
        lambda cfg: _StubSampleGenerator(),
    )
    monkeypatch.setattr(app_main, "write_transcript_atomic", lambda payload, path: None)

    transcript = Transcript.from_mapping(
        {
            "segments": [{"speaker": "SPEAKER_00", "text": "text"}],
            "metadata": {},
        }
    )
    updated_transcript = app_main._run_post_transcript_speaker_sample_step(
        stage_start="audio",
        audio_cfg_dict={
            "audio_path": "audio.wav",
            "work_dir": "artifacts/audio_stage",
            "speaker_samples": {
                "enabled": True,
                "source_audio": "source",
                "output_dir_name": "speaker_samples",
            },
        },
        project_root=tmp_path,
        transcript_path="artifacts/transcripts/latest.transcript.json",
        transcript=transcript,
    )

    assert captured_source_modes == ["vocals"]
    assert (
        updated_transcript.metadata.get("speaker_samples_manifest_path")
        == str(tmp_path / "artifacts" / "audio_stage" / "speaker_samples" / "manifest.json")
    )
