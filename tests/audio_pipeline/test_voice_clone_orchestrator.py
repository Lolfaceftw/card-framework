import json
from pathlib import Path

import pytest

from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.voice_clone_orchestrator import VoiceCloneOrchestrator


class _StubVoiceCloneProvider:
    """Simple provider used to capture reference selection in tests."""

    def __init__(self) -> None:
        self.reference_calls: list[Path] = []

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
    ) -> Path:
        del text
        self.reference_calls.append(reference_audio_path)
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        output_audio_path.write_bytes(b"wav")
        return output_audio_path


def _write_speaker_manifest(tmp_path: Path, samples: list[dict[str, object]]) -> Path:
    manifest_path = tmp_path / "speaker_samples_manifest.json"
    payload = {"samples": samples}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_voice_clone_orchestrator_generates_turn_artifacts(tmp_path: Path) -> None:
    sample_0 = tmp_path / "speaker_00.wav"
    sample_1 = tmp_path / "speaker_01.wav"
    sample_0.write_bytes(b"sample0")
    sample_1.write_bytes(b"sample1")
    manifest_path = _write_speaker_manifest(
        tmp_path,
        samples=[
            {"speaker": "SPEAKER_00", "path": str(sample_0), "duration_ms": 30_000},
            {"speaker": "SPEAKER_01", "path": str(sample_1), "duration_ms": 15_000},
        ],
    )
    provider = _StubVoiceCloneProvider()
    orchestrator = VoiceCloneOrchestrator(
        provider=provider,
        output_dir=tmp_path / "voice_clone",
        fail_on_error=True,
    )

    result = orchestrator.run(
        summary_xml=(
            "<SPEAKER_00>Hello there</SPEAKER_00>"
            "<SPEAKER_01>General Kenobi</SPEAKER_01>"
        ),
        speaker_samples_manifest_path=manifest_path,
    )

    assert len(result.artifacts) == 2
    assert all(artifact.output_audio_path.exists() for artifact in result.artifacts)
    assert result.manifest_path.exists()
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_count"] == 2


def test_voice_clone_orchestrator_raises_when_speaker_sample_missing(tmp_path: Path) -> None:
    sample_0 = tmp_path / "speaker_00.wav"
    sample_0.write_bytes(b"sample0")
    manifest_path = _write_speaker_manifest(
        tmp_path,
        samples=[
            {"speaker": "SPEAKER_00", "path": str(sample_0), "duration_ms": 30_000},
        ],
    )
    orchestrator = VoiceCloneOrchestrator(
        provider=_StubVoiceCloneProvider(),
        output_dir=tmp_path / "voice_clone",
        fail_on_error=True,
    )

    with pytest.raises(NonRetryableAudioStageError, match="Missing speaker sample"):
        orchestrator.run(
            summary_xml="<SPEAKER_01>Missing sample</SPEAKER_01>",
            speaker_samples_manifest_path=manifest_path,
        )


def test_voice_clone_orchestrator_prefers_longest_reference_sample(tmp_path: Path) -> None:
    sample_short = tmp_path / "speaker_00_short.wav"
    sample_long = tmp_path / "speaker_00_long.wav"
    sample_short.write_bytes(b"short")
    sample_long.write_bytes(b"long")
    manifest_path = _write_speaker_manifest(
        tmp_path,
        samples=[
            {"speaker": "SPEAKER_00", "path": str(sample_short), "duration_ms": 8_000},
            {"speaker": "SPEAKER_00", "path": str(sample_long), "duration_ms": 30_000},
        ],
    )
    provider = _StubVoiceCloneProvider()
    orchestrator = VoiceCloneOrchestrator(
        provider=provider,
        output_dir=tmp_path / "voice_clone",
        fail_on_error=True,
    )

    orchestrator.run(
        summary_xml="<SPEAKER_00>Longest should win</SPEAKER_00>",
        speaker_samples_manifest_path=manifest_path,
    )

    assert provider.reference_calls == [sample_long]
