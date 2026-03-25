"""Tests for stage-3 voice-clone vector and normalization behavior."""

from __future__ import annotations

import json
from pathlib import Path

from card_framework.audio_pipeline.audio_normalization import AudioNormalizationPolicy
import card_framework.audio_pipeline.voice_clone_orchestrator as voice_clone_module
from card_framework.audio_pipeline.voice_clone_orchestrator import VoiceCloneOrchestrator
from card_framework.shared.emotion_vector import EmotionVector


class _FakeVoiceCloneProvider:
    """Record synthesize calls and create placeholder output files."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None = None,
        emotion_vector: EmotionVector | None = None,
        progress_callback=None,
    ) -> Path:
        del reference_audio_path, progress_callback
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        output_audio_path.write_bytes(text.encode("utf-8"))
        self.calls.append(
            {
                "text": text,
                "emo_text": emo_text,
                "emotion_vector": emotion_vector,
                "output_audio_path": output_audio_path,
            }
        )
        return output_audio_path

    def close(self) -> None:
        return None


def test_voice_clone_orchestrator_uses_preset_vector_and_normalizes_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample_dir = tmp_path / "speaker_samples"
    sample_dir.mkdir()
    sample_path = sample_dir / "speaker.wav"
    sample_path.write_bytes(b"sample")
    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "speaker": "SPEAKER_00",
                        "path": str(sample_path),
                        "duration_ms": 1000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    normalized_paths: list[Path] = []

    def _fake_merge(
        *,
        artifact_paths,
        output_path,
        audio_codec,
        timeout_seconds,
    ) -> None:
        del artifact_paths, audio_codec, timeout_seconds
        output_path.write_bytes(b"merged")

    def _fake_normalize(*, audio_path: Path, policy: AudioNormalizationPolicy | None) -> Path:
        assert policy is not None
        assert policy.enabled is True
        normalized_paths.append(audio_path)
        return audio_path

    monkeypatch.setattr(voice_clone_module, "merge_audio_artifacts_to_wav", _fake_merge)
    monkeypatch.setattr(voice_clone_module, "normalize_audio_file", _fake_normalize)

    preset_vector = EmotionVector.from_raw([0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24])
    provider = _FakeVoiceCloneProvider()
    orchestrator = VoiceCloneOrchestrator(
        provider=provider,
        output_dir=tmp_path / "voice_clone",
        emo_preset_catalog={"warm": "warmly"},
        emo_preset_vector_catalog={"warm": preset_vector},
        merged_output_normalization=AudioNormalizationPolicy(enabled=True),
    )

    result = orchestrator.run(
        summary_xml=(
            '<SPEAKER_00 emo_preset="warm">'
            "hello there"
            "</SPEAKER_00>"
        ),
        speaker_samples_manifest_path=manifest_path,
    )

    assert provider.calls[0]["emotion_vector"] == preset_vector
    assert provider.calls[0]["emo_text"] is None
    assert normalized_paths == [result.merged_output_audio_path]
    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["artifacts"][0]["emotion_vector"] == preset_vector.to_list()
    assert manifest_payload["artifacts"][0]["emotion_vector_source"] == "preset_vector"
    assert manifest_payload["artifacts"][0]["source_segment_indices"] == []
    assert manifest_payload["artifacts"][0]["source_segment_weights"] == []


def test_voice_clone_orchestrator_disables_emo_text_fallback_by_default(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "speaker_samples"
    sample_dir.mkdir()
    sample_path = sample_dir / "speaker.wav"
    sample_path.write_bytes(b"sample")
    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "speaker": "SPEAKER_00",
                        "path": str(sample_path),
                        "duration_ms": 1000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    provider = _FakeVoiceCloneProvider()
    orchestrator = VoiceCloneOrchestrator(
        provider=provider,
        output_dir=tmp_path / "voice_clone",
        emo_preset_catalog={"warm": "warmly"},
        merge_segments=False,
    )

    orchestrator.run(
        summary_xml='<SPEAKER_00 emo_preset="warm">hello there</SPEAKER_00>',
        speaker_samples_manifest_path=manifest_path,
    )

    assert provider.calls[0]["emotion_vector"] is None
    assert provider.calls[0]["emo_text"] is None


def test_voice_clone_orchestrator_allows_opt_in_emo_text_fallback(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "speaker_samples"
    sample_dir.mkdir()
    sample_path = sample_dir / "speaker.wav"
    sample_path.write_bytes(b"sample")
    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "speaker": "SPEAKER_00",
                        "path": str(sample_path),
                        "duration_ms": 1000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    provider = _FakeVoiceCloneProvider()
    orchestrator = VoiceCloneOrchestrator(
        provider=provider,
        output_dir=tmp_path / "voice_clone",
        emo_preset_catalog={"warm": "warmly"},
        emo_text_fallback_enabled=True,
        merge_segments=False,
    )

    result = orchestrator.run(
        summary_xml='<SPEAKER_00 emo_preset="warm">hello there</SPEAKER_00>',
        speaker_samples_manifest_path=manifest_path,
    )

    assert provider.calls[0]["emotion_vector"] is None
    assert provider.calls[0]["emo_text"] == "warmly"
    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["artifacts"][0]["emotion_vector_source"] == "emo_text"
