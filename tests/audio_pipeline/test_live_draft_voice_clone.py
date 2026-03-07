"""Regression tests for incremental live-draft voice cloning."""

from __future__ import annotations

import json
from pathlib import Path

from audio_pipeline.live_draft_voice_clone import LiveDraftVoiceCloneSession
from audio_pipeline.voice_clone_orchestrator import VoiceCloneOrchestrator


class _StubVoiceCloneProvider:
    """Write placeholder WAV payloads and capture chosen reference samples."""

    def __init__(self) -> None:
        self.reference_calls: list[Path] = []

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
        self.reference_calls.append(reference_audio_path)
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        output_audio_path.write_bytes(b"wav")
        return output_audio_path


def _write_speaker_manifest(tmp_path: Path) -> Path:
    """Persist a minimal speaker-sample manifest for live-draft tests."""
    sample_0 = tmp_path / "speaker_00.wav"
    sample_1 = tmp_path / "speaker_01.wav"
    sample_0.write_bytes(b"sample0")
    sample_1.write_bytes(b"sample1")
    manifest_path = tmp_path / "speaker_samples_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "speaker": "SPEAKER_00",
                        "path": str(sample_0),
                        "duration_ms": 30_000,
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "path": str(sample_1),
                        "duration_ms": 20_000,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _build_session(tmp_path: Path, provider: _StubVoiceCloneProvider) -> LiveDraftVoiceCloneSession:
    """Build a live-draft session from a real voice-clone orchestrator."""
    orchestrator = VoiceCloneOrchestrator(
        provider=provider,
        output_dir=tmp_path / "voice_clone",
        merge_segments=False,
    )
    return LiveDraftVoiceCloneSession.from_orchestrator(
        orchestrator=orchestrator,
        state_path=tmp_path / "voice_clone" / "live_draft.state.json",
        speaker_samples_manifest_path=_write_speaker_manifest(tmp_path),
    )


def test_live_draft_session_bootstraps_and_finalizes_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Persist live turn audio, expose actual durations, and finalize stage-3 artifacts."""
    provider = _StubVoiceCloneProvider()
    session = _build_session(tmp_path, provider)
    snapshot = [
        {
            "line": 1,
            "turn_id": "turn-001",
            "speaker_id": "SPEAKER_00",
            "content": "hello there",
            "emo_preset": "neutral",
        },
        {
            "line": 2,
            "turn_id": "turn-002",
            "speaker_id": "SPEAKER_01",
            "content": "general kenobi",
            "emo_preset": "warm",
        },
    ]

    monkeypatch.setattr(
        "audio_pipeline.live_draft_voice_clone.probe_audio_duration_ms",
        lambda path: 2_000 if "turn-001" in path.name else 3_000,
    )

    session.bootstrap_from_snapshot(snapshot)
    breakdown = session.get_duration_breakdown()

    assert breakdown["duration_source"] == "actual_audio"
    assert breakdown["total_estimated_seconds"] == 5.0
    assert breakdown["messages"][0]["duration_ms"] == 2_000
    assert breakdown["messages"][1]["duration_ms"] == 3_000

    restored = session.restore_snapshot_for_draft(
        "<SPEAKER_00>hello there</SPEAKER_00>"
        '<SPEAKER_01 emo_preset="warm">general kenobi</SPEAKER_01>'
    )
    assert restored == snapshot

    result = session.finalize()
    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest_payload["artifact_count"] == 2
    assert manifest_payload["artifacts"][0]["turn_id"] == "turn-001"
    assert manifest_payload["artifacts"][0]["duration_ms"] == 2_000
    assert manifest_payload["artifacts"][0]["word_count"] == 2
    assert manifest_payload["artifacts"][0]["actual_wpm"] == 60.0
    assert result.merged_output_audio_path is None
    assert len(provider.reference_calls) == 2


def test_live_draft_session_remove_turn_deletes_audio_and_excludes_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Drop removed turns from state, breakdown totals, and the finalized manifest."""
    provider = _StubVoiceCloneProvider()
    session = _build_session(tmp_path, provider)
    snapshot = [
        {
            "line": 1,
            "turn_id": "turn-001",
            "speaker_id": "SPEAKER_00",
            "content": "hello there",
            "emo_preset": "neutral",
        },
        {
            "line": 2,
            "turn_id": "turn-002",
            "speaker_id": "SPEAKER_01",
            "content": "general kenobi",
            "emo_preset": "neutral",
        },
    ]
    removed_path = tmp_path / "voice_clone" / "live_draft_turns" / "turn-001.wav"

    monkeypatch.setattr(
        "audio_pipeline.live_draft_voice_clone.probe_audio_duration_ms",
        lambda path: 2_000 if "turn-001" in path.name else 3_000,
    )

    session.bootstrap_from_snapshot(snapshot)
    assert removed_path.exists()

    session.remove_turn("turn-001")
    session.sync_snapshot([snapshot[1]])
    breakdown = session.get_duration_breakdown()
    result = session.finalize()
    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert not removed_path.exists()
    assert breakdown["total_estimated_seconds"] == 3.0
    assert manifest_payload["artifact_count"] == 1
    assert manifest_payload["artifacts"][0]["turn_id"] == "turn-002"
