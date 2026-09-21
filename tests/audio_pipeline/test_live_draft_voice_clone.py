"""Regression tests for live-draft voice-clone state persistence."""

from __future__ import annotations

import json
from pathlib import Path

from card_framework.audio_pipeline.live_draft_voice_clone import (
    LiveDraftVoiceCloneSession,
)


class _FakeVoiceCloneProvider:
    """Write placeholder audio files for live-draft session tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None = None,
        emotion_vector=None,
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


def test_render_turn_persists_state_sidecar_before_snapshot_sync(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Persist live-draft turn metadata immediately after one render."""
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

    def _fake_probe_audio_duration_ms(_path: Path) -> int:
        return 2400

    import card_framework.audio_pipeline.live_draft_voice_clone as live_draft_module

    monkeypatch.setattr(
        live_draft_module,
        "probe_audio_duration_ms",
        _fake_probe_audio_duration_ms,
    )

    state_path = tmp_path / "voice_clone" / "live_draft_test.state.json"
    session = LiveDraftVoiceCloneSession(
        provider=_FakeVoiceCloneProvider(),
        state_path=state_path,
        output_dir=state_path.parent,
        speaker_samples_manifest_path=manifest_path,
    )

    record = session.render_turn(
        turn_id="turn-001",
        speaker="SPEAKER_00",
        text="Hello there.",
        emo_preset="neutral",
    )

    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["current_snapshot"] == []
    assert payload["turn_audio"]["turn-001"]["output_audio_path"] == str(
        record.output_audio_path
    )
    assert payload["turn_audio"]["turn-001"]["duration_ms"] == 2400
    assert payload["synthesis_count"] == 1

    resumed = LiveDraftVoiceCloneSession(
        provider=_FakeVoiceCloneProvider(),
        state_path=state_path,
        output_dir=state_path.parent,
        speaker_samples_manifest_path=manifest_path,
    )
    resumed.render_turn(
        turn_id="turn-002",
        speaker="SPEAKER_00",
        text="A second turn.",
        emo_preset="neutral",
    )

    resumed_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert resumed_payload["synthesis_count"] == 2

    resumed.bootstrap_from_snapshot(
        [
            {
                "line": 1,
                "turn_id": "stage3-turn-001",
                "speaker_id": "SPEAKER_00",
                "content": "The final rerender.",
                "emo_preset": "neutral",
            }
        ]
    )

    final_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert final_payload["synthesis_count"] == 3
    assert list(final_payload["turn_audio"]) == ["stage3-turn-001"]


def test_restore_snapshot_matches_summary_xml_with_link_metadata(
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

    def _fake_probe_audio_duration_ms(_path: Path) -> int:
        return 2400

    import card_framework.audio_pipeline.live_draft_voice_clone as live_draft_module

    monkeypatch.setattr(
        live_draft_module,
        "probe_audio_duration_ms",
        _fake_probe_audio_duration_ms,
    )

    state_path = tmp_path / "voice_clone" / "live_draft_test.state.json"
    session = LiveDraftVoiceCloneSession(
        provider=_FakeVoiceCloneProvider(),
        state_path=state_path,
        output_dir=state_path.parent,
        speaker_samples_manifest_path=manifest_path,
    )

    session.render_turn(
        turn_id="turn-001",
        speaker="SPEAKER_00",
        text="Hello there.",
        emo_preset="neutral",
        emotion_vector_source="summary_turn_linking",
        source_segment_indices=(2, 5),
        source_segment_weights=(0.7, 0.3),
    )
    snapshot = [
        {
            "line": 1,
            "turn_id": "turn-001",
            "speaker_id": "SPEAKER_00",
            "content": "Hello there.",
            "emo_preset": "neutral",
            "emotion_vector_source": "summary_turn_linking",
            "source_segment_indices": [2, 5],
            "source_segment_weights": [0.7, 0.3],
        }
    ]
    session.sync_snapshot(snapshot)

    restored = session.restore_snapshot_for_draft(
        '<SPEAKER_00 emo_preset="neutral" emotion_vector_source="summary_turn_linking" '
        'source_segment_indices="2,5" source_segment_weights="0.700000,0.300000">'
        "Hello there."
        "</SPEAKER_00>"
    )

    assert restored is not None
    assert restored[0]["source_segment_indices"] == [2, 5]
    assert restored[0]["source_segment_weights"] == [0.7, 0.3]


def test_restore_snapshot_ignores_emo_preset_when_summary_xml_omits_it(
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

    def _fake_probe_audio_duration_ms(_path: Path) -> int:
        return 2400

    import card_framework.audio_pipeline.live_draft_voice_clone as live_draft_module

    monkeypatch.setattr(
        live_draft_module,
        "probe_audio_duration_ms",
        _fake_probe_audio_duration_ms,
    )

    state_path = tmp_path / "voice_clone" / "live_draft_test.state.json"
    session = LiveDraftVoiceCloneSession(
        provider=_FakeVoiceCloneProvider(),
        state_path=state_path,
        output_dir=state_path.parent,
        speaker_samples_manifest_path=manifest_path,
    )

    session.render_turn(
        turn_id="turn-001",
        speaker="SPEAKER_00",
        text="Hello there.",
        emo_preset="warm",
        emotion_vector=[0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24],
        emotion_vector_source="manual_override",
    )
    session.sync_snapshot(
        [
            {
                "line": 1,
                "turn_id": "turn-001",
                "speaker_id": "SPEAKER_00",
                "content": "Hello there.",
                "emo_preset": "warm",
                "emotion_vector": [0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24],
                "emotion_vector_source": "manual_override",
            }
        ]
    )

    restored = session.restore_snapshot_for_draft(
        '<SPEAKER_00 emotion_categories="calm, happy, surprised" '
        'emotion_vector="0.140000,0.000000,0.000000,0.000000,0.000000,0.000000,0.020000,0.240000" '
        'emotion_vector_source="manual_override">'
        "Hello there."
        "</SPEAKER_00>"
    )

    assert restored is not None
    assert restored[0]["emo_preset"] == "warm"


def test_live_draft_render_turn_uses_explicit_vector_and_records_actual_source(
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

    def _fake_probe_audio_duration_ms(_path: Path) -> int:
        return 2400

    import card_framework.audio_pipeline.live_draft_voice_clone as live_draft_module

    monkeypatch.setattr(
        live_draft_module,
        "probe_audio_duration_ms",
        _fake_probe_audio_duration_ms,
    )

    provider = _FakeVoiceCloneProvider()
    session = LiveDraftVoiceCloneSession(
        provider=provider,
        state_path=tmp_path / "voice_clone" / "state.json",
        output_dir=tmp_path / "voice_clone",
        speaker_samples_manifest_path=manifest_path,
    )

    record = session.render_turn(
        turn_id="turn-001",
        speaker="SPEAKER_00",
        text="Hello there.",
        emo_preset="warm",
        emotion_vector=[0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24],
        emotion_vector_source="manual_override",
    )

    assert provider.calls[0]["emo_text"] is None
    assert provider.calls[0]["emotion_vector"] is not None
    assert record.emotion_vector_source == "manual_override"


def test_live_draft_render_turn_disables_emo_text_fallback_by_default(
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

    def _fake_probe_audio_duration_ms(_path: Path) -> int:
        return 2400

    import card_framework.audio_pipeline.live_draft_voice_clone as live_draft_module

    monkeypatch.setattr(
        live_draft_module,
        "probe_audio_duration_ms",
        _fake_probe_audio_duration_ms,
    )

    provider = _FakeVoiceCloneProvider()
    session = LiveDraftVoiceCloneSession(
        provider=provider,
        state_path=tmp_path / "voice_clone" / "state.json",
        output_dir=tmp_path / "voice_clone",
        speaker_samples_manifest_path=manifest_path,
    )

    record = session.render_turn(
        turn_id="turn-001",
        speaker="SPEAKER_00",
        text="Hello there.",
        emo_preset="warm",
    )

    assert provider.calls[0]["emotion_vector"] is None
    assert provider.calls[0]["emo_text"] is None
    assert record.emotion_vector_source == ""
