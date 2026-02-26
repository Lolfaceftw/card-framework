import json

from audio_pipeline.contracts import TranscriptSegment
from audio_pipeline.io import build_transcript_payload, write_transcript_atomic


def test_write_transcript_atomic_persists_payload(tmp_path) -> None:
    payload = build_transcript_payload(
        segments=[
            TranscriptSegment(
                speaker="SPEAKER_00",
                start_time=0,
                end_time=500,
                text="hello",
            )
        ],
        metadata={"source_audio_path": "input.wav"},
    )
    output_path = tmp_path / "transcript.json"

    write_transcript_atomic(payload, output_path)

    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["segments"][0]["speaker"] == "SPEAKER_00"
    assert loaded["segments"][0]["start_time"] == 0
