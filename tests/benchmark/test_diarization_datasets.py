from __future__ import annotations

import json
from pathlib import Path

from benchmark.diarization_datasets import prepare_ami_manifest


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.text = payload.decode("utf-8")

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        del exc_type, exc, exc_tb


class _FakeSession:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.requested_urls: list[str] = []

    def get(self, url: str, *, stream: bool = False, timeout=None) -> _FakeResponse:
        del stream, timeout
        self.requested_urls.append(url)
        return _FakeResponse(self.payloads[url])

    def close(self) -> None:
        return None


def test_prepare_ami_manifest_downloads_assets_and_writes_manifest(
    tmp_path: Path,
) -> None:
    payloads = {
        (
            "https://raw.githubusercontent.com/BUTSpeechFIT/"
            "AMI-diarization-setup/main/lists/test.meetings.txt"
        ): b"ES2004a\nES2004b\n",
        (
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
            "ES2004a/audio/ES2004a.Mix-Headset.wav"
        ): b"audio-a",
        (
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
            "ES2004b/audio/ES2004b.Mix-Headset.wav"
        ): b"audio-b",
        (
            "https://raw.githubusercontent.com/BUTSpeechFIT/"
            "AMI-diarization-setup/main/only_words/rttms/test/ES2004a.rttm"
        ): (
            b"SPEAKER ES2004a 1 0.000 1.000 <NA> <NA> speaker_a <NA> <NA>\n"
            b"SPEAKER ES2004a 1 1.000 1.000 <NA> <NA> speaker_b <NA> <NA>\n"
        ),
        (
            "https://raw.githubusercontent.com/BUTSpeechFIT/"
            "AMI-diarization-setup/main/only_words/rttms/test/ES2004b.rttm"
        ): b"SPEAKER ES2004b 1 0.000 2.000 <NA> <NA> speaker_c <NA> <NA>\n",
        (
            "https://raw.githubusercontent.com/BUTSpeechFIT/"
            "AMI-diarization-setup/main/uems/test/ES2004a.uem"
        ): b"ES2004a 1 0.000 2.000\n",
        (
            "https://raw.githubusercontent.com/BUTSpeechFIT/"
            "AMI-diarization-setup/main/uems/test/ES2004b.uem"
        ): b"ES2004b 1 0.000 2.000\n",
    }
    session = _FakeSession(payloads)
    output_path = tmp_path / "manifests" / "diarization_ami_test.json"
    data_root = tmp_path / "data" / "ami"

    manifest = prepare_ami_manifest(
        output_path=output_path,
        data_root=data_root,
        subset="test",
        num_samples=2,
        force_download=False,
        session=session,
    )

    assert output_path.exists()
    stored_manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert stored_manifest["source"] == "ami"
    assert len(stored_manifest["samples"]) == 2
    assert stored_manifest["samples"][0]["sample_id"] == "ES2004a"
    assert stored_manifest["samples"][0]["num_speakers"] == 2
    assert Path(stored_manifest["samples"][0]["audio_filepath"]).exists()
    assert Path(stored_manifest["samples"][0]["rttm_filepath"]).exists()
    assert Path(stored_manifest["samples"][0]["uem_filepath"]).exists()
    assert manifest["samples"][1]["sample_id"] == "ES2004b"

    expected_audio_path = data_root / "audio" / "test" / "ES2004a.Mix-Headset.wav"
    assert expected_audio_path.read_bytes() == b"audio-a"
    assert session.requested_urls[0].endswith("lists/test.meetings.txt")
