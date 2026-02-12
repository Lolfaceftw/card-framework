"""Tests for IndexTTS2 voice cloning benchmark utilities."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from benchmarks.voice_clone.manifest import load_manifest
from benchmarks.voice_clone.metrics import compute_eer
from benchmarks.voice_clone.runner import run_benchmark
from benchmarks.voice_clone.types import SpeakerEmbedderProtocol, TTSEngineProtocol
from benchmarks.voice_clone.wizard import build_manifest_from_summary


def _write_silent_wav(path: Path, duration_ms: int = 400, sample_rate: int = 16000) -> None:
    """Write a silent mono WAV file for deterministic testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = int(sample_rate * duration_ms / 1000.0)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * sample_count)


def _build_manifest(tmp_path: Path, *, one_speaker: bool = False) -> Path:
    """Create a benchmark manifest and wav fixtures for tests."""
    speaker_rows: list[dict[str, object]] = [
        {
            "speaker_id": "spk_a",
            "prompt_wav": "spk_a_prompt.wav",
            "reference_wav": "spk_a_ref.wav",
            "text": "Hello from speaker A.",
            "use_emo_text": True,
            "emo_text": "Neutral",
            "emo_alpha": 0.6,
        },
        {
            "speaker_id": "spk_b" if not one_speaker else "spk_a",
            "prompt_wav": "spk_b_prompt.wav",
            "reference_wav": "spk_b_ref.wav",
            "text": "Hello from speaker B.",
            "use_emo_text": False,
            "emo_text": "",
            "emo_alpha": 0.5,
        },
    ]
    for filename in [
        "spk_a_prompt.wav",
        "spk_a_ref.wav",
        "spk_b_prompt.wav",
        "spk_b_ref.wav",
    ]:
        _write_silent_wav(tmp_path / filename)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(speaker_rows), encoding="utf-8")
    return manifest_path


class FakeTTSEngine(TTSEngineProtocol):
    """TTS engine stub that writes deterministic WAV files."""

    def infer(
        self,
        *,
        spk_audio_prompt: str,
        text: str,
        output_path: str,
        emo_alpha: float,
        use_emo_text: bool,
        emo_text: str,
        use_random: bool,
        verbose: bool,
    ) -> object:
        del spk_audio_prompt, text, emo_alpha, use_emo_text, emo_text, use_random, verbose
        _write_silent_wav(Path(output_path), duration_ms=500)
        return output_path


class FakeEmbedder(SpeakerEmbedderProtocol):
    """Embedding stub that maps paths to synthetic speaker vectors."""

    def embed(self, wav_path: Path) -> np.ndarray:
        stem = wav_path.stem.lower()
        if "spk_a" in stem:
            return np.array([1.0, 0.0], dtype=np.float64)
        if "spk_b" in stem:
            return np.array([0.0, 1.0], dtype=np.float64)
        if "speaker_00" in stem:
            return np.array([1.0, 0.0], dtype=np.float64)
        if "speaker_01" in stem:
            return np.array([0.0, 1.0], dtype=np.float64)
        return np.array([0.5, 0.5], dtype=np.float64)


def test_load_manifest_resolves_paths(tmp_path: Path) -> None:
    """Resolve relative paths and normalize optional fields."""
    manifest_path = _build_manifest(tmp_path)
    items = load_manifest(manifest_path)
    assert len(items) == 2
    assert items[0].prompt_wav.exists()
    assert items[0].reference_wav.exists()
    assert 0.0 <= items[0].emo_alpha <= 1.0


def test_compute_eer_separable_scores_is_zero() -> None:
    """Return zero EER for perfectly separable scores."""
    scores = np.array([0.95, 0.93, 0.15, 0.1], dtype=np.float64)
    labels = np.array([True, True, False, False], dtype=np.bool_)
    assert compute_eer(scores, labels) == 0.0


def test_compute_eer_single_class_returns_none() -> None:
    """Return None when EER is undefined."""
    scores = np.array([0.8, 0.7], dtype=np.float64)
    labels = np.array([True, True], dtype=np.bool_)
    assert compute_eer(scores, labels) is None


def test_run_benchmark_writes_artifacts(tmp_path: Path) -> None:
    """Produce benchmark artifacts with fake dependencies."""
    manifest_path = _build_manifest(tmp_path)
    output_dir = tmp_path / "out"

    artifacts = run_benchmark(
        manifest_path=manifest_path,
        cfg_path=tmp_path / "cfg.yaml",
        model_dir=tmp_path,
        device="cpu",
        output_dir=output_dir,
        max_items=None,
        seed=7,
        prepare_mos_kit=True,
        tts_engine=FakeTTSEngine(),
        embedder=FakeEmbedder(),
    )

    assert artifacts.metrics_json.exists()
    assert artifacts.pair_scores_csv.exists()
    assert artifacts.run_log.exists()
    assert artifacts.mos_dir is not None
    assert (artifacts.mos_dir / "mos_pairs.csv").exists()
    assert (artifacts.mos_dir / "ratings_template.csv").exists()

    summary = json.loads(artifacts.metrics_json.read_text(encoding="utf-8"))
    assert summary["dataset"]["item_count"] == 2
    assert summary["metrics"]["top1_speaker_acc"] == 1.0


def test_run_benchmark_single_speaker_eer_none(tmp_path: Path) -> None:
    """Set ASV EER to null when only one speaker is present."""
    manifest_path = _build_manifest(tmp_path, one_speaker=True)
    artifacts = run_benchmark(
        manifest_path=manifest_path,
        cfg_path=tmp_path / "cfg.yaml",
        model_dir=tmp_path,
        device="cpu",
        output_dir=tmp_path / "single_speaker_run",
        max_items=None,
        seed=11,
        prepare_mos_kit=False,
        tts_engine=FakeTTSEngine(),
        embedder=FakeEmbedder(),
    )
    summary = json.loads(artifacts.metrics_json.read_text(encoding="utf-8"))
    assert summary["dataset"]["speaker_count"] == 1
    assert summary["metrics"]["asv_eer"] is None


def test_build_manifest_from_summary_uses_holdout_when_available(tmp_path: Path) -> None:
    """Generate manifest rows from summary JSON and holdout directory."""
    summary_payload = [
        {
            "speaker": "SPEAKER_00",
            "voice_sample": "SPEAKER_00.wav",
            "text": "Line A",
            "use_emo_text": True,
            "emo_text": "calm",
            "emo_alpha": 0.6,
        },
        {
            "speaker": "SPEAKER_01",
            "voice_sample": "SPEAKER_01.wav",
            "text": "Line B",
            "use_emo_text": False,
            "emo_text": "",
            "emo_alpha": 0.4,
        },
    ]
    summary_path = tmp_path / "sample_summary.json"
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
    _write_silent_wav(tmp_path / "SPEAKER_00.wav")
    _write_silent_wav(tmp_path / "SPEAKER_01.wav")

    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    _write_silent_wav(holdout_dir / "SPEAKER_00.wav")

    manifest_path = tmp_path / "generated_manifest.json"
    result = build_manifest_from_summary(
        summary_json_path=summary_path,
        manifest_path=manifest_path,
        holdout_dir=holdout_dir,
        max_items=None,
    )

    manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result.item_count == 2
    assert result.fallback_reference_count == 1
    assert Path(manifest_rows[0]["reference_wav"]).resolve() == (
        holdout_dir / "SPEAKER_00.wav"
    ).resolve()
    assert Path(manifest_rows[1]["reference_wav"]).resolve() == (
        tmp_path / "SPEAKER_01.wav"
    ).resolve()
