"""Shared types for voice cloning benchmark workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np


class RawManifestItem(TypedDict, total=False):
    """Represent one manifest row before validation."""

    speaker_id: str
    prompt_wav: str
    reference_wav: str
    text: str
    use_emo_text: bool
    emo_text: str
    emo_alpha: float


@dataclass(slots=True, frozen=True)
class ManifestItem:
    """Represent one validated benchmark manifest row."""

    item_id: int
    speaker_id: str
    prompt_wav: Path
    reference_wav: Path
    text: str
    use_emo_text: bool
    emo_text: str
    emo_alpha: float


@dataclass(slots=True, frozen=True)
class PairScoreRow:
    """Represent a per-utterance benchmark score row."""

    item_id: int
    speaker_id: str
    prompt_wav: Path
    reference_wav: Path
    generated_wav: Path
    same_item_cosine: float
    predicted_speaker: str
    top1_correct: bool


@dataclass(slots=True, frozen=True)
class TextScoreRow:
    """Represent a per-utterance text-fidelity score row."""

    item_id: int
    speaker_id: str
    generated_wav: Path
    reference_text: str
    hypothesis_text: str
    wer: float
    cer: float


@dataclass(slots=True, frozen=True)
class BenchmarkArtifacts:
    """Represent paths of generated benchmark artifacts."""

    output_dir: Path
    generated_dir: Path
    pair_scores_csv: Path
    metrics_json: Path
    run_log: Path
    mos_dir: Path | None


class TTSEngineProtocol(Protocol):
    """Define TTS backend interface used by the benchmark."""

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
        """Synthesize one utterance."""


class SpeakerEmbedderProtocol(Protocol):
    """Define speaker embedding extractor interface."""

    def embed(self, wav_path: Path) -> np.ndarray:
        """Return one speaker embedding vector."""


class SpeechTranscriberProtocol(Protocol):
    """Define ASR backend interface used for text-fidelity evaluation."""

    def transcribe(self, wav_path: Path) -> str:
        """Return decoded transcript text for one WAV file."""
