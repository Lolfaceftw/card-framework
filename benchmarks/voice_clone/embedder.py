"""Speaker embedding backends for benchmark scoring."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from benchmarks.voice_clone.constants import WAVLM_SPEAKER_MODEL_ID

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class WavLMSpeakerEmbedder:
    """Extract normalized speaker embeddings using WavLM x-vector head."""

    def __init__(self, *, model_id: str = WAVLM_SPEAKER_MODEL_ID, device: str) -> None:
        """Initialize embedder with lazy model loading.

        Args:
            model_id: Hugging Face model id for x-vector extraction.
            device: Runtime torch device string.
        """
        self._model_id = model_id
        self._device = device
        self._feature_extractor: object | None = None
        self._model: object | None = None
        self._sample_rate = 16000

    def _ensure_loaded(self) -> None:
        """Load transformer checkpoint and feature extractor."""
        if self._feature_extractor is not None and self._model is not None:
            return
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        except ImportError as exc:
            raise RuntimeError(
                "Missing optional dependencies for speaker embedding: transformers."
            ) from exc

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self._model_id)
        model = WavLMForXVector.from_pretrained(self._model_id)
        model = model.to(self._device)
        model.eval()
        self._feature_extractor = feature_extractor
        self._model = model
        self._sample_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
        logger.info(
            "speaker_embedder_ready model_id=%s sample_rate=%d device=%s",
            self._model_id,
            self._sample_rate,
            self._device,
        )

    def embed(self, wav_path: Path) -> np.ndarray:
        """Compute one normalized embedding for a WAV file.

        Args:
            wav_path: Source WAV path.

        Returns:
            One-dimensional normalized speaker embedding vector.
        """
        self._ensure_loaded()
        if self._feature_extractor is None or self._model is None:
            raise RuntimeError("Speaker embedder not initialized.")

        import torch
        import torchaudio
        import torch.nn.functional as torch_functional

        waveform, source_sample_rate = torchaudio.load(str(wav_path))
        mono_waveform = waveform.mean(dim=0, keepdim=True)
        if int(source_sample_rate) != self._sample_rate:
            mono_waveform = torchaudio.functional.resample(
                mono_waveform,
                int(source_sample_rate),
                self._sample_rate,
            )
        mono_np = mono_waveform.squeeze(0).cpu().numpy()
        encoded = self._feature_extractor(
            mono_np,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self._model(**encoded)
            normalized = torch_functional.normalize(output.embeddings, dim=-1)
        return normalized[0].detach().cpu().numpy().astype(np.float64, copy=False)
