"""
SentenceTransformer Embedding Provider
=======================================
Wraps a HuggingFace SentenceTransformer model behind the
EmbeddingProvider strategy interface.
"""

import time
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from card_framework.shared.llm_provider import EmbeddingProvider
from card_framework.shared.events import event_bus


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Concrete embedding strategy using ``sentence-transformers``.

    Args:
        model_name: HuggingFace model id (e.g. ``Qwen/Qwen3-Embedding-4B``).
        device:     Torch device string (``cuda``, ``cpu``, etc.).
    """

    # Mapping from config string â†’ torch dtype
    _DTYPE_MAP: dict = {}  # str â†’ torch.dtype, populated lazily

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "auto",
        batch_size: int = 4,
    ) -> None:
        import torch

        self.batch_size = batch_size
        self._device = device
        self._encode_lock = threading.Lock()
        self._closed = False
        # Build the lookup once
        if not self._DTYPE_MAP:
            self._DTYPE_MAP.update(
                {
                    "auto": torch.float32,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                }
            )

        resolved_dtype = self._DTYPE_MAP.get(torch_dtype, torch.float32)
        event_bus.publish("system_message", 
            f"Loading {model_name} on {device} (dtype={resolved_dtype}, batch_size={batch_size})..."
        )
        t0 = time.time()
        self._model = SentenceTransformer(
            model_name,
            model_kwargs={"device_map": device, "torch_dtype": resolved_dtype},
            tokenizer_kwargs={"padding_side": "left"},
        )
        event_bus.publish("system_message", f"Model loaded in {time.time() - t0:.1f}s")

    # â”€â”€ EmbeddingProvider interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
        show_progress: bool = False,
        prompt_name: str | None = None,
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Embedding provider has been closed.")
        kwargs: dict = dict(
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name

        with self._encode_lock:
            if self._closed:
                raise RuntimeError("Embedding provider has been closed.")
            return self._model.encode(texts, **kwargs)

    def close(self) -> None:
        """Release the loaded sentence-transformer model and CUDA cache."""
        if self._closed:
            return
        self._closed = True
        model = getattr(self, "_model", None)
        self._model = None
        del model
        try:
            import gc
            import torch

            gc.collect()
            if str(self._device).lower().startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return None

