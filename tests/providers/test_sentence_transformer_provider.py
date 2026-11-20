from __future__ import annotations

import numpy as np

import card_framework.providers.sentence_transformer_provider as provider_module


class _FakeSentenceTransformer:
    def __init__(
        self,
        model_name: str,
        *,
        model_kwargs: dict[str, object] | None = None,
        tokenizer_kwargs: dict[str, object] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.max_seq_length = 40960
        self.encode_calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.encode_calls.append((texts, kwargs))
        return np.zeros((len(texts), 3))


def test_sentence_transformer_provider_applies_tuned_model_kwargs(
    monkeypatch,
) -> None:
    fake_model: _FakeSentenceTransformer | None = None

    def _fake_constructor(
        model_name: str,
        *,
        model_kwargs: dict[str, object] | None = None,
        tokenizer_kwargs: dict[str, object] | None = None,
    ) -> _FakeSentenceTransformer:
        nonlocal fake_model
        fake_model = _FakeSentenceTransformer(
            model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        return fake_model

    monkeypatch.setattr(provider_module, "SentenceTransformer", _fake_constructor)
    monkeypatch.setattr(provider_module.event_bus, "publish", lambda *_args: None)

    provider = provider_module.SentenceTransformerEmbeddingProvider(
        model_name="Qwen/Qwen3-Embedding-4B",
        device="cuda",
        torch_dtype="float16",
        batch_size=1,
        max_seq_length=384,
        attention_implementation="eager",
    )

    assert fake_model is not None
    assert fake_model.model_name == "Qwen/Qwen3-Embedding-4B"
    assert fake_model.model_kwargs["device_map"] == "cuda"
    assert fake_model.model_kwargs["attn_implementation"] == "eager"
    assert fake_model.tokenizer_kwargs["padding_side"] == "left"
    assert fake_model.max_seq_length == 384

    embeddings = provider.encode(["alpha"], prompt_name="query")

    assert embeddings.shape == (1, 3)
    assert fake_model.encode_calls == [
        (
            ["alpha"],
            {
                "batch_size": 1,
                "show_progress_bar": False,
                "normalize_embeddings": True,
                "prompt_name": "query",
            },
        )
    ]
