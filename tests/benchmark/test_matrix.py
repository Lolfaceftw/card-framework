from __future__ import annotations

from card_framework.benchmark.matrix import build_cells
from card_framework.benchmark.types import BenchmarkPreset, EmbeddingProfile, ProviderProfile


def test_build_cells_prunes_with_provider_coverage() -> None:
    profiles = [
        ProviderProfile(
            provider_id="p1",
            description="Provider 1",
            llm_config={"_target_": "card_framework.providers.vllm_provider.VLLMProvider", "base_url": "http://x", "api_key": "EMPTY"},
        ),
        ProviderProfile(
            provider_id="p2",
            description="Provider 2",
            llm_config={"_target_": "card_framework.providers.vllm_provider.VLLMProvider", "base_url": "http://x", "api_key": "EMPTY"},
        ),
        ProviderProfile(
            provider_id="p3",
            description="Provider 3",
            llm_config={"_target_": "card_framework.providers.vllm_provider.VLLMProvider", "base_url": "http://x", "api_key": "EMPTY"},
        ),
    ]

    embedding_profiles = {
        "disabled": EmbeddingProfile(
            embedding_id="disabled",
            description="off",
            embedding_config={
                "_target_": "card_framework.providers.null_provider.NoOpEmbeddingProvider"
            },
        ),
        "enabled": EmbeddingProfile(
            embedding_id="enabled",
            description="on",
            embedding_config={
                "_target_": (
                    "card_framework.providers.sentence_transformer_provider."
                    "SentenceTransformerEmbeddingProvider"
                ),
                "model_name": "x",
            },
        ),
    }

    preset = BenchmarkPreset(
        name="custom",
        target_duration_seconds=300,
        estimated_sample_seconds=120,
        max_samples=20,
        repeats=2,
        include_embedding_profiles=["disabled", "enabled"],
    )

    cells = build_cells(
        profiles=profiles,
        embedding_profiles=embedding_profiles,
        preset=preset,
        sample_count=5,
    )

    covered = {(cell.provider_id, cell.embedding_id, cell.repeat_index) for cell in cells}
    assert ("p1", "disabled", 1) in covered
    assert ("p2", "disabled", 1) in covered
    assert ("p3", "disabled", 1) in covered
    assert len(cells) <= 3

