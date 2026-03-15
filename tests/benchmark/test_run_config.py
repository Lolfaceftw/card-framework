"""Coverage for benchmark runtime config compatibility helpers."""

from omegaconf import OmegaConf

from card_framework.benchmark.run import _resolve_orchestrator_word_budget


def test_resolve_orchestrator_word_budget_uses_legacy_defaults_when_missing() -> None:
    cfg = OmegaConf.create({"orchestrator": {"max_iterations": 60}})

    assert _resolve_orchestrator_word_budget(cfg) == (500, 550)


def test_resolve_orchestrator_word_budget_uses_explicit_values() -> None:
    cfg = OmegaConf.create(
        {
            "orchestrator": {
                "min_words": 120,
                "max_words": 180,
                "max_iterations": 60,
            }
        }
    )

    assert _resolve_orchestrator_word_budget(cfg) == (120, 180)
