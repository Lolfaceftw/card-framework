"""Regression coverage for the cleaned repo navigation layout."""

from __future__ import annotations

from card_framework.shared.paths import (
    DATASETS_DIR,
    DOCS_DIR,
    REPO_ROOT,
    RESEARCH_DIR,
    SPORC_HUMAN_EVAL_ASSETS_DIR,
)


def test_repo_navigation_docs_live_under_docs_directory() -> None:
    """Root-level operator docs should live under docs/."""
    assert (DOCS_DIR / "config.md").is_file()
    assert (DOCS_DIR / "logs.md").is_file()
    assert (DOCS_DIR / "navigation.md").is_file()
    assert not (REPO_ROOT / "CONFIG.MD").exists()


def test_research_assets_live_under_research_directory() -> None:
    """The paper and LaTeX manuscript should be grouped under research/."""
    assert (RESEARCH_DIR / "EEE_196_CARD_UCL.md").is_file()
    assert (RESEARCH_DIR / "paper" / "main.tex").is_file()
    assert not (REPO_ROOT / "research_doc").exists()


def test_sporc_human_eval_assets_live_under_datasets_directory() -> None:
    """Tracked benchmark handoff packages should stay out of the repo root."""
    assert DATASETS_DIR.is_dir()
    assert SPORC_HUMAN_EVAL_ASSETS_DIR.is_dir()
    assert (SPORC_HUMAN_EVAL_ASSETS_DIR / "package_summary.json").is_file()
    assert (DATASETS_DIR / "sporc" / "review_manifests" / "tmp_review_sporc").is_dir()
    assert not (REPO_ROOT / "sporc_human_eval_assets_10m").exists()


def test_manual_probe_is_not_shipped_inside_benchmark_package() -> None:
    """Manual tooling should live in scripts/manual instead of src/card_framework/benchmark."""
    assert (REPO_ROOT / "scripts" / "manual" / "vllm_model_browser.py").is_file()
    assert not (REPO_ROOT / "src" / "card_framework" / "benchmark" / "test_ui.py").exists()
    assert not (REPO_ROOT / "src" / "card_framework" / "benchmark" / "somebody.py").exists()
