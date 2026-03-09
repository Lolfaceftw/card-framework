"""Resolve repository, package, and packaged-resource paths."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent

CONFIG_DIR = PACKAGE_ROOT / "config"
PROMPTS_DIR = PACKAGE_ROOT / "prompts" / "templates"
BENCHMARK_DIR = PACKAGE_ROOT / "benchmark"
VENDOR_INDEX_TTS_DIR = PACKAGE_ROOT / "_vendor" / "index_tts"
INDEX_TTS_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints" / "index_tts"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
DEFAULT_BENCHMARK_MANIFEST_PATH = BENCHMARK_DIR / "manifests" / "benchmark_v1.json"
DEFAULT_DIARIZATION_MANIFEST_PATH = (
    BENCHMARK_DIR / "manifests" / "diarization_ami_test.json"
)
DEFAULT_PROVIDER_PROFILES_PATH = BENCHMARK_DIR / "provider_profiles.yaml"
DEFAULT_QA_CONFIG_PATH = BENCHMARK_DIR / "qa_config.yaml"
DEFAULT_JUDGE_RUBRIC_PATH = (
    BENCHMARK_DIR / "rubrics" / "default_summarization_rubric.json"
)


def resolve_repo_relative(path: str | Path) -> Path:
    """Resolve a repository-relative path into an absolute path."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def resolve_benchmark_relative(path: str | Path) -> Path:
    """Resolve a packaged benchmark-relative path into an absolute path."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (BENCHMARK_DIR / candidate).resolve()

