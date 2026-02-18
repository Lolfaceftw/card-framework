"""Unit tests for typed pipeline config loading and precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio2script_and_summarizer.pipeline_config import (
    ConfigValidationError,
    build_pipeline_config,
)


def _write_yaml(path: Path, content: str) -> None:
    """Write UTF-8 YAML test payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_pipeline_config_parses_cli_values() -> None:
    """Construct typed config from CLI-only values."""
    cfg = build_pipeline_config(
        [
            "--input",
            "podcast.wav",
            "--experimental-ui",
            "--skip-a2s",
            "--deepseek-agent-max-tool-rounds",
            "12",
        ]
    )
    assert cfg.input == "podcast.wav"
    assert cfg.plain_ui is False
    assert cfg.skip_a2s is True
    assert cfg.deepseek_agent_max_tool_rounds == 12


def test_build_pipeline_config_defaults_to_plain_ui() -> None:
    """Run in plain console mode by default unless UI opt-in is provided."""
    cfg = build_pipeline_config([])
    assert cfg.plain_ui is True
    assert cfg.wpm_calibration_cache_mode == "auto"
    assert cfg.wpm_calibration_cache_dir == "artifacts/cache/wpm_calibration"


def test_build_pipeline_config_requires_cli_flag_for_experimental_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore env toggles so only --experimental-ui can enable rich mode."""
    monkeypatch.setenv("CARD_PLAIN_UI", "0")
    cfg = build_pipeline_config([])
    assert cfg.plain_ui is True


def test_build_pipeline_config_precedence_cli_over_env_over_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prioritize CLI values over env and YAML for overlapping fields."""
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "card.yaml"
    _write_yaml(
        cfg_path,
        "\n".join(
            [
                "llm:",
                "  llm_provider: openai",
                "paths:",
                "  voice_dir: voices",
                "",
            ]
        ),
    )
    monkeypatch.setenv("CARD_LLM_PROVIDER", "deepseek")
    cfg = build_pipeline_config(
        [
            "--config",
            str(cfg_path),
            "--llm-provider",
            "openai",
        ]
    )
    assert cfg.llm_provider == "openai"
    assert cfg.voice_dir == str((tmp_path / "voices").resolve())


def test_build_pipeline_config_resolves_card_config_path_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load YAML when CARD_CONFIG_PATH is set and CLI omits --config."""
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "card.yaml"
    _write_yaml(
        cfg_path,
        "\n".join(
            [
                "runtime:",
                "  device: cpu",
                "",
            ]
        ),
    )
    monkeypatch.setenv("CARD_CONFIG_PATH", str(cfg_path))
    cfg = build_pipeline_config([])
    assert cfg.config == str(cfg_path.resolve())
    assert cfg.device == "cpu"


def test_build_pipeline_config_warns_on_unknown_yaml_keys(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ignore unknown YAML keys while emitting warnings."""
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "card.yaml"
    _write_yaml(
        cfg_path,
        "\n".join(
            [
                "unknown_top: 1",
                "llm:",
                "  unknown_option: true",
                "",
            ]
        ),
    )
    caplog.set_level("WARNING")
    build_pipeline_config(["--config", str(cfg_path)])
    assert "Ignoring unknown runtime config key: unknown_top" in caplog.text
    assert "Ignoring unknown runtime config key: llm.unknown_option" in caplog.text


def test_build_pipeline_config_forced_provider_overrides_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Force DeepSeek provider in wrapper mode and warn on conflicts."""
    caplog.set_level("WARNING")
    cfg = build_pipeline_config(
        ["--llm-provider", "openai"],
        forced_llm_provider="deepseek",
    )
    assert cfg.llm_provider == "deepseek"
    assert "Overriding llm_provider=openai with forced provider=deepseek." in caplog.text


def test_build_pipeline_config_rejects_conflicting_skip_modes() -> None:
    """Fail validation when mutually exclusive skip flags are both enabled."""
    with pytest.raises(ConfigValidationError):
        build_pipeline_config(["--skip-a2s", "--skip-a2s-summary"])


def test_build_pipeline_config_parses_stage175_cache_cli_values() -> None:
    """Accept explicit Stage 1.75 cache controls from CLI flags."""
    cfg = build_pipeline_config(
        [
            "--wpm-calibration-cache-mode",
            "refresh",
            "--wpm-calibration-cache-dir",
            "tmp/calibration-cache",
        ]
    )
    assert cfg.wpm_calibration_cache_mode == "refresh"
    assert cfg.wpm_calibration_cache_dir == "tmp/calibration-cache"
