"""Tests for DeepSeek wrapper entrypoint behavior."""

from __future__ import annotations

from audio2script_and_summarizer import run_pipeline_deepseek


def test_deepseek_wrapper_forces_provider(monkeypatch) -> None:
    """Forward argv while forcing llm provider to deepseek."""
    captured: dict[str, object] = {}

    def _fake_main(
        argv: list[str] | None = None,
        *,
        forced_llm_provider: str | None = None,
    ) -> int:
        captured["argv"] = argv
        captured["forced_llm_provider"] = forced_llm_provider
        return 7

    monkeypatch.setattr(run_pipeline_deepseek, "run_main", _fake_main)
    result = run_pipeline_deepseek.main(["--input", "audio.wav"])
    assert result == 7
    assert captured["argv"] == ["--input", "audio.wav"]
    assert captured["forced_llm_provider"] == "deepseek"

