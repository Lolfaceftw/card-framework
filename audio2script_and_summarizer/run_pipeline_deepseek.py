"""CARD Audio2Script + Summarizer Pipeline (DeepSeek wrapper)."""

from __future__ import annotations

from audio2script_and_summarizer.run_pipeline import main as run_main


def main(argv: list[str] | None = None) -> int:
    """Run pipeline with DeepSeek provider forced by wrapper policy."""
    return run_main(argv=argv, forced_llm_provider="deepseek")


if __name__ == "__main__":
    raise SystemExit(main())
