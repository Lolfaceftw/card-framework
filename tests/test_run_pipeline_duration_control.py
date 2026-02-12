"""Unit tests for run_pipeline duration control helper behavior."""

from __future__ import annotations

import json
from pathlib import Path

from audio2script_and_summarizer import run_pipeline


def test_calculate_corrected_word_budget_scales_to_duration_ratio() -> None:
    """Scale word budget by target/actual duration ratio."""
    corrected = run_pipeline._calculate_corrected_word_budget(  # noqa: SLF001
        current_word_budget=180,
        target_duration_seconds=60.0,
        measured_duration_seconds=90.0,
    )
    assert corrected == 120


def test_update_summary_report_duration_metrics_patches_report(
    tmp_path: Path,
) -> None:
    """Write final duration metrics into the summary report sidecar."""
    summary_output = tmp_path / "audio_summary.json"
    report_path = tmp_path / "audio_summary.json.report.json"
    summary_output.write_text("[]", encoding="utf-8")
    report_path.write_text(
        json.dumps({"summary_outcome": "success", "output_path": str(summary_output)}),
        encoding="utf-8",
    )

    run_pipeline._update_summary_report_duration_metrics(  # noqa: SLF001
        summary_output_path=str(summary_output),
        target_duration_seconds=60.0,
        measured_duration_seconds=63.2,
        duration_tolerance_seconds=3.5,
        duration_correction_passes=1,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["target_duration_seconds"] == 60.0
    assert payload["measured_duration_seconds"] == 63.2
    assert payload["duration_delta_seconds"] == 3.2
    assert payload["duration_within_tolerance"] is True
    assert payload["duration_correction_passes"] == 1
