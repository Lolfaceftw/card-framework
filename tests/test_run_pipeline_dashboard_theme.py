"""Unit tests for dashboard theming helpers in run_pipeline."""

from __future__ import annotations

from audio2script_and_summarizer import run_pipeline


def test_output_line_style_maps_semantic_prefixes() -> None:
    """Map common output prefixes to expected Aurora theme styles."""
    dashboard = run_pipeline._PipelineDashboard(enabled=False)

    assert (
        dashboard._output_line_style("[EVENT] stage started")  # noqa: SLF001
        == dashboard._theme.output_event_style  # noqa: SLF001
    )
    assert (
        dashboard._output_line_style("[DEEPSEEK STATUS] round 1/3")  # noqa: SLF001
        == dashboard._theme.output_deepseek_status_style  # noqa: SLF001
    )
    assert (
        dashboard._output_line_style("[DEEPSEEK TOOL_CALL] Invoked tool.")  # noqa: SLF001
        == dashboard._theme.output_deepseek_tool_style  # noqa: SLF001
    )
    assert (
        dashboard._output_line_style("[INDEXTTS2] loading weights")  # noqa: SLF001
        == dashboard._theme.output_indextts_style  # noqa: SLF001
    )
    assert (
        dashboard._output_line_style("[WARNING] degraded mode")  # noqa: SLF001
        == dashboard._theme.output_warning_style  # noqa: SLF001
    )
    assert (
        dashboard._output_line_style("[ERROR] failed operation")  # noqa: SLF001
        == dashboard._theme.output_error_style  # noqa: SLF001
    )
    assert (
        dashboard._output_line_style("normal log line")  # noqa: SLF001
        == dashboard._theme.output_text_style  # noqa: SLF001
    )


def test_runtime_output_formatter_tags_indextts_lines() -> None:
    """Tag known IndexTTS2 runtime lines and classify warning-grade messages."""
    assert (
        run_pipeline._format_runtime_output_line_for_dashboard(  # noqa: SLF001
            ">> starting inference...",
            prefer_indextts_tag=False,
        )
        == "[INDEXTTS2] >> starting inference..."
    )
    assert (
        run_pipeline._format_runtime_output_line_for_dashboard(  # noqa: SLF001
            "RuntimeError('Ninja is required')",
            prefer_indextts_tag=True,
        )
        == "[WARNING] [INDEXTTS2] RuntimeError('Ninja is required')"
    )


def test_stream_line_style_tracks_reasoning_then_answer_phase() -> None:
    """Apply per-phase styles while stream lines transition across headers."""
    dashboard = run_pipeline._PipelineDashboard(enabled=False)

    style, phase = dashboard._stream_line_style(  # noqa: SLF001
        line="[REASONING]",
        active_phase="",
    )
    assert style == dashboard._theme.stream_reasoning_header_style  # noqa: SLF001
    assert phase == "reasoning"

    style, phase = dashboard._stream_line_style(  # noqa: SLF001
        line="thinking token",
        active_phase=phase,
    )
    assert style == dashboard._theme.stream_reasoning_text_style  # noqa: SLF001
    assert phase == "reasoning"

    style, phase = dashboard._stream_line_style(  # noqa: SLF001
        line="[ANSWER]",
        active_phase=phase,
    )
    assert style == dashboard._theme.stream_answer_header_style  # noqa: SLF001
    assert phase == "answer"

    style, phase = dashboard._stream_line_style(  # noqa: SLF001
        line="final answer token",
        active_phase=phase,
    )
    assert style == dashboard._theme.stream_answer_text_style  # noqa: SLF001
    assert phase == "answer"


def test_style_for_context_percent_left_uses_thresholds() -> None:
    """Select subtitle severity style from remaining-context percent values."""
    dashboard = run_pipeline._PipelineDashboard(enabled=False)

    assert (
        dashboard._style_for_context_percent_left(None)  # noqa: SLF001
        == dashboard._theme.stream_meta_style  # noqa: SLF001
    )
    assert (
        dashboard._style_for_context_percent_left(0.80)  # noqa: SLF001
        == dashboard._theme.subtitle_ok_style  # noqa: SLF001
    )
    assert (
        dashboard._style_for_context_percent_left(0.20)  # noqa: SLF001
        == dashboard._theme.subtitle_warn_style  # noqa: SLF001
    )
    assert (
        dashboard._style_for_context_percent_left(0.10)  # noqa: SLF001
        == dashboard._theme.subtitle_critical_style  # noqa: SLF001
    )


def test_heartbeat_symbol_falls_back_for_ascii_terminals(
    monkeypatch,
) -> None:
    """Return deterministic heartbeat frame for Unicode and ASCII terminals."""
    dashboard = run_pipeline._PipelineDashboard(enabled=False)

    monkeypatch.setattr(run_pipeline, "_supports_unicode_output", lambda: True)
    monkeypatch.setattr(run_pipeline.time, "monotonic", lambda: 1.0)
    assert (
        dashboard._heartbeat_symbol_locked()  # noqa: SLF001
        == dashboard._theme.unicode_heartbeat_frames[2]  # noqa: SLF001
    )

    monkeypatch.setattr(run_pipeline, "_supports_unicode_output", lambda: False)
    monkeypatch.setattr(run_pipeline.time, "monotonic", lambda: 0.5)
    assert (
        dashboard._heartbeat_symbol_locked()  # noqa: SLF001
        == dashboard._theme.ascii_heartbeat_frames[1]  # noqa: SLF001
    )
