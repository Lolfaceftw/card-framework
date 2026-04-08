"""Unit tests for Textual execution monitor state parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Static

from card_framework.cli.textual_monitor import (
    DetailProgressBar,
    MonitorState,
    PlainAgentStreamParser,
    TextualMonitorApp,
    TextualMonitorRequest,
    _sanitize_agent_panel_text,
    _show_in_timeline,
    fold_monitor_line,
    render_stage_checklist,
)


def test_fold_monitor_line_tracks_wrapper_stage_and_eta() -> None:
    state = MonitorState()

    state, classified = fold_monitor_line(
        state,
        "[5/7] Resolve runtime input\n",
        is_error=False,
    )

    assert state.total_steps == 7
    assert state.current_step_index == 5
    assert state.current_stage == "Resolve runtime input"
    assert state.progress_total == 7.0
    assert state.progress_value == 5.0
    assert state.progress_label == "Setup 5/7"
    assert classified.show_in_diagnostics is False

    state, _classified = fold_monitor_line(
        state,
        "Audio stage: transcription segment 3/12; estimated time left 2m 10s\n",
        is_error=False,
    )

    assert state.current_stage == "Transcription"
    assert state.eta_text == "2m 10s"
    assert state.progress_total == 12.0
    assert state.progress_value == 3.0
    assert state.progress_label == "3/12"
    assert state.progress_detail == "segment 3/12"


def test_fold_monitor_line_uses_specific_audio_substage_labels() -> None:
    state = MonitorState(progress_label="Runtime starting")

    state, _classified = fold_monitor_line(
        state,
        "Audio stage: separation chunk 1/4; estimated time left 30s\n",
        is_error=False,
    )

    assert state.current_stage == "Source separation"
    assert state.progress_label == "1/4"
    assert state.progress_detail == "chunk 1/4"


def test_fold_monitor_line_tracks_eta_overrun_updates() -> None:
    state = MonitorState(current_stage="Transcription", eta_text="2m 10s")

    state, _classified = fold_monitor_line(
        state,
        "Audio stage: transcription running longer than estimate by 45s\n",
        is_error=False,
    )

    assert state.current_stage == "Transcription"
    assert state.eta_text == "Overrun 45s"


def test_fold_monitor_line_maps_loading_status_to_runtime_stage() -> None:
    state = MonitorState(current_stage="Running pipeline")

    state, _classified = fold_monitor_line(
        state,
        "Loading Demucs source separation 'htdemucs'...\n",
        is_error=False,
    )

    assert state.current_stage == "Source separation"
    assert state.progress_label == "Live"


def test_fold_monitor_line_maps_speaker_sample_messages_to_runtime_stage() -> None:
    state = MonitorState(current_stage="Diarization")

    state, _classified = fold_monitor_line(
        state,
        "Reusing speaker sample manifest from transcript metadata: artifacts/speaker_samples/manifest.json\n",
        is_error=False,
    )

    assert state.current_stage == "Speaker samples"


def test_fold_monitor_line_does_not_treat_server_startup_as_summarizer_stage() -> None:
    state = MonitorState(current_stage="Preparing launch")

    state, _classified = fold_monitor_line(
        state,
        "Starting Summarizer A2A server on port 9010...\n",
        is_error=False,
    )

    assert state.current_stage == "Preparing launch"


def test_fold_monitor_line_maps_real_voice_clone_messages_to_runtime_stage() -> None:
    state = MonitorState(current_stage="Summarizer & Critic")

    state, _classified = fold_monitor_line(
        state,
        "Running voice cloning from speaker samples using manifest artifacts/voice/manifest.json\n",
        is_error=False,
    )

    assert state.current_stage == "Voice cloning"


def test_fold_monitor_line_does_not_treat_calibration_as_voice_cloning_stage() -> None:
    state = MonitorState(current_stage="Summarizer & Critic")

    state, _classified = fold_monitor_line(
        state,
        "Ensuring voice-clone calibration artifact is available...\n",
        is_error=False,
    )

    assert state.current_stage == "Summarizer & Critic"


def test_fold_monitor_line_does_not_treat_interjector_setup_as_live_stage() -> None:
    state = MonitorState(current_stage="Voice cloning")

    state, _classified = fold_monitor_line(
        state,
        "Interjector stage enabled with output dir artifacts/interjector\n",
        is_error=False,
    )

    assert state.current_stage == "Voice cloning"


def test_fold_monitor_line_resets_progress_when_runtime_phase_starts() -> None:
    state = MonitorState()

    state, _classified = fold_monitor_line(
        state,
        "[7/7] Running pipeline\n",
        is_error=False,
    )

    assert state.current_stage == "Running pipeline"
    assert state.progress_total == 1.0
    assert state.progress_value == 0.0
    assert state.progress_label == "Runtime starting"
    assert state.progress_detail == "Runtime starting"


def test_fold_monitor_line_resets_from_done_when_follow_on_stage_starts() -> None:
    state = MonitorState(progress_total=1.0, progress_value=1.0, progress_label="Done")

    state, _classified = fold_monitor_line(
        state,
        "Running stage-4 interjection from voice-clone manifest artifacts/voice/manifest.json\n",
        is_error=False,
    )

    assert state.current_stage == "Interjection"
    assert state.progress_total == 1.0
    assert state.progress_value == 0.0
    assert state.progress_label == "Live"
    assert state.progress_detail == "Live"


def test_render_stage_checklist_marks_only_completed_steps_checked() -> None:
    rendered = render_stage_checklist(
        expected_stages=("Source separation", "Speaker samples", "Voice cloning"),
        completed_stages=("Source separation",),
        current_stage="Speaker samples",
        finished=False,
    )

    assert rendered.splitlines() == [
        "[x] Source separation",
        "[ ] Speaker samples <- current",
        "[ ] Voice cloning",
    ]


def test_fold_monitor_line_ignores_generic_complete_messages() -> None:
    state = MonitorState(progress_total=1.0, progress_value=0.0, progress_label="Live")

    state, _classified = fold_monitor_line(
        state,
        "Staged discovery complete. Draft mutations are now allowed.\n",
        is_error=False,
    )

    assert state.progress_label == "Live"


def test_show_in_timeline_hides_prefixed_agent_stream_lines() -> None:
    assert _show_in_timeline("[THINKING] plan the next steps", is_error=False) is False
    assert _show_in_timeline("[CONTENT] summary text", is_error=False) is False


def test_show_in_timeline_keeps_prefixed_transport_log_lines() -> None:
    assert (
        _show_in_timeline(
            '[CONTENT] [2026-04-03 02:25:46,148] - HTTP Request: POST http://127.0.0.1:9010 "HTTP/1.1 200 OK"',
            is_error=False,
        )
        is True
    )


def test_plain_agent_stream_parser_keeps_only_latest_agent_stream() -> None:
    parser = PlainAgentStreamParser()

    parser.consume("\n[Summarizer]\n[THINKING] plan").reasoning
    state = parser.consume(" more\n[CONTENT] summary text")

    assert state.agent_name == "Summarizer"
    assert "plan more" in state.reasoning
    assert "summary text" in state.content


def test_plain_agent_stream_parser_handles_windows_newlines() -> None:
    parser = PlainAgentStreamParser()

    state = parser.consume("\r\n[Critic]\r\n[THINKING] plan\r\n[CONTENT] done")

    assert state.agent_name == "Critic"
    assert "plan" in state.reasoning
    assert "done" in state.content


def test_sanitize_agent_panel_text_drops_http_request_noise() -> None:
    sanitized = _sanitize_agent_panel_text(
        "plan the turn\n"
        '[2026-04-03 02:25:46,148] - HTTP Request: POST http://127.0.0.1:9010 "HTTP/1.1 200 OK"\n'
        "final answer"
    )

    assert "HTTP Request:" not in sanitized
    assert sanitized == "plan the turn\nfinal answer"


def test_sanitize_agent_panel_text_keeps_literal_http_request_content() -> None:
    sanitized = _sanitize_agent_panel_text(
        'log example\nHTTP Request: POST http://127.0.0.1:9010 "HTTP/1.1 200 OK"'
    )

    assert sanitized.endswith(
        'HTTP Request: POST http://127.0.0.1:9010 "HTTP/1.1 200 OK"'
    )


@pytest.mark.anyio
async def test_textual_monitor_hides_http_request_noise_from_agent_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TextualMonitorApp, "_run_subprocess", lambda self: None)
    app = TextualMonitorApp(
        request=TextualMonitorRequest(
            title="Test monitor",
            summary_lines=("Workflow: test",),
            command=("python", "-c", "print('hi')"),
            cwd=Path.cwd(),
        )
    )

    async with app.run_test() as pilot:
        monitor = app
        monitor._latest_agent.reasoning = (
            "plan the turn\n"
            '[2026-04-03 02:25:46,148] - HTTP Request: POST http://127.0.0.1:9010 "HTTP/1.1 200 OK"\n'
            "continue reasoning"
        )
        monitor._latest_agent.content = (
            'draft line\nINFO:httpx:HTTP Request: POST http://127.0.0.1:9010 "HTTP/1.1 200 OK"\n'
            "final content"
        )
        monitor._sync_state_widgets()
        await pilot.pause()

        rendered = str(monitor.query_one("#agent_panel", Static).render())
        assert "HTTP Request:" not in rendered
        assert "plan the turn" in rendered
        assert "final content" in rendered


@pytest.mark.anyio
async def test_textual_monitor_sets_visible_stopped_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TextualMonitorApp, "_run_subprocess", lambda self: None)
    app = TextualMonitorApp(
        request=TextualMonitorRequest(
            title="Test monitor",
            summary_lines=("Workflow: test",),
            command=("python", "-c", "print('hi')"),
            cwd=Path.cwd(),
        )
    )

    async with app.run_test() as pilot:
        monitor = app
        monitor._started_at = 1.0
        monitor._terminate_requested = True
        monitor._state = MonitorState(
            total_steps=7,
            current_step_index=7,
            step_labels=(
                "Preflight checks",
                "IndexTTS repository sync",
                "Dependency sync",
                "Model provisioning",
                "Resolve runtime input",
                "Duration setup",
                "Running pipeline",
            ),
            progress_total=12.0,
            progress_value=3.0,
            progress_label="3/12",
            progress_detail="segment 3/12",
            current_stage="Audio stage",
            eta_text="2m 10s",
        )
        await pilot.pause()

        monitor._set_process_finished(1)
        await pilot.pause()

        assert str(monitor.query_one("#stage_value", Static).render()) == "Stopped"
        assert str(monitor.query_one("#progress_value", Static).render()) == "Stopped"
        assert monitor.query_one("#overall_progress", DetailProgressBar).detail == (
            "Stopped"
        )
        assert str(monitor.query_one("#eta_value", Static).render()) == "ETA Stopped"
        assert monitor.query_one("#stop_run", Button).label == "Stopped"


@pytest.mark.anyio
async def test_textual_monitor_sets_visible_failed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TextualMonitorApp, "_run_subprocess", lambda self: None)
    app = TextualMonitorApp(
        request=TextualMonitorRequest(
            title="Test monitor",
            summary_lines=("Workflow: test",),
            command=("python", "-c", "print('hi')"),
            cwd=Path.cwd(),
        )
    )

    async with app.run_test() as pilot:
        monitor = app
        monitor._started_at = 1.0
        monitor._state = MonitorState(
            progress_total=12.0,
            progress_value=3.0,
            progress_label="3/12",
            progress_detail="segment 3/12",
            current_stage="Transcription",
            eta_text="2m 10s",
        )
        await pilot.pause()

        monitor._set_process_finished(1)
        await pilot.pause()

        assert str(monitor.query_one("#stage_value", Static).render()) == "Failed"
        assert str(monitor.query_one("#progress_value", Static).render()) == "Failed"
        assert monitor.query_one("#overall_progress", DetailProgressBar).detail == (
            "Failed"
        )
        assert str(monitor.query_one("#eta_value", Static).render()) == "ETA Failed"


@pytest.mark.anyio
async def test_textual_monitor_hides_agent_panel_outside_summarization_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TextualMonitorApp, "_run_subprocess", lambda self: None)
    app = TextualMonitorApp(
        request=TextualMonitorRequest(
            title="Test monitor",
            summary_lines=("Workflow: test",),
            command=("python", "-c", "print('hi')"),
            cwd=Path.cwd(),
        )
    )

    async with app.run_test() as pilot:
        monitor = app
        monitor._state = MonitorState(current_stage="Source separation")
        monitor._sync_state_widgets()
        await pilot.pause()

        assert monitor.query_one("#agent_panel", Static).region.height == 0

        monitor._state = MonitorState(current_stage="Summarizer & Critic")
        monitor._sync_state_widgets()
        await pilot.pause()

        assert monitor.query_one("#agent_panel", Static).region.height > 0


@pytest.mark.anyio
async def test_textual_monitor_renders_progress_detail_in_bar_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TextualMonitorApp, "_run_subprocess", lambda self: None)
    app = TextualMonitorApp(
        request=TextualMonitorRequest(
            title="Test monitor",
            summary_lines=("Workflow: test",),
            command=("python", "-c", "print('hi')"),
            cwd=Path.cwd(),
        )
    )

    async with app.run_test() as pilot:
        monitor = app
        monitor._state = MonitorState(
            progress_total=12.0,
            progress_value=3.0,
            progress_label="3/12",
            progress_detail="segment 3/12",
            current_stage="Transcription",
            eta_text="2m 10s",
        )
        monitor._sync_state_widgets()
        await pilot.pause()

        progress_widget = monitor.query_one("#overall_progress", DetailProgressBar)
        assert progress_widget.detail == "segment 3/12"
        assert progress_widget.styles.padding.top == 1
        assert progress_widget.styles.padding.bottom == 1
        progress_region = progress_widget.region
        eta_region = monitor.query_one("#eta_value", Static).region
        assert eta_region.y > progress_region.y


def test_fold_monitor_line_filters_loading_status_from_diagnostics() -> None:
    state = MonitorState()

    _next_state, classified = fold_monitor_line(
        state,
        "Loading runtime dependency sync...\n",
        is_error=True,
    )

    assert classified.show_in_diagnostics is False
    assert classified.style == "cyan"


def test_fold_monitor_line_keeps_real_errors_in_diagnostics() -> None:
    state = MonitorState()

    _next_state, classified = fold_monitor_line(
        state,
        "Error: command failed with exit code 1\n",
        is_error=True,
    )

    assert classified.show_in_diagnostics is True
    assert classified.style == "bold red"


def test_fold_monitor_line_keeps_failed_loading_lines_in_diagnostics() -> None:
    state = MonitorState()

    _next_state, classified = fold_monitor_line(
        state,
        "Failed loading vendored IndexTTS runtime after 12.40 seconds.\n",
        is_error=True,
    )

    assert classified.show_in_diagnostics is True
    assert classified.style == "bold red"
