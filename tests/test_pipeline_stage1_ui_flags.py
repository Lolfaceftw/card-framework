"""Tests for Stage 1 UI/progress flag forwarding in pipeline flows."""

from __future__ import annotations

import logging
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from audio2script_and_summarizer.pipeline import flows


@dataclass(slots=True)
class _FakeArgs:
    """Provide the minimal args interface used before Stage 1 exits."""

    skip_a2s_summary: bool = False
    skip_a2s: bool = False
    no_stem: bool = False
    show_deprecation_warnings: bool = False
    plain_ui: bool = True
    no_progress: bool = False
    heartbeat_seconds: float = 1.0


@dataclass(slots=True)
class _FakeDashboard:
    """Represent the minimal dashboard interface needed by flow tests."""

    enabled: bool = False

    def complete_stage(self, _label: str) -> None:
        """No-op completion hook used by run_pipeline_modes."""


def _failing_stage_runner(captured: list[list[str]]) -> Callable[..., None]:
    """Create a stage runner that captures commands then fails Stage 1."""

    def _run_stage_command(*, cmd: list[str], **_: Any) -> None:
        captured.append(cmd)
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    return _run_stage_command


def _run_until_stage1_failure(
    *,
    plain_ui: bool,
    no_progress: bool,
) -> list[str]:
    """Run pipeline flow until Stage 1 and return captured diarize command."""

    captured_cmds: list[list[str]] = []
    args = _FakeArgs(plain_ui=plain_ui, no_progress=no_progress)
    exit_code = flows.run_pipeline_modes(
        args=args,
        use_rich=False,
        dashboard=_FakeDashboard(enabled=False),
        runtime_device="cpu",
        normalized_wpm_source="tts_preflight",
        target_minutes=1.0,
        llm_provider="deepseek",
        input_path="audio.wav",
        current_env={},
        _ACTIVE_DASHBOARD=None,
        _print_stage_banner=lambda _message, use_rich: None,
        _print_checkpoint=lambda _message, use_rich: None,
        _print_error=lambda _message, use_rich: None,
        _print_warning=lambda _message, use_rich: None,
        _print_success=lambda _message, use_rich: None,
        _print_info=lambda _message, use_rich: None,
        _count_wav_files=lambda _path: 0,
        _prompt_for_transcript_json=lambda search_root, use_rich: Path("unused.json"),
        _format_speaker_wpm_summary=lambda per_speaker_wpm, max_items=6: "unused",
        _resolve_deepseek_agent_max_tool_rounds=lambda **_: (1, "unused"),
        _run_stage_command=_failing_stage_runner(captured_cmds),
        _run_stage3_from_summary=lambda **_: (
            Path("out.wav"),
            Path("out.json"),
            True,
            1.0,
        ),
        _calculate_corrected_word_budget=lambda **_: 1,
        _update_summary_report_duration_metrics=lambda **_: None,
        _capture_stage175_output_lines=lambda _enabled: nullcontext(),
        logger=logging.getLogger(__name__),
    )
    assert exit_code == 1
    assert captured_cmds, "Expected Stage 1 command to be executed."
    return captured_cmds[0]


def test_stage1_plain_ui_forwards_plain_flag_without_no_progress() -> None:
    """Forward --plain-ui alone when plain mode is active and progress is enabled."""

    cmd = _run_until_stage1_failure(plain_ui=True, no_progress=False)
    assert "--plain-ui" in cmd
    assert "--no-progress" not in cmd


def test_stage1_plain_ui_with_no_progress_forwards_both_flags() -> None:
    """Forward both --plain-ui and --no-progress when both modes are requested."""

    cmd = _run_until_stage1_failure(plain_ui=True, no_progress=True)
    assert "--plain-ui" in cmd
    assert "--no-progress" in cmd
