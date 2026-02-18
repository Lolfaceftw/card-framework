import contextlib
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Iterator, Literal
from audio2script_and_summarizer.pipeline import stage_runner
from audio2script_and_summarizer.pipeline import helpers as pipeline_helpers
from audio2script_and_summarizer.pipeline import flows as pipeline_flows

try:
    from audio2script_and_summarizer.deepseek.stream_events import (
        DEEPSEEK_STREAM_EVENT_PREFIX,  # noqa: F401
        parse_deepseek_stream_event_line as _parse_deepseek_stream_event_line,
        route_deepseek_stream_event as _route_deepseek_stream_event,
    )
    from audio2script_and_summarizer.logging_utils import configure_logging
    from audio2script_and_summarizer.pipeline_config import (
        ConfigValidationError,
        build_pipeline_config,
    )
except ModuleNotFoundError:
    # Allow direct script execution by bootstrapping the repository root.
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from audio2script_and_summarizer.deepseek.stream_events import (
        DEEPSEEK_STREAM_EVENT_PREFIX,  # noqa: F401
        parse_deepseek_stream_event_line as _parse_deepseek_stream_event_line,
        route_deepseek_stream_event as _route_deepseek_stream_event,
    )
    from audio2script_and_summarizer.logging_utils import configure_logging
    from audio2script_and_summarizer.pipeline_config import (
        ConfigValidationError,
        build_pipeline_config,
    )

torch_module: ModuleType | None
try:
    import torch as _torch_module
except ImportError:
    torch_module = None
else:
    torch_module = _torch_module

rich_console: ModuleType | None
rich_layout: ModuleType | None
rich_live: ModuleType | None
rich_panel: ModuleType | None
rich_progress: ModuleType | None
rich_text: ModuleType | None
try:
    import rich.console as _rich_console
    import rich.layout as _rich_layout
    import rich.live as _rich_live
    import rich.panel as _rich_panel
    import rich.progress as _rich_progress
    import rich.text as _rich_text

    RICH_AVAILABLE = True
except ImportError:
    rich_console = None
    rich_layout = None
    rich_live = None
    rich_panel = None
    rich_progress = None
    rich_text = None
    RICH_AVAILABLE = False
else:
    rich_console = _rich_console
    rich_layout = _rich_layout
    rich_live = _rich_live
    rich_panel = _rich_panel
    rich_progress = _rich_progress
    rich_text = _rich_text

UI_CONSOLE: Any = rich_console.Console() if rich_console is not None else None
_ACTIVE_DASHBOARD: "_PipelineDashboard | None" = None

DEFAULT_DEVICE: Final[str] = (
    "cuda" if torch_module is not None and torch_module.cuda.is_available() else "cpu"
)
PIPELINE_TOTAL_STAGES: Final[int] = 5
LOG_HISTORY_MAX_LINES: Final[int] = 5000
PROGRESS_PANEL_HEIGHT: Final[int] = 5
CONTROLS_PANEL_HEIGHT: Final[int] = 9
DEFAULT_HEARTBEAT_SECONDS: Final[float] = 5.0
DEFAULT_DURATION_TOLERANCE_SECONDS: Final[float] = 3.0
DEFAULT_MAX_DURATION_CORRECTION_PASSES: Final[int] = 1
KEYBOARD_POLL_SECONDS: Final[float] = 0.05
STATUS_REFRESH_SECONDS: Final[float] = 1.0
REFRESH_THROTTLE_SECONDS: Final[float] = 0.05
PROMPT_ENTER_GRACE_SECONDS: Final[float] = 0.25
DEFAULT_DEEPSEEK_HARD_CEILING_TOKENS: Final[int] = 64000
DEFAULT_DEEPSEEK_AGENT_MAX_TOOL_ROUNDS: Final[int] = 0
DEFAULT_DEEPSEEK_AGENT_LOOP_EXHAUSTION_POLICY: Final[str] = "auto_salvage"
DEFAULT_DEEPSEEK_BUDGET_FAILURE_POLICY: Final[str] = "degraded_success"
STREAM_PANEL_MIN_WIDTH: Final[int] = 28
SKIP_A2S_EXCLUDED_DIRS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "checkpoints",
    }
)

logger = logging.getLogger(__name__)


from audio2script_and_summarizer.pipeline import dashboard as dashboard_module

_DashboardTheme = dashboard_module._DashboardTheme
DEFAULT_DASHBOARD_THEME: Final[_DashboardTheme] = dashboard_module.DEFAULT_DASHBOARD_THEME
_DashboardOutputCaptureStream = dashboard_module._DashboardOutputCaptureStream
_looks_like_indextts_runtime_line = dashboard_module._looks_like_indextts_runtime_line
_with_indextts_prefix = dashboard_module._with_indextts_prefix
_format_runtime_output_line_for_dashboard = (
    dashboard_module._format_runtime_output_line_for_dashboard
)
_PipelineDashboard = dashboard_module._PipelineDashboard

# Keep dashboard module behavior tied to this module-level probe so tests and
# compatibility monkeypatching on `run_pipeline._supports_unicode_output` remain valid.
_dashboard_supports_unicode_output = dashboard_module._supports_unicode_output

def _supports_unicode_output() -> bool:
    """Return True when the active terminal encoding supports Unicode output."""
    return _dashboard_supports_unicode_output()


dashboard_module._supports_unicode_output = lambda: _supports_unicode_output()

# Re-export dashboard capabilities and rich handles used by the rest of this module.
RICH_AVAILABLE = dashboard_module.RICH_AVAILABLE
UI_CONSOLE = dashboard_module.UI_CONSOLE
rich_panel = dashboard_module.rich_panel
# Expose stdlib `time` for compatibility with tests and monkeypatch-based tooling.
dashboard_module.time = time


@contextlib.contextmanager
def _capture_stage175_output_lines(enabled: bool) -> Iterator[None]:
    """Capture Stage 1.75 runtime stdout/stderr and mirror into dashboard logs."""
    if not enabled:
        yield
        return

    def _on_line(line: str) -> None:
        formatted_line = _format_runtime_output_line_for_dashboard(
            line,
            prefer_indextts_tag=True,
        )
        if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled and formatted_line:
            _ACTIVE_DASHBOARD.log(formatted_line)

    capture_stream = _DashboardOutputCaptureStream(_on_line)
    with (
        contextlib.redirect_stdout(capture_stream),
        contextlib.redirect_stderr(capture_stream),
    ):
        try:
            yield
        finally:
            capture_stream.flush()


@contextlib.contextmanager
def _capture_stage3_output_lines(enabled: bool) -> Iterator[None]:
    """Capture Stage 3 runtime stdout/stderr and mirror into dashboard logs."""
    if not enabled:
        yield
        return

    def _on_line(line: str) -> None:
        formatted_line = _format_runtime_output_line_for_dashboard(
            line,
            prefer_indextts_tag=False,
        )
        if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled and formatted_line:
            _ACTIVE_DASHBOARD.log(formatted_line)

    capture_stream = _DashboardOutputCaptureStream(_on_line)
    with (
        contextlib.redirect_stdout(capture_stream),
        contextlib.redirect_stderr(capture_stream),
    ):
        try:
            yield
        finally:
            capture_stream.flush()
def _print_info(message: str, use_rich: bool) -> None:
    """Print an informational message."""
    logger.info(message)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[cyan]{message}[/cyan]")
        return
    print(message)


def _print_warning(message: str, use_rich: bool) -> None:
    """Print a warning message."""
    logger.warning(message)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[yellow]{message}[/yellow]")
        return
    print(message)


def _print_error(message: str, use_rich: bool) -> None:
    """Print an error message."""
    logger.error(message)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[bold red]{message}[/bold red]")
        return
    print(message)


def _print_success(message: str, use_rich: bool) -> None:
    """Print a success message."""
    logger.info(message)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[bold green]{message}[/bold green]")
        return
    print(message)


def _print_checkpoint(message: str, use_rich: bool) -> None:
    """Print a timestamped status checkpoint."""
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.event(message)
        return
    utc_stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    _print_info(f"[CHECKPOINT {utc_stamp}Z] {message}", use_rich=use_rich)


def _print_stage_banner(title: str, use_rich: bool) -> None:
    """Print a section header for a pipeline stage."""
    logger.info(title)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log("=" * 50)
        _ACTIVE_DASHBOARD.log(title)
        _ACTIVE_DASHBOARD.log("=" * 50)
        return
    if use_rich and UI_CONSOLE is not None and rich_panel is not None:
        panel_module = rich_panel
        UI_CONSOLE.print(
            panel_module.Panel.fit(f"[bold]{title}[/bold]", border_style="cyan")
        )
        return

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def _resolve_device(requested_device: str) -> str:
    """Return a valid runtime device, falling back to CPU when CUDA is unavailable."""
    normalized = requested_device.strip().lower()
    if normalized.startswith("cuda"):
        if torch_module is None or not torch_module.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return requested_device
    return "cpu"


def _find_dotenv_file() -> Path | None:
    """Find a ``.env`` file from CWD upward, then fall back to repo root."""
    return pipeline_helpers.find_dotenv_file(script_file=Path(__file__))


def _load_dotenv_file(dotenv_path: Path) -> int:
    """Load missing environment variables from a dotenv file.

    Args:
        dotenv_path: Path to ``.env`` file.

    Returns:
        Number of environment variables added to ``os.environ``.
    """
    return pipeline_helpers.load_dotenv_file(dotenv_path)


def _auto_load_dotenv() -> None:
    """Automatically load `.env` values into process environment when available."""
    dotenv_path = _find_dotenv_file()
    if dotenv_path is None:
        return

    loaded_count = _load_dotenv_file(dotenv_path)
    _print_info(
        f"[INFO] Loaded .env from: {dotenv_path} ({loaded_count} vars)", use_rich=False
    )


def _extract_module_name(cmd: list[str]) -> str:
    """Extract ``python -m`` module name from a subprocess command."""
    return stage_runner.extract_module_name(cmd)


def _run_stage_command(
    cmd: list[str],
    current_env: dict[str, str],
    use_dashboard: bool,
    stage_name: str,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    model_info: str = "-",
) -> None:
    """Run a child command and optionally stream output into dashboard logs.

    Args:
        cmd: Subprocess command line.
        current_env: Environment variables for the child process.
        use_dashboard: Stream process output into the split UI when True.
        stage_name: Human-readable stage label for status updates.
        heartbeat_seconds: How often to emit activity heartbeat while output is silent.
        model_info: Model/runtime details shown in the controls panel.

    Raises:
        subprocess.CalledProcessError: When child process exits non-zero.
    """
    stage_runner.run_stage_command(
        cmd=cmd,
        current_env=current_env,
        use_dashboard=use_dashboard,
        stage_name=stage_name,
        heartbeat_seconds=heartbeat_seconds,
        model_info=model_info,
        active_dashboard=_ACTIVE_DASHBOARD,
        parse_stream_event_line=_parse_deepseek_stream_event_line,
        route_stream_event=_route_deepseek_stream_event,
        logger=logger,
        subprocess_module=subprocess,
    )


def _count_wav_files(directory: str) -> int:
    """Count ``.wav`` files in a directory.

    Args:
        directory: Directory path.

    Returns:
        Number of ``.wav`` files in ``directory``.
    """
    return pipeline_helpers.count_wav_files(directory)


def _format_speaker_wpm_summary(
    per_speaker_wpm: dict[str, float], max_items: int = 6
) -> str:
    """Build a compact human-readable per-speaker WPM summary.

    Args:
        per_speaker_wpm: Mapping of speaker labels to WPM values.
        max_items: Max speaker entries to include before truncating.

    Returns:
        Formatted summary string.
    """
    return pipeline_helpers.format_speaker_wpm_summary(
        per_speaker_wpm,
        max_items=max_items,
    )


def _prompt_for_provider() -> str:
    """Prompt user to select an LLM provider."""
    while True:
        raw = input("Choose LLM provider [openai/deepseek]: ").strip().lower()
        if raw in {"openai", "deepseek"}:
            return raw
        print("Please type 'openai' or 'deepseek'.")


def _prompt_for_target_minutes() -> float:
    """Prompt user for target summary duration in minutes."""
    while True:
        raw = input("Enter target summary duration in minutes: ").strip()
        try:
            minutes = float(raw)
        except ValueError:
            print("Please enter a numeric value (e.g., 5 or 12.5).")
            continue
        if minutes > 0:
            return minutes
        print("Duration must be greater than 0.")


def _discover_transcript_json_files(search_root: Path) -> list[Path]:
    """Discover diarized transcript JSON files under a search root.

    A file is considered a candidate when it can be parsed by
    ``transcript_wpm.load_transcript_segments`` and yields at least one segment.
    This excludes summary outputs and other JSON files that do not include
    diarized timestamped transcript segments.

    Args:
        search_root: Root directory to scan recursively.

    Returns:
        Candidate transcript JSON paths, newest first.
    """
    return pipeline_helpers.discover_transcript_json_files(
        search_root,
        excluded_dirs=SKIP_A2S_EXCLUDED_DIRS,
    )


def _prompt_for_transcript_json(search_root: Path, use_rich: bool) -> Path:
    """Prompt the operator to choose a transcript JSON file from discovered options.

    Args:
        search_root: Root directory to scan recursively.
        use_rich: Whether rich UI output is active.

    Returns:
        Selected transcript JSON path.

    Raises:
        FileNotFoundError: No transcript-like JSON files were discovered.
    """
    candidates = _discover_transcript_json_files(search_root)
    if not candidates:
        raise FileNotFoundError(f"No transcript JSON files found under {search_root}.")

    _print_info(
        f"[INFO] --skip-a2s discovered {len(candidates)} transcript JSON candidate(s).",
        use_rich=use_rich,
    )
    display_options: list[str] = []
    for candidate in candidates:
        try:
            display_path = candidate.relative_to(search_root)
        except ValueError:
            display_path = candidate
        display_options.append(str(display_path))

    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        selected_idx = _ACTIVE_DASHBOARD.prompt_numeric_choice(
            title="Select transcript JSON file",
            options=display_options,
            default_choice=1,
        )
        if selected_idx is not None:
            return candidates[selected_idx - 1]
        _print_warning(
            "[WARN] Dashboard prompt unavailable; falling back to console input.",
            use_rich=use_rich,
        )

    print("Select transcript JSON file:")
    for idx, display_label in enumerate(display_options, start=1):
        print(f"  {idx}. {display_label}")

    while True:
        raw = input(f"Enter choice [1-{len(candidates)}] (default 1): ").strip()
        if raw == "":
            return candidates[0]
        if raw.isdigit():
            selected_idx = int(raw)
            if 1 <= selected_idx <= len(candidates):
                return candidates[selected_idx - 1]
        print(f"Please enter a number between 1 and {len(candidates)}.")


def _calculate_corrected_word_budget(
    *,
    current_word_budget: int,
    target_duration_seconds: float,
    measured_duration_seconds: float,
) -> int:
    """Scale word budget toward the target/actual duration ratio.

    Args:
        current_word_budget: Current Stage 2 target word budget.
        target_duration_seconds: Desired final duration in seconds.
        measured_duration_seconds: Measured Stage 3 duration in seconds.

    Returns:
        Corrected positive integer word budget.
    """
    return pipeline_helpers.calculate_corrected_word_budget(
        current_word_budget=current_word_budget,
        target_duration_seconds=target_duration_seconds,
        measured_duration_seconds=measured_duration_seconds,
    )


def _calculate_adaptive_tool_rounds(
    *,
    word_budget: int,
    target_minutes: float,
) -> int:
    """Estimate DeepSeek tool-loop rounds from summary size and target duration.

    Args:
        word_budget: Current Stage 2 dialogue word budget target.
        target_minutes: Requested summary duration in minutes.

    Returns:
        Adaptive tool-loop round limit clamped to [10, 30].
    """
    return pipeline_helpers.calculate_adaptive_tool_rounds(
        word_budget=word_budget,
        target_minutes=target_minutes,
    )


def _resolve_deepseek_agent_max_tool_rounds(
    *,
    configured_max_tool_rounds: int,
    current_word_budget: int,
    target_minutes: float,
) -> tuple[int, str]:
    """Resolve explicit or adaptive DeepSeek max tool rounds for this pass.

    Args:
        configured_max_tool_rounds: CLI-provided max rounds (0 means adaptive mode).
        current_word_budget: Current Stage 2 word budget for this pass.
        target_minutes: Requested summary duration in minutes.

    Returns:
        Tuple of `(resolved_max_rounds, source_label)` where source label is
        `"override"` or `"adaptive"`.
    """
    return pipeline_helpers.resolve_deepseek_agent_max_tool_rounds(
        configured_max_tool_rounds=configured_max_tool_rounds,
        current_word_budget=current_word_budget,
        target_minutes=target_minutes,
    )


def _summary_report_path_for_output(summary_output_path: str) -> Path:
    """Resolve default summary report path for a summary JSON output path."""
    return pipeline_helpers.summary_report_path_for_output(summary_output_path)


def _update_summary_report_duration_metrics(
    *,
    summary_output_path: str,
    target_duration_seconds: float | None,
    measured_duration_seconds: float | None,
    duration_tolerance_seconds: float,
    duration_correction_passes: int,
) -> None:
    """Patch summary report JSON with Stage 3 duration metrics when available.

    Args:
        summary_output_path: Stage 2 summary JSON path.
        target_duration_seconds: Requested output duration in seconds.
        measured_duration_seconds: Actual measured Stage 3 duration in seconds.
        duration_tolerance_seconds: Absolute allowed duration delta.
        duration_correction_passes: Number of correction passes executed.
    """
    updated = pipeline_helpers.update_summary_report_duration_metrics(
        summary_output_path=summary_output_path,
        target_duration_seconds=target_duration_seconds,
        measured_duration_seconds=measured_duration_seconds,
        duration_tolerance_seconds=duration_tolerance_seconds,
        duration_correction_passes=duration_correction_passes,
    )
    if not updated:
        logger.info(
            "Summary report duration metrics were not updated for: %s",
            summary_output_path,
        )


def _run_stage3_from_summary(
    *,
    summary_json_path: Path,
    output_wav_path: Path | None,
    runtime_device: str,
    interjection_max_ratio: float,
    mistral_model_id: str,
    mistral_max_new_tokens: int,
) -> tuple[Path, Path, bool, float]:
    """Run Stage 3 voice cloning and interjection pipeline.

    Args:
        summary_json_path: Path to summary JSON produced by Stage 2.
        output_wav_path: Optional output WAV path.
        runtime_device: Runtime device string (cuda/cpu).
        interjection_max_ratio: Maximum interjection ratio.
        mistral_model_id: Hugging Face model id for interjection planning.
        mistral_max_new_tokens: Max generation tokens for planner.

    Returns:
        Tuple of final output WAV path, interjection log path, Mistral
        availability flag, and measured output duration in seconds.
    """
    from audio2script_and_summarizer.stage3_voice import run_stage3_pipeline

    repo_root = Path(__file__).resolve().parents[2]
    checkpoints_root = repo_root / "voice-cloner-and-interjector" / "checkpoints"
    capture_stage3_runtime_output = (
        _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled
    )
    with _capture_stage3_output_lines(enabled=capture_stage3_runtime_output):
        stage3_result = run_stage3_pipeline(
            summary_json_path=summary_json_path,
            output_wav_path=output_wav_path,
            indextts_cfg_path=checkpoints_root / "config.yaml",
            indextts_model_dir=checkpoints_root,
            device=runtime_device,
            interjection_max_ratio=interjection_max_ratio,
            mistral_model_id=mistral_model_id,
            mistral_max_new_tokens=mistral_max_new_tokens,
        )
    return (
        stage3_result.output_wav_path,
        stage3_result.interjection_log_path,
        stage3_result.mistral_enabled,
        stage3_result.output_duration_ms / 1000.0,
    )


def main(
    argv: list[str] | None = None,
    *,
    forced_llm_provider: Literal["deepseek"] | None = None,
) -> int:
    """Run CARD Audio2Script, summarization, and Stage 3 resynthesis pipeline."""
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    requested_experimental_ui = "--experimental-ui" in raw_argv
    requested_plain_ui = "--plain-ui" in raw_argv
    config_error_rich = (
        RICH_AVAILABLE
        and requested_experimental_ui
        and not requested_plain_ui
        and sys.stdout.isatty()
    )
    try:
        args = build_pipeline_config(
            raw_argv,
            forced_llm_provider=forced_llm_provider,
        )
    except ConfigValidationError as exc:
        _print_error(f"[ERROR] {exc}", use_rich=config_error_rich)
        return 1

    log_file = configure_logging(
        level=args.log_level,
        component="run_pipeline",
        enable_console=False,
    )
    os.environ["AUDIO2SCRIPT_LOG_LEVEL"] = args.log_level
    use_rich = RICH_AVAILABLE and not args.plain_ui and sys.stdout.isatty()

    dashboard = _PipelineDashboard(enabled=use_rich)
    global _ACTIVE_DASHBOARD
    _ACTIVE_DASHBOARD = dashboard if dashboard.enabled else None
    dashboard.start()
    _print_info(f"[INFO] File logging enabled: {log_file}", use_rich=use_rich)
    if dashboard.enabled:
        dashboard.set_status(
            stage_name="Pipeline setup",
            substep="Initializing runtime environment",
            module_name="audio2script_and_summarizer.run_pipeline",
            command_display="python -m audio2script_and_summarizer.run_pipeline ...",
            model_info="-",
            pid=None,
            reset_elapsed=True,
        )

    try:
        if not use_rich and not args.plain_ui:
            if not RICH_AVAILABLE:
                _print_info(
                    "[INFO] Rich not installed. Falling back to plain console output.",
                    use_rich=False,
                )
            else:
                _print_info(
                    "[INFO] Interactive terminal not detected. Falling back to plain console output.",
                    use_rich=False,
                )

        _auto_load_dotenv()
        runtime_device = _resolve_device(args.device)
        _print_checkpoint(
            f"Runtime device resolved: {runtime_device}", use_rich=use_rich
        )
        normalized_wpm_source = (
            "tts_preflight" if args.wpm_source == "indextts" else args.wpm_source
        )
        if args.wpm_source == "indextts":
            _print_info(
                "[INFO] --wpm-source indextts is treated as alias of tts_preflight.",
                use_rich=use_rich,
            )

        if args.skip_a2s:
            llm_provider = "deepseek"
            if args.llm_provider and args.llm_provider != "deepseek":
                _print_warning(
                    "[WARN] --skip-a2s requires DeepSeek summarizer; overriding --llm-provider to deepseek.",
                    use_rich=use_rich,
                )
        elif args.skip_a2s_summary:
            llm_provider = None
        else:
            llm_provider = args.llm_provider or _prompt_for_provider()
        if llm_provider is not None:
            _print_checkpoint(
                f"LLM provider selected: {llm_provider}", use_rich=use_rich
            )

        target_minutes: float | None = None
        if args.skip_a2s_summary:
            if args.target_minutes is not None and args.target_minutes <= 0:
                _print_error("[ERROR] Target minutes must be > 0.", use_rich=use_rich)
                return 1
            target_minutes = args.target_minutes
        elif args.target_minutes is None:
            target_minutes = _prompt_for_target_minutes()
        else:
            target_minutes = args.target_minutes
        if target_minutes is not None and target_minutes <= 0:
            _print_error("[ERROR] Target minutes must be > 0.", use_rich=use_rich)
            return 1

        input_path = os.path.abspath(args.input) if args.input else ""
        if not args.skip_a2s and not args.skip_a2s_summary:
            if not input_path:
                _print_error(
                    "[ERROR] --input is required unless --skip-a2s is used.",
                    use_rich=use_rich,
                )
                return 1
            if not os.path.exists(input_path):
                _print_error(
                    f"[ERROR] Input file not found: {input_path}", use_rich=use_rich
                )
                return 1
            _print_checkpoint(f"Input file verified: {input_path}", use_rich=use_rich)

        current_env = os.environ.copy()
        try:
            # Find where python keeps packages
            site_packages = next(p for p in sys.path if "site-packages" in p)
            nvidia_path = os.path.join(site_packages, "nvidia")

            cudnn_lib = os.path.join(nvidia_path, "cudnn", "lib")
            cublas_lib = os.path.join(nvidia_path, "cublas", "lib")

            # If the folders exist, force the system to look there first
            if os.path.exists(cudnn_lib):
                current_ld = current_env.get("LD_LIBRARY_PATH", "")
                libs = [cudnn_lib, cublas_lib]
                current_env["LD_LIBRARY_PATH"] = (
                    os.pathsep.join([*libs, current_ld])
                    if current_ld
                    else os.pathsep.join(libs)
                )
        except Exception as e:
            _print_warning(
                f"[WARN] Could not auto-detect NVIDIA libs (might crash): {e}",
                use_rich=use_rich,
            )

        return pipeline_flows.run_pipeline_modes(
            args=args,
            use_rich=use_rich,
            dashboard=dashboard,
            runtime_device=runtime_device,
            normalized_wpm_source=normalized_wpm_source,
            target_minutes=target_minutes,
            llm_provider=llm_provider,
            input_path=input_path,
            current_env=current_env,
            _ACTIVE_DASHBOARD=_ACTIVE_DASHBOARD,
            _print_stage_banner=_print_stage_banner,
            _print_checkpoint=_print_checkpoint,
            _print_error=_print_error,
            _print_warning=_print_warning,
            _print_success=_print_success,
            _print_info=_print_info,
            _count_wav_files=_count_wav_files,
            _prompt_for_transcript_json=_prompt_for_transcript_json,
            _format_speaker_wpm_summary=_format_speaker_wpm_summary,
            _resolve_deepseek_agent_max_tool_rounds=_resolve_deepseek_agent_max_tool_rounds,
            _run_stage_command=_run_stage_command,
            _run_stage3_from_summary=_run_stage3_from_summary,
            _calculate_corrected_word_budget=_calculate_corrected_word_budget,
            _update_summary_report_duration_metrics=_update_summary_report_duration_metrics,
            _capture_stage175_output_lines=_capture_stage175_output_lines,
            logger=logger,
        )
    finally:
        dashboard.stop()
        _ACTIVE_DASHBOARD = None


if __name__ == "__main__":
    raise SystemExit(main())
