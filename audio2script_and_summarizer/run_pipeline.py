import argparse
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

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
try:
    import rich.console as _rich_console
    import rich.layout as _rich_layout
    import rich.live as _rich_live
    import rich.panel as _rich_panel
    import rich.progress as _rich_progress

    RICH_AVAILABLE = True
except ImportError:
    rich_console = None
    rich_layout = None
    rich_live = None
    rich_panel = None
    rich_progress = None
    RICH_AVAILABLE = False
else:
    rich_console = _rich_console
    rich_layout = _rich_layout
    rich_live = _rich_live
    rich_panel = _rich_panel
    rich_progress = _rich_progress

UI_CONSOLE: Any = rich_console.Console() if rich_console is not None else None
_ACTIVE_DASHBOARD: "_PipelineDashboard | None" = None

DEFAULT_DEVICE: Final[str] = (
    "cuda"
    if torch_module is not None and torch_module.cuda.is_available()
    else "cpu"
)
PIPELINE_TOTAL_STAGES: Final[int] = 3
LOG_PANEL_MAX_LINES: Final[int] = 18


class _PipelineDashboard:
    """Render split-pane pipeline UI with progress and output logs."""

    def __init__(self, enabled: bool) -> None:
        """Initialize dashboard state.

        Args:
            enabled: Enable rich dashboard rendering when True.
        """
        self.enabled = bool(
            enabled
            and rich_console is not None
            and rich_layout is not None
            and rich_live is not None
            and rich_panel is not None
            and rich_progress is not None
        )
        self._started = False
        self._logs: deque[str] = deque(maxlen=LOG_PANEL_MAX_LINES)
        self._console: Any = None
        self._layout: Any = None
        self._live: Any = None
        self._progress: Any = None
        self._task_id: Any = None

        if not self.enabled:
            return

        console_module = cast(Any, rich_console)
        layout_module = cast(Any, rich_layout)
        live_module = cast(Any, rich_live)
        progress_module = cast(Any, rich_progress)

        self._console = console_module.Console()
        self._layout = layout_module.Layout()
        self._layout.split_column(
            layout_module.Layout(name="progress", size=8),
            layout_module.Layout(name="output"),
        )
        self._progress = progress_module.Progress(
            progress_module.SpinnerColumn(),
            progress_module.TextColumn("[progress.description]{task.description}"),
            progress_module.BarColumn(bar_width=None),
            progress_module.TextColumn("{task.completed}/{task.total} stages"),
            progress_module.TimeElapsedColumn(),
            console=self._console,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            "Pipeline: setup",
            total=PIPELINE_TOTAL_STAGES,
        )
        self._live = live_module.Live(
            self._layout,
            console=self._console,
            refresh_per_second=8,
        )

    def start(self) -> None:
        """Start live rendering."""
        if not self.enabled or self._started:
            return
        self._progress.start()
        self._live.start()
        self._started = True
        self._refresh()

    def stop(self) -> None:
        """Stop live rendering."""
        if not self.enabled or not self._started:
            return
        self._refresh()
        self._live.stop()
        self._progress.stop()
        self._started = False

    def log(self, message: str) -> None:
        """Append a message to the output pane."""
        if not self.enabled:
            return
        clean_line = message.rstrip("\r")
        if clean_line:
            self._logs.append(clean_line)
        self._refresh()

    def complete_stage(self, label: str) -> None:
        """Advance stage progress with a new status label."""
        if not self.enabled:
            return
        self._progress.update(
            self._task_id,
            advance=1,
            description=f"Pipeline: {label}",
        )
        self._refresh()

    def _refresh(self) -> None:
        """Refresh dashboard panels."""
        if not self.enabled:
            return
        panel_module = cast(Any, rich_panel)
        output_text = "\n".join(self._logs) if self._logs else "Waiting for output..."
        self._layout["progress"].update(
            panel_module.Panel(self._progress, title="Progress", border_style="cyan")
        )
        self._layout["output"].update(
            panel_module.Panel(output_text, title="Output", border_style="white")
        )


def _print_info(message: str, use_rich: bool) -> None:
    """Print an informational message."""
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[cyan]{message}[/cyan]")
        return
    print(message)


def _print_warning(message: str, use_rich: bool) -> None:
    """Print a warning message."""
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[yellow]{message}[/yellow]")
        return
    print(message)


def _print_error(message: str, use_rich: bool) -> None:
    """Print an error message."""
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[bold red]{message}[/bold red]")
        return
    print(message)


def _print_success(message: str, use_rich: bool) -> None:
    """Print a success message."""
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[bold green]{message}[/bold green]")
        return
    print(message)


def _print_stage_banner(title: str, use_rich: bool) -> None:
    """Print a section header for a pipeline stage."""
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log("=" * 50)
        _ACTIVE_DASHBOARD.log(title)
        _ACTIVE_DASHBOARD.log("=" * 50)
        return
    if use_rich and UI_CONSOLE is not None and rich_panel is not None:
        panel_module = rich_panel
        UI_CONSOLE.print(panel_module.Panel.fit(f"[bold]{title}[/bold]", border_style="cyan"))
        return

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def _resolve_device(requested_device: str) -> str:
    """Return a valid runtime device, falling back to CPU when CUDA is unavailable."""
    normalized = requested_device.strip().lower()
    if normalized.startswith("cuda"):
        if torch_module is None or not torch_module.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return requested_device
    return "cpu"


def _find_dotenv_file() -> Path | None:
    """Find a ``.env`` file from CWD upward, then fall back to repo root."""
    current_dir = Path.cwd().resolve()
    for directory in [current_dir, *current_dir.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate

    repo_root_candidate = Path(__file__).resolve().parent.parent / ".env"
    if repo_root_candidate.is_file():
        return repo_root_candidate
    return None


def _load_dotenv_file(dotenv_path: Path) -> int:
    """Load missing environment variables from a dotenv file.

    Args:
        dotenv_path: Path to ``.env`` file.

    Returns:
        Number of environment variables added to ``os.environ``.
    """
    loaded_count = 0
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if key not in os.environ:
            os.environ[key] = value
            loaded_count += 1

    return loaded_count


def _auto_load_dotenv() -> None:
    """Automatically load `.env` values into process environment when available."""
    dotenv_path = _find_dotenv_file()
    if dotenv_path is None:
        return

    loaded_count = _load_dotenv_file(dotenv_path)
    _print_info(f"[INFO] Loaded .env from: {dotenv_path} ({loaded_count} vars)", use_rich=False)


def _run_stage_command(
    cmd: list[str],
    current_env: dict[str, str],
    use_dashboard: bool,
) -> None:
    """Run a child command and optionally stream output into dashboard logs.

    Args:
        cmd: Subprocess command line.
        current_env: Environment variables for the child process.
        use_dashboard: Stream process output into the split UI when True.

    Raises:
        subprocess.CalledProcessError: When child process exits non-zero.
    """
    if not use_dashboard:
        subprocess.run(cmd, check=True, env=current_env)
        return

    child_env = current_env.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        if line and _ACTIVE_DASHBOARD is not None:
            _ACTIVE_DASHBOARD.log(line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _count_wav_files(directory: str) -> int:
    """Count ``.wav`` files in a directory.

    Args:
        directory: Directory path.

    Returns:
        Number of ``.wav`` files in ``directory``.
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        return 0
    return sum(1 for _ in directory_path.glob("*.wav"))


def main() -> int:
    """Run the three-stage Audio2Script, splitter, and summarizer pipeline."""
    parser = argparse.ArgumentParser(description="CARD Audio2Script and Summarizer")
    parser.add_argument("--input", required=True, help="Path to input podcast audio")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help=f"Device to run on (cuda/cpu, default: {DEFAULT_DEVICE})")
    parser.add_argument(
        "--openai-key",
        "--api-key",
        dest="openai_key",
        help="OpenAI API key (alias: --api-key)",
    )
    parser.add_argument(
        "--voice-dir",
        default=None,
        help="Directory for speaker samples (default: <input_basename>_voices)",
    )
    parser.add_argument(
        "--no-stem",
        action="store_true",
        default=False,
        help="Skip Demucs source separation in diarization stage.",
    )
    parser.add_argument(
        "--show-deprecation-warnings",
        action="store_true",
        default=False,
        help="Show third-party deprecation warnings from diarization dependencies.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable progress bars in child pipeline stages.",
    )
    parser.add_argument(
        "--plain-ui",
        action="store_true",
        default=False,
        help="Disable rich terminal UI even when rich is installed.",
    )
    args = parser.parse_args()
    use_rich = RICH_AVAILABLE and not args.plain_ui
    dashboard = _PipelineDashboard(enabled=use_rich)
    global _ACTIVE_DASHBOARD
    _ACTIVE_DASHBOARD = dashboard if dashboard.enabled else None
    dashboard.start()

    try:
        if not use_rich and not args.plain_ui:
            _print_info(
                "[INFO] Rich not installed. Falling back to plain console output.",
                use_rich=False,
            )

        _auto_load_dotenv()
        runtime_device = _resolve_device(args.device)

        # 1. API KEY CHECK
        api_key = (
            args.openai_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not api_key:
            _print_error(
                "[ERROR] No OpenAI API key found. Use --openai-key/--api-key or set OPENAI_API_KEY.",
                use_rich=use_rich,
            )
            return 1

        input_path = os.path.abspath(args.input)
        if not os.path.exists(input_path):
            _print_error(f"[ERROR] Input file not found: {input_path}", use_rich=use_rich)
            return 1

        current_env = os.environ.copy()
        try:
            # Find where python keeps packages
            site_packages = next(p for p in sys.path if 'site-packages' in p)
            nvidia_path = os.path.join(site_packages, 'nvidia')

            cudnn_lib = os.path.join(nvidia_path, 'cudnn', 'lib')
            cublas_lib = os.path.join(nvidia_path, 'cublas', 'lib')

            # If the folders exist, force the system to look there first
            if os.path.exists(cudnn_lib):
                current_ld = current_env.get('LD_LIBRARY_PATH', '')
                libs = [cudnn_lib, cublas_lib]
                current_env['LD_LIBRARY_PATH'] = os.pathsep.join([*libs, current_ld]) if current_ld else os.pathsep.join(libs)
        except Exception as e:
            _print_warning(
                f"[WARN] Could not auto-detect NVIDIA libs (might crash): {e}",
                use_rich=use_rich,
            )

        # ==========================================
        # STAGE 1: Diarization
        # ==========================================
        _print_stage_banner("[START] STAGE 1: Audio2Script", use_rich=use_rich)

        try:
            diarize_cmd = [
                sys.executable, "-m", "audio2script_and_summarizer.diarize",
                "-a", input_path,
                "--device", runtime_device,
                "--batch-size", "2" if runtime_device.lower().startswith("cuda") else "1",
            ]
            if args.no_stem:
                diarize_cmd.append("--no-stem")
            if args.show_deprecation_warnings:
                diarize_cmd.append("--show-deprecation-warnings")
            if args.no_progress or dashboard.enabled:
                diarize_cmd.append("--no-progress")
            if dashboard.enabled:
                _print_info(
                    "[INFO] Stage 1 child progress bars disabled; rendering consolidated progress panel.",
                    use_rich=use_rich,
                )

            _run_stage_command(
                cmd=diarize_cmd,
                current_env=current_env,
                use_dashboard=dashboard.enabled,
            )
        except subprocess.CalledProcessError as e:
            _print_error(f"[ERROR] Stage 1 crashed with code {e.returncode}", use_rich=use_rich)
            _print_warning(
                "Tip: If code is -6 or -11, it's a library/driver mismatch.",
                use_rich=use_rich,
            )
            return 1
        dashboard.complete_stage("Stage 1 complete")

        base_name = os.path.splitext(input_path)[0]
        diarization_json = f"{base_name}.json"
        voice_dir = args.voice_dir or f"{base_name}_voices"

        if not os.path.exists(diarization_json):
            _print_error(f"[ERROR] Expected output not found: {diarization_json}", use_rich=use_rich)
            _print_warning("Did you update diarize.py to export JSON?", use_rich=use_rich)
            return 1

        _print_success(
            f"[SUCCESS] Stage 1 Complete. Output: {diarization_json}",
            use_rich=use_rich,
        )

        # ==========================================
        # STAGE 1.5: Audio Splitting
        # ==========================================
        _print_stage_banner("[START] STAGE 1.5: Audio Splitting", use_rich=use_rich)

        try:
            splitter_cmd = [
                sys.executable,
                "-m",
                "audio2script_and_summarizer.audio_splitter",
                "--audio",
                input_path,
                "--json",
                diarization_json,
                "--output-dir",
                voice_dir,
            ]
            _run_stage_command(
                cmd=splitter_cmd,
                current_env=current_env,
                use_dashboard=dashboard.enabled,
            )
        except subprocess.CalledProcessError as e:
            _print_error(
                f"[ERROR] Stage 1.5 crashed with code {e.returncode}",
                use_rich=use_rich,
            )
            return 1
        dashboard.complete_stage("Stage 1.5 complete")

        voice_sample_count = _count_wav_files(voice_dir)
        if voice_sample_count <= 0:
            _print_error(
                f"[ERROR] No speaker samples were generated in: {voice_dir}",
                use_rich=use_rich,
            )
            return 1
        _print_success(
            f"[SUCCESS] Stage 1.5 Complete. Generated {voice_sample_count} speaker sample(s) in {voice_dir}",
            use_rich=use_rich,
        )

        # ==========================================
        # STAGE 2: Summarizer
        # ==========================================
        _print_stage_banner("[START] STAGE 2: Summarizer", use_rich=use_rich)

        summary_output = f"{base_name}_summary.json"

        try:
            summarize_cmd = [
                sys.executable, "-m", "audio2script_and_summarizer.summarizer",
                "--transcript", diarization_json,
                "--voice-dir", voice_dir,
                "--output", summary_output,
                "--api-key", api_key
            ]
            _run_stage_command(
                cmd=summarize_cmd,
                current_env=current_env,
                use_dashboard=dashboard.enabled,
            )
        except subprocess.CalledProcessError as e:
            _print_error(f"[ERROR] Stage 2 crashed with code {e.returncode}", use_rich=use_rich)
            return 1
        dashboard.complete_stage("Stage 2 complete")

        _print_stage_banner("[DONE] PIPELINE COMPLETE", use_rich=use_rich)
        _print_success(f"Summary saved to: {summary_output}", use_rich=use_rich)
        return 0
    finally:
        dashboard.stop()
        _ACTIVE_DASHBOARD = None

if __name__ == "__main__":
    raise SystemExit(main())

