"""Build and validate runtime configuration for the pipeline CLI."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Literal

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProvider = Literal["openai", "deepseek"]
DeepseekAgentToolMode = Literal["constraints_only", "full_agentic"]
DeepseekLoopExhaustionPolicy = Literal["auto_salvage", "fail_fast"]
DeepseekBudgetFailurePolicy = Literal["degraded_success", "strict_fail"]
WpmSource = Literal["tts_preflight", "indextts", "transcript"]

VALID_LOG_LEVELS: Final[set[str]] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_LLM_PROVIDERS: Final[set[str]] = {"openai", "deepseek"}
VALID_DEEPSEEK_AGENT_TOOL_MODES: Final[set[str]] = {
    "constraints_only",
    "full_agentic",
}
VALID_DEEPSEEK_LOOP_EXHAUSTION_POLICIES: Final[set[str]] = {
    "auto_salvage",
    "fail_fast",
}
VALID_DEEPSEEK_BUDGET_FAILURE_POLICIES: Final[set[str]] = {
    "degraded_success",
    "strict_fail",
}
VALID_WPM_SOURCES: Final[set[str]] = {"tts_preflight", "indextts", "transcript"}

DEFAULT_DURATION_TOLERANCE_SECONDS: Final[float] = 3.0
DEFAULT_MAX_DURATION_CORRECTION_PASSES: Final[int] = 1
DEFAULT_HEARTBEAT_SECONDS: Final[float] = 5.0
DEFAULT_DEEPSEEK_HARD_CEILING_TOKENS: Final[int] = 64000
DEFAULT_DEEPSEEK_AGENT_MAX_TOOL_ROUNDS: Final[int] = 0
DEFAULT_DEEPSEEK_AGENT_LOOP_EXHAUSTION_POLICY: Final[str] = "auto_salvage"
DEFAULT_DEEPSEEK_BUDGET_FAILURE_POLICY: Final[str] = "degraded_success"

_YAML_PATH_FIELDS: Final[set[str]] = {
    "input",
    "voice_dir",
    "skip_a2s_search_root",
    "summary_json",
    "stage3_output",
    "calibration_presets_path",
}
_YAML_ALLOWED_TOP_LEVEL_FIELDS: Final[set[str]] = {
    "input",
    "device",
    "openai_key",
    "llm_provider",
    "target_minutes",
    "duration_tolerance_seconds",
    "max_duration_correction_passes",
    "word_budget_tolerance",
    "voice_dir",
    "skip_a2s",
    "skip_a2s_summary",
    "skip_a2s_search_root",
    "summary_json",
    "stage3_output",
    "skip_stage3",
    "interjection_max_ratio",
    "mistral_model_id",
    "mistral_max_new_tokens",
    "deepseek_max_completion_tokens",
    "deepseek_agent_tool_mode",
    "deepseek_agent_read_max_lines",
    "deepseek_agent_max_tool_rounds",
    "deepseek_agent_loop_exhaustion_policy",
    "deepseek_budget_failure_policy",
    "wpm_source",
    "calibration_presets_path",
    "no_stem",
    "show_deprecation_warnings",
    "no_progress",
    "plain_ui",
    "heartbeat_seconds",
    "log_level",
}
_YAML_SECTION_FIELDS: Final[dict[str, set[str]]] = {
    "runtime": {
        "input",
        "device",
        "heartbeat_seconds",
        "log_level",
    },
    "llm": {
        "openai_key",
        "llm_provider",
        "target_minutes",
        "word_budget_tolerance",
        "deepseek_max_completion_tokens",
        "deepseek_agent_tool_mode",
        "deepseek_agent_read_max_lines",
        "deepseek_agent_max_tool_rounds",
        "deepseek_agent_loop_exhaustion_policy",
        "deepseek_budget_failure_policy",
        "wpm_source",
    },
    "duration": {
        "duration_tolerance_seconds",
        "max_duration_correction_passes",
    },
    "stage3": {
        "interjection_max_ratio",
        "mistral_model_id",
        "mistral_max_new_tokens",
    },
    "paths": {
        "voice_dir",
        "skip_a2s_search_root",
        "summary_json",
        "stage3_output",
        "calibration_presets_path",
    },
    "skip_modes": {
        "skip_a2s",
        "skip_a2s_summary",
        "skip_stage3",
    },
    "features": {
        "no_stem",
        "show_deprecation_warnings",
        "no_progress",
        "plain_ui",
    },
}


class ConfigValidationError(ValueError):
    """Raise when user-provided pipeline configuration is invalid."""


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    """Represent runtime/system settings."""

    input: str | None
    device: str
    heartbeat_seconds: float
    log_level: LogLevel


@dataclass(slots=True, frozen=True)
class SkipModeConfig:
    """Represent stage skip mode settings."""

    skip_a2s: bool
    skip_a2s_summary: bool
    skip_a2s_search_root: str
    summary_json: str | None
    stage3_output: str | None
    skip_stage3: bool


@dataclass(slots=True, frozen=True)
class LLMConfig:
    """Represent LLM and summarization settings."""

    openai_key: str | None
    llm_provider: LLMProvider | None
    target_minutes: float | None
    word_budget_tolerance: float
    deepseek_max_completion_tokens: int
    deepseek_agent_tool_mode: DeepseekAgentToolMode
    deepseek_agent_read_max_lines: int
    deepseek_agent_max_tool_rounds: int
    deepseek_agent_loop_exhaustion_policy: DeepseekLoopExhaustionPolicy
    deepseek_budget_failure_policy: DeepseekBudgetFailurePolicy
    wpm_source: WpmSource


@dataclass(slots=True, frozen=True)
class DurationConfig:
    """Represent duration-control settings."""

    duration_tolerance_seconds: float
    max_duration_correction_passes: int


@dataclass(slots=True, frozen=True)
class Stage3Config:
    """Represent Stage 3 synthesis settings."""

    interjection_max_ratio: float
    mistral_model_id: str
    mistral_max_new_tokens: int


@dataclass(slots=True, frozen=True)
class UIConfig:
    """Represent UI and child-stage behavior toggles."""

    no_stem: bool
    show_deprecation_warnings: bool
    no_progress: bool
    plain_ui: bool


@dataclass(slots=True, frozen=True)
class PathsConfig:
    """Represent file/path based settings."""

    config: str | None
    voice_dir: str | None
    calibration_presets_path: str


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    """Represent complete pipeline runtime configuration."""

    runtime: RuntimeConfig
    skip_modes: SkipModeConfig
    llm: LLMConfig
    duration: DurationConfig
    stage3: Stage3Config
    ui: UIConfig
    paths: PathsConfig

    @property
    def input(self) -> str | None:
        """Return compatibility accessor for input path."""
        return self.runtime.input

    @property
    def device(self) -> str:
        """Return compatibility accessor for runtime device."""
        return self.runtime.device

    @property
    def heartbeat_seconds(self) -> float:
        """Return compatibility accessor for heartbeat seconds."""
        return self.runtime.heartbeat_seconds

    @property
    def log_level(self) -> LogLevel:
        """Return compatibility accessor for log level."""
        return self.runtime.log_level

    @property
    def skip_a2s(self) -> bool:
        """Return compatibility accessor for skip-a2s mode."""
        return self.skip_modes.skip_a2s

    @property
    def skip_a2s_summary(self) -> bool:
        """Return compatibility accessor for skip-a2s-summary mode."""
        return self.skip_modes.skip_a2s_summary

    @property
    def skip_a2s_search_root(self) -> str:
        """Return compatibility accessor for transcript search root."""
        return self.skip_modes.skip_a2s_search_root

    @property
    def summary_json(self) -> str | None:
        """Return compatibility accessor for summary JSON path."""
        return self.skip_modes.summary_json

    @property
    def stage3_output(self) -> str | None:
        """Return compatibility accessor for Stage 3 output path."""
        return self.skip_modes.stage3_output

    @property
    def skip_stage3(self) -> bool:
        """Return compatibility accessor for skip-stage3 mode."""
        return self.skip_modes.skip_stage3

    @property
    def openai_key(self) -> str | None:
        """Return compatibility accessor for OpenAI key."""
        return self.llm.openai_key

    @property
    def llm_provider(self) -> LLMProvider | None:
        """Return compatibility accessor for LLM provider."""
        return self.llm.llm_provider

    @property
    def target_minutes(self) -> float | None:
        """Return compatibility accessor for target summary duration."""
        return self.llm.target_minutes

    @property
    def word_budget_tolerance(self) -> float:
        """Return compatibility accessor for word-budget tolerance."""
        return self.llm.word_budget_tolerance

    @property
    def deepseek_max_completion_tokens(self) -> int:
        """Return compatibility accessor for DeepSeek max completion tokens."""
        return self.llm.deepseek_max_completion_tokens

    @property
    def deepseek_agent_tool_mode(self) -> DeepseekAgentToolMode:
        """Return compatibility accessor for DeepSeek tool mode."""
        return self.llm.deepseek_agent_tool_mode

    @property
    def deepseek_agent_read_max_lines(self) -> int:
        """Return compatibility accessor for DeepSeek read max lines."""
        return self.llm.deepseek_agent_read_max_lines

    @property
    def deepseek_agent_max_tool_rounds(self) -> int:
        """Return compatibility accessor for DeepSeek max tool rounds."""
        return self.llm.deepseek_agent_max_tool_rounds

    @property
    def deepseek_agent_loop_exhaustion_policy(self) -> DeepseekLoopExhaustionPolicy:
        """Return compatibility accessor for DeepSeek loop exhaustion policy."""
        return self.llm.deepseek_agent_loop_exhaustion_policy

    @property
    def deepseek_budget_failure_policy(self) -> DeepseekBudgetFailurePolicy:
        """Return compatibility accessor for DeepSeek budget failure policy."""
        return self.llm.deepseek_budget_failure_policy

    @property
    def wpm_source(self) -> WpmSource:
        """Return compatibility accessor for WPM source."""
        return self.llm.wpm_source

    @property
    def duration_tolerance_seconds(self) -> float:
        """Return compatibility accessor for duration tolerance."""
        return self.duration.duration_tolerance_seconds

    @property
    def max_duration_correction_passes(self) -> int:
        """Return compatibility accessor for max duration correction passes."""
        return self.duration.max_duration_correction_passes

    @property
    def interjection_max_ratio(self) -> float:
        """Return compatibility accessor for interjection ratio."""
        return self.stage3.interjection_max_ratio

    @property
    def mistral_model_id(self) -> str:
        """Return compatibility accessor for Mistral model id."""
        return self.stage3.mistral_model_id

    @property
    def mistral_max_new_tokens(self) -> int:
        """Return compatibility accessor for Mistral max new tokens."""
        return self.stage3.mistral_max_new_tokens

    @property
    def no_stem(self) -> bool:
        """Return compatibility accessor for no-stem flag."""
        return self.ui.no_stem

    @property
    def show_deprecation_warnings(self) -> bool:
        """Return compatibility accessor for deprecation warning visibility."""
        return self.ui.show_deprecation_warnings

    @property
    def no_progress(self) -> bool:
        """Return compatibility accessor for no-progress flag."""
        return self.ui.no_progress

    @property
    def plain_ui(self) -> bool:
        """Return compatibility accessor for plain-ui flag."""
        return self.ui.plain_ui

    @property
    def voice_dir(self) -> str | None:
        """Return compatibility accessor for voice sample directory."""
        return self.paths.voice_dir

    @property
    def calibration_presets_path(self) -> str:
        """Return compatibility accessor for calibration preset path."""
        return self.paths.calibration_presets_path

    @property
    def config(self) -> str | None:
        """Return resolved runtime config path when provided."""
        return self.paths.config


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for pipeline runtime settings."""
    parser = argparse.ArgumentParser(description="CARD Audio2Script and Summarizer")
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Runtime YAML config path (alias of CARD_CONFIG_PATH).",
    )
    parser.add_argument("--input", default=argparse.SUPPRESS, help="Path to input podcast audio")
    parser.add_argument("--device", default=argparse.SUPPRESS, help="Device to run on (cuda/cpu).")
    parser.add_argument(
        "--openai-key",
        "--api-key",
        dest="openai_key",
        default=argparse.SUPPRESS,
        help="OpenAI API key (alias: --api-key)",
    )
    parser.add_argument(
        "--llm-provider",
        choices=sorted(VALID_LLM_PROVIDERS),
        default=argparse.SUPPRESS,
        help="LLM provider to use (openai or deepseek). If omitted, prompt at runtime.",
    )
    parser.add_argument(
        "--target-minutes",
        type=float,
        default=argparse.SUPPRESS,
        help="Target summary duration in minutes (prompted if omitted).",
    )
    parser.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            "Allowed absolute delta between target and measured Stage 3 duration "
            f"in seconds (default: {DEFAULT_DURATION_TOLERANCE_SECONDS})."
        ),
    )
    parser.add_argument(
        "--max-duration-correction-passes",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Maximum closed-loop duration correction passes that re-run Stage 2/3 "
            f"(default: {DEFAULT_MAX_DURATION_CORRECTION_PASSES})."
        ),
    )
    parser.add_argument(
        "--word-budget-tolerance",
        type=float,
        default=argparse.SUPPRESS,
        help="Tolerance ratio for word budget (default: 0.05 = +/-5%%)",
    )
    parser.add_argument(
        "--voice-dir",
        default=argparse.SUPPRESS,
        help="Directory for speaker samples (default: <input_basename>_voices)",
    )
    parser.add_argument(
        "--skip-a2s",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Skip Stage 1/1.5 Audio2Script processing and jump to DeepSeek "
            "summarization using a selected existing transcript JSON."
        ),
    )
    parser.add_argument(
        "--skip-a2s-summary",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Skip Stage 1/1.5/1.75/2 and run Stage 3 voice cloning + interjections "
            "using an existing summary JSON."
        ),
    )
    parser.add_argument(
        "--skip-a2s-search-root",
        default=argparse.SUPPRESS,
        help=(
            "Root directory used by --skip-a2s when searching for transcript "
            "JSON files (default: current directory)."
        ),
    )
    parser.add_argument(
        "--summary-json",
        default=argparse.SUPPRESS,
        help=(
            "Optional Stage 2 summary JSON path. If omitted with --skip-a2s-summary, "
            "the newest *_summary.json under --skip-a2s-search-root is used."
        ),
    )
    parser.add_argument(
        "--stage3-output",
        default=argparse.SUPPRESS,
        help="Optional final Stage 3 output WAV path.",
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip Stage 3 voice cloning and interjection synthesis.",
    )
    parser.add_argument(
        "--interjection-max-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum ratio of eligible segments that receive interjections.",
    )
    parser.add_argument(
        "--mistral-model-id",
        default=argparse.SUPPRESS,
        help="Hugging Face model id for Stage 3 interjection planning.",
    )
    parser.add_argument(
        "--mistral-max-new-tokens",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum generation tokens per Stage 3 interjection-planner call.",
    )
    parser.add_argument(
        "--deepseek-max-completion-tokens",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Hard output token ceiling forwarded to DeepSeek summarizer "
            f"(default: {DEFAULT_DEEPSEEK_HARD_CEILING_TOKENS})."
        ),
    )
    parser.add_argument(
        "--deepseek-agent-tool-mode",
        choices=sorted(VALID_DEEPSEEK_AGENT_TOOL_MODES),
        default=argparse.SUPPRESS,
        help=(
            "DeepSeek summarizer agentic tool profile: constraints_only "
            "or full_agentic (default: full_agentic)."
        ),
    )
    parser.add_argument(
        "--deepseek-agent-read-max-lines",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Maximum transcript lines returned in each DeepSeek read tool call "
            "(default: 120)."
        ),
    )
    parser.add_argument(
        "--deepseek-agent-max-tool-rounds",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "DeepSeek max agentic tool rounds per attempt. Use 0 for adaptive "
            f"auto-scaling (default: {DEFAULT_DEEPSEEK_AGENT_MAX_TOOL_ROUNDS})."
        ),
    )
    parser.add_argument(
        "--deepseek-agent-loop-exhaustion-policy",
        choices=sorted(VALID_DEEPSEEK_LOOP_EXHAUSTION_POLICIES),
        default=argparse.SUPPRESS,
        help=(
            "DeepSeek tool-loop exhaustion policy forwarded to Stage 2 "
            "(default: auto_salvage)."
        ),
    )
    parser.add_argument(
        "--deepseek-budget-failure-policy",
        choices=sorted(VALID_DEEPSEEK_BUDGET_FAILURE_POLICIES),
        default=argparse.SUPPRESS,
        help=(
            "DeepSeek budget failure policy forwarded to Stage 2 "
            "(default: degraded_success)."
        ),
    )
    parser.add_argument(
        "--wpm-source",
        choices=sorted(VALID_WPM_SOURCES),
        default=argparse.SUPPRESS,
        help=(
            "Source for Stage 1.75 word-rate estimation: "
            "'tts_preflight' runs emotion-aware IndexTTS preflight calibration "
            "(default), 'transcript' computes from diarized transcript timestamps. "
            "'indextts' is kept as a compatibility alias for 'tts_preflight'."
        ),
    )
    parser.add_argument(
        "--calibration-presets-path",
        default=argparse.SUPPRESS,
        help=(
            "Path to emotion pacing presets JSON used by --wpm-source "
            "tts_preflight."
        ),
    )
    parser.add_argument(
        "--no-stem",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip Demucs source separation in diarization stage.",
    )
    parser.add_argument(
        "--show-deprecation-warnings",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show third-party deprecation warnings from diarization dependencies.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable progress bars in child pipeline stages.",
    )
    ui_mode_group = parser.add_mutually_exclusive_group()
    ui_mode_group.add_argument(
        "--experimental-ui",
        dest="plain_ui",
        action="store_false",
        default=argparse.SUPPRESS,
        help=(
            "Enable experimental rich terminal UI when rich is installed and output "
            "is a TTY."
        ),
    )
    ui_mode_group.add_argument(
        "--plain-ui",
        dest="plain_ui",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Force plain console output. This is the default behavior and is kept "
            "for compatibility."
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            "Heartbeat interval for status updates during silent child process periods "
            "(dashboard mode only)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=argparse.SUPPRESS,
        choices=sorted(VALID_LOG_LEVELS),
        type=str.upper,
        help="Console log level propagated to child processes.",
    )
    return parser


def resolve_config_path(cli_config_path: str | None) -> Path | None:
    """Resolve runtime config YAML path from CLI or environment.

    Args:
        cli_config_path: CLI ``--config`` value, if supplied.

    Returns:
        Absolute config path when configured, otherwise ``None``.

    Raises:
        ConfigValidationError: Config path does not exist or is not a file.
    """
    raw_value = cli_config_path or os.getenv("CARD_CONFIG_PATH")
    if raw_value is None or not str(raw_value).strip():
        return None
    candidate = Path(str(raw_value).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise ConfigValidationError(f"Config file not found: {candidate}")
    if not candidate.is_file():
        raise ConfigValidationError(f"Config path is not a file: {candidate}")
    return candidate


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    """Load and flatten runtime YAML config values."""
    if path is None:
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ConfigValidationError(
            "YAML support is unavailable because PyYAML is not installed."
        ) from exc

    raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw_payload is None:
        return {}
    if not isinstance(raw_payload, dict):
        raise ConfigValidationError("Runtime config must be a YAML mapping object.")

    flat_values: dict[str, Any] = {}
    for key, value in raw_payload.items():
        if key in _YAML_ALLOWED_TOP_LEVEL_FIELDS:
            flat_values[key] = value
            continue
        section_fields = _YAML_SECTION_FIELDS.get(str(key))
        if section_fields is None:
            logger.warning("Ignoring unknown runtime config key: %s", key)
            continue
        if not isinstance(value, dict):
            logger.warning(
                "Ignoring runtime config section '%s' because it is not an object.",
                key,
            )
            continue
        for section_key, section_value in value.items():
            if section_key not in section_fields:
                logger.warning(
                    "Ignoring unknown runtime config key: %s.%s",
                    key,
                    section_key,
                )
                continue
            flat_values[section_key] = section_value

    for field_name in _YAML_PATH_FIELDS:
        raw_field_value = flat_values.get(field_name)
        if raw_field_value is None or not isinstance(raw_field_value, str):
            continue
        stripped = raw_field_value.strip()
        if not stripped:
            continue
        candidate = Path(stripped).expanduser()
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        flat_values[field_name] = str(candidate)
    return flat_values


def _parse_env_bool(var_name: str, raw_value: str) -> bool:
    """Parse boolean environment variable values."""
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigValidationError(
        f"Environment variable {var_name} must be a boolean value."
    )


def _parse_env_int(var_name: str, raw_value: str) -> int:
    """Parse integer environment variable values."""
    try:
        return int(raw_value.strip())
    except ValueError as exc:
        raise ConfigValidationError(
            f"Environment variable {var_name} must be an integer."
        ) from exc


def _parse_env_float(var_name: str, raw_value: str) -> float:
    """Parse float environment variable values."""
    try:
        return float(raw_value.strip())
    except ValueError as exc:
        raise ConfigValidationError(
            f"Environment variable {var_name} must be a number."
        ) from exc


def _parse_env_str(_: str, raw_value: str) -> str:
    """Parse string environment variable values."""
    return raw_value.strip()


def load_env_overrides() -> dict[str, Any]:
    """Load supported environment overrides for pipeline settings."""
    env_mapping: dict[str, tuple[str, Callable[[str, str], Any]]] = {
        "OPENAI_API_KEY": ("openai_key", _parse_env_str),
        "AUDIO2SCRIPT_LOG_LEVEL": ("log_level", _parse_env_str),
        "CARD_INPUT": ("input", _parse_env_str),
        "CARD_DEVICE": ("device", _parse_env_str),
        "CARD_LLM_PROVIDER": ("llm_provider", _parse_env_str),
        "CARD_TARGET_MINUTES": ("target_minutes", _parse_env_float),
        "CARD_WORD_BUDGET_TOLERANCE": ("word_budget_tolerance", _parse_env_float),
        "CARD_VOICE_DIR": ("voice_dir", _parse_env_str),
        "CARD_SKIP_A2S": ("skip_a2s", _parse_env_bool),
        "CARD_SKIP_A2S_SUMMARY": ("skip_a2s_summary", _parse_env_bool),
        "CARD_SKIP_A2S_SEARCH_ROOT": ("skip_a2s_search_root", _parse_env_str),
        "CARD_SUMMARY_JSON": ("summary_json", _parse_env_str),
        "CARD_STAGE3_OUTPUT": ("stage3_output", _parse_env_str),
        "CARD_SKIP_STAGE3": ("skip_stage3", _parse_env_bool),
        "CARD_INTERJECTION_MAX_RATIO": ("interjection_max_ratio", _parse_env_float),
        "CARD_MISTRAL_MODEL_ID": ("mistral_model_id", _parse_env_str),
        "CARD_MISTRAL_MAX_NEW_TOKENS": ("mistral_max_new_tokens", _parse_env_int),
        "CARD_DEEPSEEK_MAX_COMPLETION_TOKENS": (
            "deepseek_max_completion_tokens",
            _parse_env_int,
        ),
        "CARD_DEEPSEEK_AGENT_TOOL_MODE": ("deepseek_agent_tool_mode", _parse_env_str),
        "CARD_DEEPSEEK_AGENT_READ_MAX_LINES": (
            "deepseek_agent_read_max_lines",
            _parse_env_int,
        ),
        "CARD_DEEPSEEK_AGENT_MAX_TOOL_ROUNDS": (
            "deepseek_agent_max_tool_rounds",
            _parse_env_int,
        ),
        "CARD_DEEPSEEK_AGENT_LOOP_EXHAUSTION_POLICY": (
            "deepseek_agent_loop_exhaustion_policy",
            _parse_env_str,
        ),
        "CARD_DEEPSEEK_BUDGET_FAILURE_POLICY": (
            "deepseek_budget_failure_policy",
            _parse_env_str,
        ),
        "CARD_WPM_SOURCE": ("wpm_source", _parse_env_str),
        "CARD_CALIBRATION_PRESETS_PATH": ("calibration_presets_path", _parse_env_str),
        "CARD_NO_STEM": ("no_stem", _parse_env_bool),
        "CARD_SHOW_DEPRECATION_WARNINGS": (
            "show_deprecation_warnings",
            _parse_env_bool,
        ),
        "CARD_NO_PROGRESS": ("no_progress", _parse_env_bool),
        "CARD_PLAIN_UI": ("plain_ui", _parse_env_bool),
        "CARD_HEARTBEAT_SECONDS": ("heartbeat_seconds", _parse_env_float),
        "CARD_DURATION_TOLERANCE_SECONDS": (
            "duration_tolerance_seconds",
            _parse_env_float,
        ),
        "CARD_MAX_DURATION_CORRECTION_PASSES": (
            "max_duration_correction_passes",
            _parse_env_int,
        ),
    }
    overrides: dict[str, Any] = {}
    for env_name, (field_name, parser) in env_mapping.items():
        raw_value = os.getenv(env_name)
        if raw_value is None or not raw_value.strip():
            continue
        overrides[field_name] = parser(env_name, raw_value)
    return overrides


def _auto_detect_device() -> str:
    """Resolve default runtime device."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_flat_values() -> dict[str, Any]:
    """Return default flat values matching CLI argument names."""
    return {
        "config": None,
        "input": None,
        "device": _auto_detect_device(),
        "openai_key": None,
        "llm_provider": None,
        "target_minutes": None,
        "duration_tolerance_seconds": DEFAULT_DURATION_TOLERANCE_SECONDS,
        "max_duration_correction_passes": DEFAULT_MAX_DURATION_CORRECTION_PASSES,
        "word_budget_tolerance": 0.05,
        "voice_dir": None,
        "skip_a2s": False,
        "skip_a2s_summary": False,
        "skip_a2s_search_root": ".",
        "summary_json": None,
        "stage3_output": None,
        "skip_stage3": False,
        "interjection_max_ratio": 0.35,
        "mistral_model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral_max_new_tokens": 64,
        "deepseek_max_completion_tokens": DEFAULT_DEEPSEEK_HARD_CEILING_TOKENS,
        "deepseek_agent_tool_mode": "full_agentic",
        "deepseek_agent_read_max_lines": 120,
        "deepseek_agent_max_tool_rounds": DEFAULT_DEEPSEEK_AGENT_MAX_TOOL_ROUNDS,
        "deepseek_agent_loop_exhaustion_policy": DEFAULT_DEEPSEEK_AGENT_LOOP_EXHAUSTION_POLICY,
        "deepseek_budget_failure_policy": DEFAULT_DEEPSEEK_BUDGET_FAILURE_POLICY,
        "wpm_source": "tts_preflight",
        "calibration_presets_path": str(
            Path(__file__).resolve().with_name("emotion_pacing_presets.json")
        ),
        "no_stem": False,
        "show_deprecation_warnings": False,
        "no_progress": False,
        "plain_ui": True,
        "heartbeat_seconds": DEFAULT_HEARTBEAT_SECONDS,
        "log_level": os.getenv("AUDIO2SCRIPT_LOG_LEVEL", "INFO").upper(),
    }


def _to_optional_string(value: Any) -> str | None:
    """Coerce value into optional non-empty string."""
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate or None


def _to_bool(value: Any, field_name: str) -> bool:
    """Coerce value into boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _parse_env_bool(field_name, value)
    raise ConfigValidationError(f"Field '{field_name}' must be a boolean.")


def _to_int(value: Any, field_name: str) -> int:
    """Coerce value into integer."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"Field '{field_name}' must be an integer.") from exc


def _to_float(value: Any, field_name: str) -> float:
    """Coerce value into float."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"Field '{field_name}' must be a number.") from exc


def _normalize_values(flat_values: dict[str, Any]) -> dict[str, Any]:
    """Normalize flat values into final primitive representation."""
    normalized = dict(flat_values)
    normalized["config"] = _to_optional_string(normalized.get("config"))
    normalized["input"] = _to_optional_string(normalized.get("input"))
    normalized["device"] = str(normalized.get("device")).strip()
    normalized["openai_key"] = _to_optional_string(normalized.get("openai_key"))
    normalized["llm_provider"] = _to_optional_string(normalized.get("llm_provider"))
    if normalized["llm_provider"] is not None:
        normalized["llm_provider"] = str(normalized["llm_provider"]).lower()

    target_minutes = normalized.get("target_minutes")
    normalized["target_minutes"] = (
        None if target_minutes is None else _to_float(target_minutes, "target_minutes")
    )
    normalized["duration_tolerance_seconds"] = _to_float(
        normalized.get("duration_tolerance_seconds"),
        "duration_tolerance_seconds",
    )
    normalized["max_duration_correction_passes"] = _to_int(
        normalized.get("max_duration_correction_passes"),
        "max_duration_correction_passes",
    )
    normalized["word_budget_tolerance"] = _to_float(
        normalized.get("word_budget_tolerance"),
        "word_budget_tolerance",
    )
    normalized["voice_dir"] = _to_optional_string(normalized.get("voice_dir"))
    normalized["skip_a2s"] = _to_bool(normalized.get("skip_a2s"), "skip_a2s")
    normalized["skip_a2s_summary"] = _to_bool(
        normalized.get("skip_a2s_summary"),
        "skip_a2s_summary",
    )
    normalized["skip_a2s_search_root"] = str(
        normalized.get("skip_a2s_search_root")
    ).strip()
    normalized["summary_json"] = _to_optional_string(normalized.get("summary_json"))
    normalized["stage3_output"] = _to_optional_string(normalized.get("stage3_output"))
    normalized["skip_stage3"] = _to_bool(normalized.get("skip_stage3"), "skip_stage3")
    normalized["interjection_max_ratio"] = _to_float(
        normalized.get("interjection_max_ratio"),
        "interjection_max_ratio",
    )
    normalized["mistral_model_id"] = str(normalized.get("mistral_model_id")).strip()
    normalized["mistral_max_new_tokens"] = _to_int(
        normalized.get("mistral_max_new_tokens"),
        "mistral_max_new_tokens",
    )
    normalized["deepseek_max_completion_tokens"] = _to_int(
        normalized.get("deepseek_max_completion_tokens"),
        "deepseek_max_completion_tokens",
    )
    normalized["deepseek_agent_tool_mode"] = str(
        normalized.get("deepseek_agent_tool_mode")
    ).strip()
    normalized["deepseek_agent_read_max_lines"] = _to_int(
        normalized.get("deepseek_agent_read_max_lines"),
        "deepseek_agent_read_max_lines",
    )
    normalized["deepseek_agent_max_tool_rounds"] = _to_int(
        normalized.get("deepseek_agent_max_tool_rounds"),
        "deepseek_agent_max_tool_rounds",
    )
    normalized["deepseek_agent_loop_exhaustion_policy"] = str(
        normalized.get("deepseek_agent_loop_exhaustion_policy")
    ).strip()
    normalized["deepseek_budget_failure_policy"] = str(
        normalized.get("deepseek_budget_failure_policy")
    ).strip()
    normalized["wpm_source"] = str(normalized.get("wpm_source")).strip()
    normalized["calibration_presets_path"] = str(
        normalized.get("calibration_presets_path")
    ).strip()
    normalized["no_stem"] = _to_bool(normalized.get("no_stem"), "no_stem")
    normalized["show_deprecation_warnings"] = _to_bool(
        normalized.get("show_deprecation_warnings"),
        "show_deprecation_warnings",
    )
    normalized["no_progress"] = _to_bool(normalized.get("no_progress"), "no_progress")
    normalized["plain_ui"] = _to_bool(normalized.get("plain_ui"), "plain_ui")
    normalized["heartbeat_seconds"] = _to_float(
        normalized.get("heartbeat_seconds"),
        "heartbeat_seconds",
    )
    normalized["log_level"] = str(normalized.get("log_level")).strip().upper()
    return normalized


def _validate_values(flat_values: dict[str, Any]) -> None:
    """Validate normalized flat values."""
    if flat_values["log_level"] not in VALID_LOG_LEVELS:
        raise ConfigValidationError(
            "Field 'log_level' must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    llm_provider = flat_values["llm_provider"]
    if llm_provider is not None and llm_provider not in VALID_LLM_PROVIDERS:
        raise ConfigValidationError("Field 'llm_provider' must be openai or deepseek.")
    if flat_values["deepseek_agent_tool_mode"] not in VALID_DEEPSEEK_AGENT_TOOL_MODES:
        raise ConfigValidationError(
            "Field 'deepseek_agent_tool_mode' must be constraints_only or full_agentic."
        )
    if (
        flat_values["deepseek_agent_loop_exhaustion_policy"]
        not in VALID_DEEPSEEK_LOOP_EXHAUSTION_POLICIES
    ):
        raise ConfigValidationError(
            "Field 'deepseek_agent_loop_exhaustion_policy' must be auto_salvage or fail_fast."
        )
    if (
        flat_values["deepseek_budget_failure_policy"]
        not in VALID_DEEPSEEK_BUDGET_FAILURE_POLICIES
    ):
        raise ConfigValidationError(
            "Field 'deepseek_budget_failure_policy' must be degraded_success or strict_fail."
        )
    if flat_values["wpm_source"] not in VALID_WPM_SOURCES:
        raise ConfigValidationError(
            "Field 'wpm_source' must be tts_preflight, indextts, or transcript."
        )
    if flat_values["heartbeat_seconds"] < 0:
        raise ConfigValidationError("--heartbeat-seconds must be >= 0.")
    if flat_values["deepseek_max_completion_tokens"] <= 0:
        raise ConfigValidationError("--deepseek-max-completion-tokens must be > 0.")
    if flat_values["deepseek_agent_read_max_lines"] <= 0:
        raise ConfigValidationError("--deepseek-agent-read-max-lines must be > 0.")
    if flat_values["deepseek_agent_max_tool_rounds"] < 0:
        raise ConfigValidationError("--deepseek-agent-max-tool-rounds must be >= 0.")
    if flat_values["duration_tolerance_seconds"] < 0:
        raise ConfigValidationError("--duration-tolerance-seconds must be >= 0.")
    if flat_values["max_duration_correction_passes"] < 0:
        raise ConfigValidationError("--max-duration-correction-passes must be >= 0.")
    if not 0 <= flat_values["interjection_max_ratio"] <= 1:
        raise ConfigValidationError("--interjection-max-ratio must be within [0, 1].")
    if flat_values["mistral_max_new_tokens"] <= 0:
        raise ConfigValidationError("--mistral-max-new-tokens must be > 0.")
    if flat_values["skip_a2s"] and flat_values["skip_a2s_summary"]:
        raise ConfigValidationError(
            "--skip-a2s and --skip-a2s-summary are mutually exclusive."
        )
    if flat_values["skip_a2s_summary"] and flat_values["skip_stage3"]:
        raise ConfigValidationError(
            "--skip-a2s-summary requires Stage 3 and cannot be used with --skip-stage3."
        )
    target_minutes = flat_values["target_minutes"]
    if target_minutes is not None and target_minutes <= 0:
        raise ConfigValidationError("Target minutes must be > 0.")
    if not flat_values["device"]:
        raise ConfigValidationError("Field 'device' cannot be empty.")
    if not flat_values["skip_a2s_search_root"]:
        raise ConfigValidationError("Field 'skip_a2s_search_root' cannot be empty.")
    if not flat_values["mistral_model_id"]:
        raise ConfigValidationError("Field 'mistral_model_id' cannot be empty.")
    if not flat_values["calibration_presets_path"]:
        raise ConfigValidationError("Field 'calibration_presets_path' cannot be empty.")


def _build_dataclass_config(flat_values: dict[str, Any]) -> PipelineConfig:
    """Construct typed config dataclasses from normalized values."""
    return PipelineConfig(
        runtime=RuntimeConfig(
            input=flat_values["input"],
            device=flat_values["device"],
            heartbeat_seconds=flat_values["heartbeat_seconds"],
            log_level=flat_values["log_level"],
        ),
        skip_modes=SkipModeConfig(
            skip_a2s=flat_values["skip_a2s"],
            skip_a2s_summary=flat_values["skip_a2s_summary"],
            skip_a2s_search_root=flat_values["skip_a2s_search_root"],
            summary_json=flat_values["summary_json"],
            stage3_output=flat_values["stage3_output"],
            skip_stage3=flat_values["skip_stage3"],
        ),
        llm=LLMConfig(
            openai_key=flat_values["openai_key"],
            llm_provider=flat_values["llm_provider"],
            target_minutes=flat_values["target_minutes"],
            word_budget_tolerance=flat_values["word_budget_tolerance"],
            deepseek_max_completion_tokens=flat_values["deepseek_max_completion_tokens"],
            deepseek_agent_tool_mode=flat_values["deepseek_agent_tool_mode"],
            deepseek_agent_read_max_lines=flat_values["deepseek_agent_read_max_lines"],
            deepseek_agent_max_tool_rounds=flat_values["deepseek_agent_max_tool_rounds"],
            deepseek_agent_loop_exhaustion_policy=flat_values[
                "deepseek_agent_loop_exhaustion_policy"
            ],
            deepseek_budget_failure_policy=flat_values["deepseek_budget_failure_policy"],
            wpm_source=flat_values["wpm_source"],
        ),
        duration=DurationConfig(
            duration_tolerance_seconds=flat_values["duration_tolerance_seconds"],
            max_duration_correction_passes=flat_values["max_duration_correction_passes"],
        ),
        stage3=Stage3Config(
            interjection_max_ratio=flat_values["interjection_max_ratio"],
            mistral_model_id=flat_values["mistral_model_id"],
            mistral_max_new_tokens=flat_values["mistral_max_new_tokens"],
        ),
        ui=UIConfig(
            no_stem=flat_values["no_stem"],
            show_deprecation_warnings=flat_values["show_deprecation_warnings"],
            no_progress=flat_values["no_progress"],
            plain_ui=flat_values["plain_ui"],
        ),
        paths=PathsConfig(
            config=flat_values["config"],
            voice_dir=flat_values["voice_dir"],
            calibration_presets_path=flat_values["calibration_presets_path"],
        ),
    )


def build_pipeline_config(
    argv: list[str] | None = None,
    *,
    forced_llm_provider: Literal["deepseek"] | None = None,
) -> PipelineConfig:
    """Build validated, typed pipeline configuration.

    Args:
        argv: Optional CLI argument list (without executable name).
        forced_llm_provider: Optional provider override used by wrapper entrypoints.

    Returns:
        Validated pipeline configuration object.

    Raises:
        ConfigValidationError: Input config is invalid.
    """
    raw_argv = list(argv) if argv is not None else []
    parser = build_parser()
    cli_values = vars(parser.parse_args(raw_argv))
    resolved_config_path = resolve_config_path(cli_values.get("config"))
    yaml_values = load_yaml_config(resolved_config_path)
    env_values = load_env_overrides()
    merged_values = _default_flat_values()
    merged_values.update(yaml_values)
    merged_values.update(env_values)
    merged_values.update(cli_values)
    if resolved_config_path is not None:
        merged_values["config"] = str(resolved_config_path)

    # UI mode is CLI-controlled by design: plain mode is default, and rich mode
    # requires an explicit --experimental-ui opt-in.
    if "--experimental-ui" in raw_argv:
        merged_values["plain_ui"] = False
    else:
        merged_values["plain_ui"] = True

    if forced_llm_provider is not None:
        configured_provider = merged_values.get("llm_provider")
        if (
            configured_provider is not None
            and str(configured_provider).strip().lower() != forced_llm_provider
        ):
            logger.warning(
                "Overriding llm_provider=%s with forced provider=%s.",
                configured_provider,
                forced_llm_provider,
            )
        merged_values["llm_provider"] = forced_llm_provider

    normalized_values = _normalize_values(merged_values)
    _validate_values(normalized_values)
    return _build_dataclass_config(normalized_values)
