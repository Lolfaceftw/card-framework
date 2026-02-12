"""Logging utilities for Audio2Script applications."""

from __future__ import annotations

import json
import logging
import logging.config
import os
import time
from pathlib import Path
from typing import Final


class UTCFormatter(logging.Formatter):
    """Format timestamps in UTC."""

    def formatTime(  # noqa: N802
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        """Render ``record.created`` using UTC time."""
        created_utc = time.gmtime(record.created)
        timestamp_format = datefmt or DEFAULT_DATE_FORMAT
        return time.strftime(timestamp_format, created_utc)


DEFAULT_LOG_FORMAT: Final[str] = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"
DEFAULT_LOG_DIR_NAME: Final[str] = "logs"
DEFAULT_LOG_FILE_PREFIX: Final[str] = "audio2script"
DEFAULT_MAX_LOG_FILES: Final[int] = 3
LOG_FILE_ENV_VAR: Final[str] = "AUDIO2SCRIPT_LOG_FILE"
LOG_DIR_ENV_VAR: Final[str] = "AUDIO2SCRIPT_LOG_DIR"


class UTCJsonFormatter(UTCFormatter):
    """Render log records as JSON with UTC timestamps."""

    def __init__(self, component: str, environment: str, version: str) -> None:
        """Initialize formatter metadata.

        Args:
            component: Logical component name for the logger context.
            environment: Runtime environment label.
            version: Application version identifier.
        """
        super().__init__()
        self._component = component
        self._environment = environment
        self._version = version

    def format(self, record: logging.LogRecord) -> str:
        """Format a record as a compact JSON line."""
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, DEFAULT_DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "component": self._component,
            "environment": self._environment,
            "version": self._version,
            "trace_id": getattr(record, "trace_id", None),
            "span_id": getattr(record, "span_id", None),
            "request_id": getattr(record, "request_id", None),
            "job_id": getattr(record, "job_id", None),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _normalize_level(level: str) -> str:
    """Normalize a user-provided log level name.

    Args:
        level: Desired level string.

    Returns:
        Uppercased logging level name, defaulting to ``INFO``.
    """
    normalized = level.strip().upper()
    if normalized in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        return normalized
    return "INFO"


def _resolve_log_dir(logs_dir: str | Path | None) -> Path:
    """Resolve the runtime logs directory and create it when missing."""
    if logs_dir is not None:
        resolved = Path(logs_dir)
    else:
        env_dir = os.getenv(LOG_DIR_ENV_VAR)
        resolved = Path(env_dir) if env_dir else Path.cwd() / DEFAULT_LOG_DIR_NAME
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved.resolve()


def _resolve_log_file_path(log_dir: Path) -> Path:
    """Resolve the active log file path for this process."""
    from_env = os.getenv(LOG_FILE_ENV_VAR)
    if from_env:
        inherited = Path(from_env).expanduser()
        if not inherited.is_absolute():
            inherited = log_dir / inherited
        inherited.parent.mkdir(parents=True, exist_ok=True)
        return inherited.resolve()

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    candidate = log_dir / f"{DEFAULT_LOG_FILE_PREFIX}_{timestamp}.log"
    suffix = 1
    while candidate.exists():
        candidate = log_dir / f"{DEFAULT_LOG_FILE_PREFIX}_{timestamp}_{suffix}.log"
        suffix += 1
    return candidate.resolve()


def _prune_old_log_files(log_dir: Path, active_log_path: Path, max_files: int) -> None:
    """Keep only the newest ``max_files`` log files in ``log_dir``.

    Args:
        log_dir: Directory containing log files.
        active_log_path: Log file currently attached to handlers.
        max_files: Maximum number of files to keep.
    """
    if max_files < 1:
        return

    log_files = [path for path in log_dir.glob("*.log") if path.is_file()]
    if len(log_files) <= max_files:
        return

    log_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    retained: list[Path] = [active_log_path]
    for path in log_files:
        if path == active_log_path:
            continue
        retained.append(path)
    for stale_path in retained[max_files:]:
        try:
            stale_path.unlink()
        except OSError:
            logging.getLogger(__name__).warning(
                "Failed to remove stale log file: %s", stale_path
            )


def configure_logging(
    level: str = "INFO",
    *,
    logs_dir: str | Path | None = None,
    component: str = "audio2script",
    environment: str | None = None,
    version: str | None = None,
    max_log_files: int = DEFAULT_MAX_LOG_FILES,
    enable_console: bool = True,
) -> Path:
    """Configure process-wide logging for CLI entry points.

    Args:
        level: Console log level.
        logs_dir: Directory where runtime log files are written.
        component: Component name added to structured log payloads.
        environment: Runtime environment, defaults to ``ENVIRONMENT`` or ``local``.
        version: Application version, defaults to ``APP_VERSION`` or ``unknown``.
        max_log_files: Number of newest log files to retain.
        enable_console: Enable terminal logging when ``True``.

    Returns:
        Active log file path.
    """
    normalized_level = _normalize_level(level)
    resolved_environment = environment or os.getenv("ENVIRONMENT", "local")
    resolved_version = version or os.getenv("APP_VERSION", "unknown")
    log_dir = _resolve_log_dir(logs_dir)
    active_log_path = _resolve_log_file_path(log_dir)
    os.environ[LOG_FILE_ENV_VAR] = str(active_log_path)

    handlers: dict[str, dict[str, str]] = {
        "file": {
            "class": "logging.FileHandler",
            "formatter": "json_utc",
            "level": "DEBUG",
            "filename": str(active_log_path),
            "encoding": "utf-8",
        }
    }
    root_handlers = ["file"]
    if enable_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "utc",
            "level": normalized_level,
        }
        root_handlers.append("console")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "utc": {
                    "()": "audio2script_and_summarizer.logging_utils.UTCFormatter",
                    "format": DEFAULT_LOG_FORMAT,
                    "datefmt": DEFAULT_DATE_FORMAT,
                },
                "json_utc": {
                    "()": "audio2script_and_summarizer.logging_utils.UTCJsonFormatter",
                    "component": component,
                    "environment": resolved_environment,
                    "version": resolved_version,
                },
            },
            "handlers": handlers,
            "root": {
                "handlers": root_handlers,
                "level": "DEBUG",
            },
        }
    )
    _prune_old_log_files(
        log_dir, active_log_path=active_log_path, max_files=max_log_files
    )
    logging.getLogger(__name__).debug(
        "Logging configured",
        extra={
            "component": component,
            "environment": resolved_environment,
            "version": resolved_version,
        },
    )
    return active_log_path
