"""Logging utilities for Audio2Script applications."""

from __future__ import annotations

import logging
import logging.config
import time
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


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide logging for CLI entry points.

    Args:
        level: Root logger level.
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "utc": {
                    "()": "audio2script_and_summarizer.logging_utils.UTCFormatter",
                    "format": DEFAULT_LOG_FORMAT,
                    "datefmt": DEFAULT_DATE_FORMAT,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "utc",
                    "level": level,
                }
            },
            "root": {
                "handlers": ["console"],
                "level": level,
            },
        }
    )
