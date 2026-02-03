"""
Centralized logging utility for podcast modules.

This module provides industry-standard logging with:
- ISO 8601 timestamps
- Configurable log levels via LOG_LEVEL environment variable
- Clean console output with proper formatting
- Debug logs hidden by default to avoid flooding

Usage:
    from tools.podcast.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing segment 1/5")
    logger.debug("LLM response: ...")  # Hidden by default
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class TimestampFormatter(logging.Formatter):
    """Custom formatter with ISO 8601 timestamps in local timezone."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Format timestamp as ISO 8601 with timezone.

        Args:
            record: The log record.
            datefmt: Date format string (ignored, using ISO 8601).

        Returns:
            Formatted timestamp string.
        """
        ct = datetime.fromtimestamp(record.created)
        return ct.strftime("%Y-%m-%d %H:%M:%S")


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Optional log level override. Defaults to LOG_LEVEL env var or INFO.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Determine log level from env or parameter
    level_str = level or os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level_str, logging.INFO)

    logger.setLevel(log_level)

    # Console handler with timestamp formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] message
    formatter = TimestampFormatter(
        fmt="[%(asctime)s] [%(levelname)-7s] %(message)s"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def configure_root_logger(level: Optional[str] = None) -> None:
    """Configure the root logger for the application.

    Suppresses noisy loggers from dependencies while keeping
    application logs clean.

    Args:
        level: Optional log level override.
    """
    level_str = level or os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level_str, logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout
    )

    # Suppress noisy third-party loggers
    noisy_loggers = [
        "urllib3",
        "requests",
        "httpx",
        "httpcore",
        "torch",
        "transformers",
        "accelerate",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
