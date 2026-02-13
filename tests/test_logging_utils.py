"""Unit tests for runtime logging configuration helpers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pytest import MonkeyPatch

from audio2script_and_summarizer.logging_utils import (
    LOG_DIR_ENV_VAR,
    LOG_FILE_ENV_VAR,
    configure_logging,
)


def _flush_root_handlers() -> None:
    """Flush all active root handlers."""
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_configure_logging_writes_json_log_records(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Write structured JSON entries into the configured log file."""
    monkeypatch.setenv(LOG_DIR_ENV_VAR, str(tmp_path))
    monkeypatch.delenv(LOG_FILE_ENV_VAR, raising=False)

    log_path = configure_logging(
        level="INFO",
        component="unit_test",
        enable_console=False,
    )
    logging.getLogger("tests.logging").info("hello from test")
    _flush_root_handlers()

    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert log_lines
    payload = json.loads(log_lines[-1])
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello from test"
    assert payload["component"] == "unit_test"


def test_configure_logging_retains_only_latest_three_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Retain at most three log files sorted by the most recent timestamps."""
    monkeypatch.setenv(LOG_DIR_ENV_VAR, str(tmp_path))
    created_paths: list[Path] = []

    for index in range(5):
        monkeypatch.delenv(LOG_FILE_ENV_VAR, raising=False)
        log_path = configure_logging(
            level="INFO",
            component="unit_test",
            max_log_files=3,
            enable_console=False,
        )
        logging.getLogger("tests.logging").info("entry %d", index)
        _flush_root_handlers()
        os.utime(log_path, (index + 1, index + 1))
        created_paths.append(log_path.resolve())

    existing_paths = {path.resolve() for path in tmp_path.glob("*.log")}
    assert len(existing_paths) == 3
    assert existing_paths == set(created_paths[-3:])
