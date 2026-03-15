from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from card_framework.shared.logger_utils import (
    ResilientRotatingFileHandler,
    _summarize_tool_result,
    configure_logger,
    logger,
)


def test_summarize_tool_result_reports_dict_keys_for_json_payload() -> None:
    summary = _summarize_tool_result(
        '{"excerpt":"[SPEAKER_00]: hello\\n[SPEAKER_01]: world","match_count":2}'
    )

    assert summary == "dict keys=['excerpt', 'match_count']"


def test_summarize_tool_result_truncates_plain_text_payloads() -> None:
    summary = _summarize_tool_result("word " * 80)

    assert len(summary) <= 160
    assert summary.endswith("...")


def test_configure_logger_uses_rich_console_handler_when_enabled(tmp_path) -> None:
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    try:
        configured = configure_logger(
            {
                "enabled": True,
                "log_file": str(tmp_path / "agent.log"),
                "level": "INFO",
                "print_to_terminal": True,
            }
        )

        assert configured is logger
        assert any(
            isinstance(handler, RotatingFileHandler) for handler in logger.handlers
        )
        rich_handlers = [
            handler for handler in logger.handlers if isinstance(handler, RichHandler)
        ]
        assert len(rich_handlers) == 1
        assert logger.level == logging.INFO
    finally:
        created_handlers = [
            handler for handler in logger.handlers if handler not in original_handlers
        ]
        for handler in created_handlers:
            handler.close()
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_configure_logger_skips_console_handler_when_terminal_disabled(
    tmp_path,
) -> None:
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    try:
        configure_logger(
            {
                "enabled": True,
                "log_file": str(tmp_path / "agent.log"),
                "level": "INFO",
                "print_to_terminal": False,
            }
        )

        assert any(
            isinstance(handler, RotatingFileHandler) for handler in logger.handlers
        )
        assert not any(isinstance(handler, RichHandler) for handler in logger.handlers)
    finally:
        created_handlers = [
            handler for handler in logger.handlers if handler not in original_handlers
        ]
        for handler in created_handlers:
            handler.close()
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_resilient_rotating_file_handler_falls_back_when_rollover_is_locked(
    tmp_path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "agent.log"
    handler = ResilientRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
        rollover_retry_seconds=30.0,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        handler.emit(
            logging.LogRecord(
                name="AgentLogger",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="first",
                args=(),
                exc_info=None,
            )
        )

        def _locked_rollover() -> None:
            if handler.stream is not None:
                handler.stream.close()
                handler.stream = None
            raise PermissionError("locked")

        monkeypatch.setattr(handler, "doRollover", _locked_rollover)
        locked_record = logging.LogRecord(
            name="AgentLogger",
            level=logging.INFO,
            pathname=__file__,
            lineno=2,
            msg="second",
            args=(),
            exc_info=None,
        )

        handler.emit(locked_record)
        assert handler.shouldRollover(locked_record) is False
    finally:
        handler.close()

    assert log_path.read_text(encoding="utf-8").splitlines() == ["first", "second"]
