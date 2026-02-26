import logging
import json
from logging.handlers import RotatingFileHandler

from events import event_bus


def setup_logger(name: str, level=logging.INFO):
    """Set up a base logger without handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def _format_with_metadata(message: str, **metadata) -> str:
    """Attach structured metadata to a log line when present."""
    if not metadata:
        return message
    return f"{message} | meta={json.dumps(metadata, default=str, sort_keys=True)}"


def configure_logger(cfg):
    """
    Configures the logger based on the provided configuration object.
    Expected cfg attributes: enabled, log_file, level, print_to_terminal
    """
    # Disable propagation to prevent root logger (Hydra) from printing to console
    logger.propagate = False

    if not cfg.get("enabled", True):
        logger.handlers = []
        logger.setLevel(logging.CRITICAL + 1)  # Practically disable
        return

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    # Level
    level_name = cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File Handler
    log_file = cfg.get("log_file", "agent_interactions.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler (only if enabled)
    if cfg.get("print_to_terminal", False):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Global instance for easier access
logger = setup_logger("AgentLogger")


def _on_system_message(message: str, **kwargs):
    logger.info(_format_with_metadata(f"[System] {message}", **kwargs))


def _on_status_message(message: str, **kwargs):
    if kwargs.get("inline", False):
        return
    logger.info(_format_with_metadata(f"[Status] {message}", **kwargs))


def _on_error_message(message: str, **kwargs):
    logger.error(_format_with_metadata(f"[Error] {message}", **kwargs))


def _on_agent_message(agent_name: str, message: str, **kwargs):
    logger.info(_format_with_metadata(f"[{agent_name}] {message}", **kwargs))


def _on_agent_thought(agent_name: str, thought: str, **kwargs):
    logger.debug(_format_with_metadata(f"[{agent_name} Thought] {thought}", **kwargs))


def _on_tool_invocation(tool_name: str, arguments: dict, **kwargs):
    logger.info(
        _format_with_metadata(f"[Tool Invocation] {tool_name} with {arguments}", **kwargs)
    )


def _on_tool_result(tool_name: str, result: str, **kwargs):
    logger.info(
        _format_with_metadata(f"[Tool Result] {tool_name}: {str(result)[:500]}", **kwargs)
    )


def _on_a2a_call_started(
    call_id: str, port: int, timeout: float, max_retries: int, **kwargs
):
    logger.debug(
        _format_with_metadata(
            f"[A2A] started call_id={call_id} port={port} timeout={timeout} retries={max_retries}",
            **kwargs,
        )
    )


def _on_a2a_call_retry(
    call_id: str, port: int, attempt: int, delay_seconds: float, error: str, **kwargs
):
    logger.warning(
        _format_with_metadata(
            f"[A2A] retry call_id={call_id} port={port} attempt={attempt} delay={delay_seconds}s error={error}",
            **kwargs,
        )
    )


def _on_a2a_call_succeeded(
    call_id: str, port: int, attempt: int, latency_ms: int, **kwargs
):
    logger.debug(
        _format_with_metadata(
            f"[A2A] succeeded call_id={call_id} port={port} attempt={attempt} latency_ms={latency_ms}",
            **kwargs,
        )
    )


def _on_a2a_call_failed(call_id: str, port: int, error: str, **kwargs):
    logger.error(
        _format_with_metadata(
            f"[A2A] failed call_id={call_id} port={port} error={error}", **kwargs
        )
    )


def _on_llm_call_completed(
    operation: str,
    provider: str,
    latency_ms: int,
    input_messages: int | None = None,
    tool_count: int | None = None,
    **kwargs,
):
    logger.debug(
        _format_with_metadata(
            f"[LLM] completed operation={operation} provider={provider} latency_ms={latency_ms}",
            input_messages=input_messages,
            tool_count=tool_count,
            **kwargs,
        )
    )


event_bus.subscribe("system_message", _on_system_message)
event_bus.subscribe("status_message", _on_status_message)
event_bus.subscribe("error_message", _on_error_message)
event_bus.subscribe("agent_message", _on_agent_message)
event_bus.subscribe("agent_thought", _on_agent_thought)
event_bus.subscribe("tool_invocation", _on_tool_invocation)
event_bus.subscribe("tool_result", _on_tool_result)
event_bus.subscribe("a2a_call_started", _on_a2a_call_started)
event_bus.subscribe("a2a_call_retry", _on_a2a_call_retry)
event_bus.subscribe("a2a_call_succeeded", _on_a2a_call_succeeded)
event_bus.subscribe("a2a_call_failed", _on_a2a_call_failed)
event_bus.subscribe("llm_call_completed", _on_llm_call_completed)
