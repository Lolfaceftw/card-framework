import logging
from logging.handlers import RotatingFileHandler

from events import event_bus


def setup_logger(name: str, level=logging.INFO):
    """Set up a base logger without handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


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


def _on_system_message(message: str):
    logger.info(f"[System] {message}")


def _on_status_message(message: str):
    logger.info(f"[Status] {message}")


def _on_error_message(message: str):
    logger.error(f"[Error] {message}")


def _on_agent_message(agent_name: str, message: str, **kwargs):
    logger.info(f"[{agent_name}] {message}")


def _on_agent_thought(agent_name: str, thought: str):
    logger.debug(f"[{agent_name} Thought] {thought}")


def _on_tool_invocation(tool_name: str, arguments: dict):
    logger.info(f"[Tool Invocation] {tool_name} with {arguments}")


def _on_tool_result(tool_name: str, result: str):
    logger.info(f"[Tool Result] {tool_name}: {str(result)[:500]}")


event_bus.subscribe("system_message", _on_system_message)
event_bus.subscribe("status_message", _on_status_message)
event_bus.subscribe("error_message", _on_error_message)
event_bus.subscribe("agent_message", _on_agent_message)
event_bus.subscribe("agent_thought", _on_agent_thought)
event_bus.subscribe("tool_invocation", _on_tool_invocation)
event_bus.subscribe("tool_result", _on_tool_result)
