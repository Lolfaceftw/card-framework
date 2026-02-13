"""DeepSeek stream marker parsing and dashboard routing helpers."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

DEEPSEEK_STREAM_EVENT_PREFIX = "[DEEPSEEK_STREAM] "

logger = logging.getLogger(__name__)


class DeepSeekStreamDashboard(Protocol):
    """Protocol for dashboard interactions used by stream event routing."""

    def open_deepseek_stream_panel(self, model_name: str) -> None:
        """Open and reset the DeepSeek stream panel."""

    def append_deepseek_stream_token(self, *, phase: str, text: str) -> None:
        """Append stream token text into the dashboard panel."""

    def update_deepseek_context_usage(
        self,
        *,
        tokens_used: int,
        tokens_limit: int,
        tokens_left: int,
        percent_left: float,
        rollover_count: int = 0,
    ) -> None:
        """Update context usage subtitle data in the dashboard."""

    def close_deepseek_stream_panel(self) -> None:
        """Hide the DeepSeek stream panel."""

    def log(self, message: str) -> None:
        """Append a line into the output log panel."""


def parse_deepseek_stream_event_line(line: str) -> dict[str, Any] | None:
    """Parse a DeepSeek stream event marker from subprocess output."""
    if not line.startswith(DEEPSEEK_STREAM_EVENT_PREFIX):
        return None

    payload_text = line[len(DEEPSEEK_STREAM_EVENT_PREFIX) :].strip()
    if not payload_text:
        logger.warning("Received empty DeepSeek stream marker payload.")
        return {}

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid DeepSeek stream marker payload: %s", exc)
        return {}

    if not isinstance(payload, dict):
        logger.warning("DeepSeek stream marker payload must be a JSON object.")
        return {}
    return payload


def route_deepseek_stream_event(
    dashboard: DeepSeekStreamDashboard,
    payload: dict[str, Any],
) -> bool:
    """Route one parsed DeepSeek stream event into dashboard updates."""
    event_name = str(payload.get("event", "")).strip().lower()
    if not event_name:
        return False

    if event_name == "start":
        model_name = str(payload.get("model", "deepseek-reasoner")).strip()
        dashboard.open_deepseek_stream_panel(model_name=model_name)
        return True

    if event_name == "token":
        text = payload.get("text")
        if isinstance(text, str) and text:
            phase = str(payload.get("phase", "answer")).strip().lower()
            if phase in {"reasoning", "answer"}:
                dashboard.append_deepseek_stream_token(phase=phase, text=text)
            else:
                phase_tag = phase.upper() if phase else "TOKEN"
                dashboard.log(f"[DEEPSEEK {phase_tag}] {text}")
        return True

    if event_name == "context_usage":
        tokens_used_raw = payload.get("tokens_used")
        tokens_limit_raw = payload.get("tokens_limit")
        tokens_left_raw = payload.get("tokens_left")
        percent_left_raw = payload.get("percent_left")
        rollover_count_raw = payload.get("rollover_count", 0)
        try:
            tokens_used = max(0, int(tokens_used_raw))
            tokens_limit = max(1, int(tokens_limit_raw))
            tokens_left = max(0, int(tokens_left_raw))
            percent_left = max(0.0, min(1.0, float(percent_left_raw)))
            rollover_count = max(0, int(rollover_count_raw))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid DeepSeek context usage payload: %s",
                payload,
            )
            return True
        dashboard.update_deepseek_context_usage(
            tokens_used=tokens_used,
            tokens_limit=tokens_limit,
            tokens_left=tokens_left,
            percent_left=percent_left,
            rollover_count=rollover_count,
        )
        return True

    if event_name == "summary_json_ready":
        output_path = str(payload.get("path", "")).strip()
        if output_path:
            dashboard.log(
                "[DEEPSEEK STATUS] Summary JSON ready; finalizing Stage 2 "
                f"subprocess (output={output_path})."
            )
        else:
            dashboard.log(
                "[DEEPSEEK STATUS] Summary JSON ready; finalizing Stage 2 subprocess."
            )
        return True

    if event_name == "done":
        dashboard.log("[DEEPSEEK STATUS] Stream finished for current DeepSeek call.")
        return True

    return False
