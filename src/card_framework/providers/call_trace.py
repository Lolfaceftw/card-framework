"""Write complete, navigable agent exchanges without token-stream fragments.

These records contain prompts and model-returned reasoning, never authentication
headers or client configuration. Keep run directories private until reviewed.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any
from uuid import uuid4

_INDEX_LOCK = Lock()


def fenced(value: str, language: str = "") -> str:
    """Render text in a Markdown fence without letting embedded fences close it.

    Args:
        value: Complete text to preserve in the rendered record.
        language: Optional Markdown syntax-highlighting language.

    Returns:
        A fenced block containing the entire input without truncation.
    """
    runs = re.findall(r"`+", value)
    width = max([3, *(len(run) + 1 for run in runs)])
    fence = "`" * width
    return f"{fence}{language}\n{value}\n{fence}\n"


def write_call_trace(root: Path, record: dict[str, Any]) -> Path:
    """Persist one finished or failed call and append it to the run index.

    Args:
        root: Run-local directory for exchange JSON, Markdown, and README index.
        record: Complete serializable call data, including agent, status,
            elapsed_seconds, request, response, and provider_response. The caller
            must exclude credentials and HTTP client configuration.

    Returns:
        Path to the readable Markdown exchange. JSON is saved beside it.

    Raises:
        OSError: A record cannot be written; logging failure is not suppressed.
        TypeError: The supplied record contains unsupported JSON values.
    """
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    agent = re.sub(r"[^A-Za-z0-9_-]", "_", record["agent"])
    stem = f"{stamp}_{agent}_{uuid4().hex[:8]}"
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    (root / f"{stem}.json").write_text(payload + "\n", encoding="utf-8")
    response = record["provider_response"]
    if response is None:
        response = record["response"]
    sections = [
        f"# {record['agent']} - {record['operation']}\n",
        f"Status: **{record['status']}** | Time: {record['elapsed_seconds']:.3f} s\n",
        f"[Run index](README.md) | [Complete JSON]({stem}.json)\n",
    ]
    if record["error"] is not None:
        sections += ["## Error\n", fenced(record["error"])]
    if response is not None:
        reasoning = response.get("reasoning_content")
        sections += ["## Reasoning returned by the server\n"]
        sections += [
            fenced(reasoning) if reasoning else "No reasoning text was returned.\n"
        ]
        sections += ["## Final output\n", fenced(response.get("content", ""))]
        sections += [
            "## Tool calls\n",
            fenced(json.dumps(response.get("tool_calls"), indent=2), "json"),
            "## Response metadata\n",
            fenced(json.dumps({
                key: value for key, value in response.items()
                if key not in {"reasoning_content", "content", "tool_calls"}
            }, indent=2), "json"),
        ]
    sections += [
        "## Complete request\n",
        "<details>\n<summary>Expand full prompts, tool results, and settings</summary>\n",
        fenced(json.dumps(record["request"], ensure_ascii=False, indent=2), "json"),
        "</details>\n",
        "## Actual provider request\n",
        "<details>\n<summary>Expand exact request sent to the server</summary>\n",
        fenced(json.dumps(record["provider_request"], ensure_ascii=False, indent=2), "json"),
        "</details>\n",
    ]
    readable = root / f"{stem}.md"
    readable.write_text("\n".join(sections), encoding="utf-8")
    index = root / "README.md"
    with _INDEX_LOCK:
        if not index.exists():
            index.write_text(
                "# Agent call log\n\nEach link contains one complete exchange. "
                "Reasoning is only the text returned by the server.\n\n"
                "[Executed tool events](tool_events.md)\n\n"
                "| Time (UTC) | Agent | Operation | Status | Seconds |\n"
                "| --- | --- | --- | --- | ---: |\n",
                encoding="utf-8",
            )
        with index.open("a", encoding="utf-8") as handle:
            handle.write(
                f"| {stamp} | [{agent}]({stem}.md) | {record['operation']} | "
                f"{record['status']} | {record['elapsed_seconds']:.3f} |\n"
            )
    return readable


def write_tool_event(root: Path, event: str, name: str, value: Any) -> None:
    """Append a complete executed tool invocation or result to the run log.

    Args:
        root: The same run-local directory used by write_call_trace.
        event: Event category, such as invocation or result.
        name: Tool name as emitted by the application.
        value: Complete arguments or result; strings remain unchanged.

    Raises:
        OSError: The event record cannot be written.
    """
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    entry = {"time": stamp, "event": event, "tool": name, "value": value}
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    with _INDEX_LOCK:
        with (root / "tool_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with (root / "tool_events.md").open("a", encoding="utf-8") as handle:
            handle.write(f"## {stamp} - {event}: {name}\n\n{fenced(text, 'json')}\n")
