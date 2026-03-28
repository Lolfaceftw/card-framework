"""Helpers for persisting final summarization artifacts."""

from pathlib import Path


def write_summary_xml(summary_xml: str, output_path: Path) -> Path:
    """Persist one XML summary atomically to the requested output path."""
    normalized_summary = summary_xml.strip()
    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved_output_path.with_suffix(f"{resolved_output_path.suffix}.tmp")
    temp_path.write_text(
        f"{normalized_summary}\n" if normalized_summary else "",
        encoding="utf-8",
    )
    temp_path.replace(resolved_output_path)
    return resolved_output_path


def write_summary_xml_to_workspace(summary_xml: str, workspace_root: Path) -> Path:
    """Persist an XML summary at workspace root as ``summary.xml``."""
    return write_summary_xml(summary_xml, workspace_root / "summary.xml")
