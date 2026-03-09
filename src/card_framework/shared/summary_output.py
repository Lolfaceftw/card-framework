"""Helpers for persisting final summarization artifacts."""

from pathlib import Path


def write_summary_xml_to_workspace(summary_xml: str, workspace_root: Path) -> Path:
    """Persist an XML summary at workspace root as ``summary.xml``."""
    output_path = workspace_root / "summary.xml"
    normalized_summary = summary_xml.strip()
    output_path.write_text(
        f"{normalized_summary}\n" if normalized_summary else "",
        encoding="utf-8",
    )
    return output_path
