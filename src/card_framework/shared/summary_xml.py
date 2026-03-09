"""Summary XML parsing and serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree
from xml.sax.saxutils import escape, quoteattr

DEFAULT_EMO_PRESET = "neutral"


@dataclass(slots=True, frozen=True)
class SummaryTurn:
    """Represent one speaker-tagged summary turn."""

    speaker: str
    text: str
    emo_preset: str = DEFAULT_EMO_PRESET

    def __post_init__(self) -> None:
        if not self.speaker.strip():
            raise ValueError("speaker must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        if not self.emo_preset.strip():
            raise ValueError("emo_preset must be non-empty")


def parse_summary_xml(summary_xml: str) -> list[SummaryTurn]:
    """Parse speaker-tagged summary XML fragments into typed turns."""
    normalized = summary_xml.strip()
    if not normalized:
        return []

    wrapped_xml = f"<summary>{normalized}</summary>"
    try:
        root = ElementTree.fromstring(wrapped_xml)
    except ElementTree.ParseError as exc:
        raise ValueError("Summary XML is not well-formed.") from exc

    turns: list[SummaryTurn] = []
    for child in root:
        speaker = str(child.tag).strip()
        text = "".join(child.itertext()).strip()
        if not speaker or not text:
            continue
        emo_preset = str(
            child.attrib.get("emo_preset", DEFAULT_EMO_PRESET)
        ).strip() or DEFAULT_EMO_PRESET
        turns.append(
            SummaryTurn(
                speaker=speaker,
                text=text,
                emo_preset=emo_preset,
            )
        )
    return turns


def serialize_summary_turns(turns: list[SummaryTurn]) -> str:
    """Serialize summary turns into speaker-tagged XML fragments."""
    return "\n".join(
        (
            f"<{turn.speaker} emo_preset={quoteattr(turn.emo_preset)}>"
            f"{escape(turn.text)}</{turn.speaker}>"
        )
        for turn in turns
    )


def count_summary_turns(summary_xml: str) -> int:
    """Return the number of speaker-tagged turns in summary XML."""
    return len(parse_summary_xml(summary_xml))
