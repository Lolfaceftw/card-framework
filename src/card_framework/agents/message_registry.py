"""
Message Registry
=================
Tracks speaker messages incrementally during summarizer generation.
Each message is assigned a 1-indexed line number for prompt-facing
operations, while a stable turn_id remains the durable identity used by
audio-backed drafting flows.
"""

from __future__ import annotations

import uuid

from card_framework.agents.utils import count_words
from card_framework.audio_pipeline.calibration import VoiceCloneCalibration
from card_framework.shared.summary_xml import DEFAULT_EMO_PRESET, SummaryTurn


class MessageRegistry:
    """In-memory registry of speaker messages built up during one summarizer run."""

    def __init__(self) -> None:
        # Each entry: {"speaker_id": str, "content": str, "emo_preset": str}
        self._messages: list[dict] = []

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def add(
        self,
        speaker_id: str,
        content: str,
        emo_preset: str = DEFAULT_EMO_PRESET,
        *,
        turn_id: str | None = None,
    ) -> dict:
        """Append a speaker message and return its line info."""
        normalized_preset = emo_preset.strip() or DEFAULT_EMO_PRESET
        resolved_turn_id = (turn_id or uuid.uuid4().hex).strip()
        if not resolved_turn_id:
            resolved_turn_id = uuid.uuid4().hex
        self._messages.append(
            {
                "turn_id": resolved_turn_id,
                "speaker_id": speaker_id,
                "content": content,
                "emo_preset": normalized_preset,
            }
        )
        line = len(self._messages)
        wc = count_words(content)
        return {
            "line": line,
            "turn_id": resolved_turn_id,
            "speaker_id": speaker_id,
            "emo_preset": normalized_preset,
            "word_count": wc,
        }

    def get_counts(self) -> dict:
        """Return full word-count breakdown: total, per-message, per-speaker."""
        messages = []
        per_speaker: dict[str, int] = {}
        total = 0

        for idx, msg in enumerate(self._messages, start=1):
            wc = count_words(msg["content"])
            total += wc
            messages.append(
                {
                    "line": idx,
                    "turn_id": msg["turn_id"],
                    "speaker_id": msg["speaker_id"],
                    "emo_preset": msg.get("emo_preset", DEFAULT_EMO_PRESET),
                    "word_count": wc,
                }
            )
            per_speaker[msg["speaker_id"]] = per_speaker.get(msg["speaker_id"], 0) + wc

        return {
            "total_word_count": total,
            "messages": messages,
            "per_speaker_totals": per_speaker,
        }

    def get_duration_breakdown(self, calibration: VoiceCloneCalibration) -> dict:
        """Return estimated duration breakdown derived from calibrated WPM."""
        messages = []
        per_speaker: dict[str, float] = {}
        total_seconds = 0.0

        for idx, msg in enumerate(self._messages, start=1):
            turn = SummaryTurn(
                speaker=str(msg["speaker_id"]),
                text=str(msg["content"]),
                emo_preset=str(msg.get("emo_preset", DEFAULT_EMO_PRESET)),
            )
            estimated_seconds = calibration.estimate_turn_seconds(turn)
            total_seconds += estimated_seconds
            messages.append(
                {
                    "line": idx,
                    "turn_id": turn.turn_id if hasattr(turn, "turn_id") else msg["turn_id"],
                    "speaker_id": turn.speaker,
                    "emo_preset": turn.emo_preset,
                    "word_count": count_words(turn.text),
                    "estimated_seconds": round(estimated_seconds, 3),
                }
            )
            per_speaker[turn.speaker] = round(
                per_speaker.get(turn.speaker, 0.0) + estimated_seconds,
                3,
            )

        return {
            "total_estimated_seconds": round(total_seconds, 3),
            "messages": messages,
            "per_speaker_estimated_seconds": per_speaker,
        }

    def edit(
        self,
        line: int,
        new_content: str,
        emo_preset: str | None = None,
    ) -> dict:
        """Replace content at *line* (1-indexed). Returns old/new counts."""
        idx = line - 1
        if idx < 0 or idx >= len(self._messages):
            return {
                "error": f"Line {line} does not exist.",
                "error_code": "line_not_found",
                "line": line,
            }

        old_content = self._messages[idx]["content"]
        old_emo_preset = str(
            self._messages[idx].get("emo_preset", DEFAULT_EMO_PRESET)
        )
        old_wc = count_words(old_content)
        new_emo_preset = old_emo_preset
        turn_id = str(self._messages[idx].get("turn_id", "")).strip() or uuid.uuid4().hex
        self._messages[idx]["turn_id"] = turn_id
        if emo_preset is not None:
            normalized_preset = emo_preset.strip()
            if normalized_preset:
                new_emo_preset = normalized_preset
        if old_content == new_content and old_emo_preset == new_emo_preset:
            return {
                "line": line,
                "turn_id": turn_id,
                "old_word_count": old_wc,
                "new_word_count": old_wc,
                "delta_words": 0,
                "old_emo_preset": old_emo_preset,
                "new_emo_preset": old_emo_preset,
                "changed": False,
                "stagnation_hint": "no_change",
                "total_word_count": self.get_counts()["total_word_count"],
            }

        self._messages[idx]["content"] = new_content
        self._messages[idx]["emo_preset"] = new_emo_preset
        new_wc = count_words(new_content)

        return {
            "line": line,
            "turn_id": turn_id,
            "old_word_count": old_wc,
            "new_word_count": new_wc,
            "delta_words": new_wc - old_wc,
            "old_emo_preset": old_emo_preset,
            "new_emo_preset": new_emo_preset,
            "changed": True,
            "total_word_count": self.get_counts()["total_word_count"],
        }

    def remove(self, line: int) -> dict:
        """Remove message at *line* (1-indexed). Remaining lines shift down."""
        idx = line - 1
        if idx < 0 or idx >= len(self._messages):
            return {"error": f"Line {line} does not exist."}

        removed = self._messages.pop(idx)
        return {
            "removed_line": line,
            "removed_turn_id": removed.get("turn_id", ""),
            "removed_speaker": removed["speaker_id"],
            "removed_emo_preset": removed.get("emo_preset", DEFAULT_EMO_PRESET),
            "total_word_count": self.get_counts()["total_word_count"],
            "remaining_messages": len(self._messages),
        }

    def get_message(self, line: int) -> dict:
        """Return one message snapshot by 1-indexed line number."""
        idx = line - 1
        if idx < 0 or idx >= len(self._messages):
            return {
                "error": f"Line {line} does not exist.",
                "error_code": "line_not_found",
                "line": line,
            }
        message = self._messages[idx]
        return {
            "line": line,
            "turn_id": message["turn_id"],
            "speaker_id": message["speaker_id"],
            "content": message["content"],
            "emo_preset": message.get("emo_preset", DEFAULT_EMO_PRESET),
            "word_count": count_words(message["content"]),
        }

    def get_all(self) -> list[SummaryTurn]:
        """Return all messages as typed summary turns for final assembly."""
        return [
            SummaryTurn(
                speaker=str(m["speaker_id"]),
                text=str(m["content"]),
                emo_preset=str(m.get("emo_preset", DEFAULT_EMO_PRESET)),
            )
            for m in self._messages
        ]

    def snapshot(self) -> list[dict]:
        """Return a deep copy of all messages for persistence."""
        return [
            {
                "line": idx,
                "turn_id": m["turn_id"],
                "speaker_id": m["speaker_id"],
                "content": m["content"],
                "emo_preset": m.get("emo_preset", DEFAULT_EMO_PRESET),
            }
            for idx, m in enumerate(self._messages, start=1)
        ]

    def load_from_snapshot(self, snapshot: list[dict]) -> None:
        """Restore registry state from a previous snapshot."""
        self._messages = [
            {
                "turn_id": str(m.get("turn_id", "")).strip() or uuid.uuid4().hex,
                "speaker_id": m["speaker_id"],
                "content": m["content"],
                "emo_preset": m.get("emo_preset", DEFAULT_EMO_PRESET),
            }
            for m in snapshot
        ]

    def __len__(self) -> int:
        return len(self._messages)

