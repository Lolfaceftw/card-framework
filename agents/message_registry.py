"""
Message Registry
=================
Tracks speaker messages incrementally during summarizer generation.
Each message is assigned a 1-indexed line number. Provides word-count
breakdowns and supports surgical edits / removals so the LLM can
stay within its duration budget while preserving telemetry.
"""

from agents.utils import count_words
from audio_pipeline.calibration import VoiceCloneCalibration
from summary_xml import DEFAULT_EMO_PRESET, SummaryTurn


class MessageRegistry:
    """In-memory registry of speaker messages built up during one summarizer run."""

    def __init__(self) -> None:
        # Each entry: {"speaker_id": str, "content": str, "emo_preset": str}
        self._messages: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────

    def add(
        self,
        speaker_id: str,
        content: str,
        emo_preset: str = DEFAULT_EMO_PRESET,
    ) -> dict:
        """Append a speaker message and return its line info."""
        normalized_preset = emo_preset.strip() or DEFAULT_EMO_PRESET
        self._messages.append(
            {
                "speaker_id": speaker_id,
                "content": content,
                "emo_preset": normalized_preset,
            }
        )
        line = len(self._messages)
        wc = count_words(content)
        return {
            "line": line,
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
        if emo_preset is not None:
            normalized_preset = emo_preset.strip()
            if normalized_preset:
                new_emo_preset = normalized_preset
        if old_content == new_content and old_emo_preset == new_emo_preset:
            return {
                "line": line,
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
            "removed_speaker": removed["speaker_id"],
            "removed_emo_preset": removed.get("emo_preset", DEFAULT_EMO_PRESET),
            "total_word_count": self.get_counts()["total_word_count"],
            "remaining_messages": len(self._messages),
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
                "speaker_id": m["speaker_id"],
                "content": m["content"],
                "emo_preset": m.get("emo_preset", DEFAULT_EMO_PRESET),
            }
            for m in snapshot
        ]

    def __len__(self) -> int:
        return len(self._messages)
