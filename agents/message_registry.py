"""
Message Registry
=================
Tracks speaker messages incrementally during summarizer generation.
Each message is assigned a 1-indexed line number. Provides word-count
breakdowns and supports surgical edits / removals so the LLM can
stay within its word budget.
"""

from agents.utils import count_words


class MessageRegistry:
    """In-memory registry of speaker messages built up during one summarizer run."""

    def __init__(self) -> None:
        # Each entry: {"speaker_id": str, "content": str}
        self._messages: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────

    def add(self, speaker_id: str, content: str) -> dict:
        """Append a speaker message and return its line info."""
        self._messages.append({"speaker_id": speaker_id, "content": content})
        line = len(self._messages)
        wc = count_words(content)
        return {"line": line, "speaker_id": speaker_id, "word_count": wc}

    def get_counts(self) -> dict:
        """Return full word-count breakdown: total, per-message, per-speaker."""
        messages = []
        per_speaker: dict[str, int] = {}
        total = 0

        for idx, msg in enumerate(self._messages, start=1):
            wc = count_words(msg["content"])
            total += wc
            messages.append(
                {"line": idx, "speaker_id": msg["speaker_id"], "word_count": wc}
            )
            per_speaker[msg["speaker_id"]] = per_speaker.get(msg["speaker_id"], 0) + wc

        return {
            "total_word_count": total,
            "messages": messages,
            "per_speaker_totals": per_speaker,
        }

    def edit(self, line: int, new_content: str) -> dict:
        """Replace content at *line* (1-indexed). Returns old/new counts."""
        idx = line - 1
        if idx < 0 or idx >= len(self._messages):
            return {"error": f"Line {line} does not exist."}

        old_wc = count_words(self._messages[idx]["content"])
        self._messages[idx]["content"] = new_content
        new_wc = count_words(new_content)

        return {
            "line": line,
            "old_word_count": old_wc,
            "new_word_count": new_wc,
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
            "total_word_count": self.get_counts()["total_word_count"],
            "remaining_messages": len(self._messages),
        }

    def get_all(self) -> list[tuple[str, str]]:
        """Return all messages as (speaker_id, content) tuples for final assembly."""
        return [(m["speaker_id"], m["content"]) for m in self._messages]

    def snapshot(self) -> list[dict]:
        """Return a deep copy of all messages for persistence."""
        return [
            {"line": idx, "speaker_id": m["speaker_id"], "content": m["content"]}
            for idx, m in enumerate(self._messages, start=1)
        ]

    def load_from_snapshot(self, snapshot: list[dict]) -> None:
        """Restore registry state from a previous snapshot."""
        self._messages = [
            {"speaker_id": m["speaker_id"], "content": m["content"]} for m in snapshot
        ]

    def __len__(self) -> int:
        return len(self._messages)
