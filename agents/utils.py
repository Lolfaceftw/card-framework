import json
import os
import re

import hydra


def count_words(text: str) -> int:
    """Deterministic tool — counts words, ignoring XML/HTML tags."""
    if not text:
        return 0
    clean_text = re.sub(r"<[^>]+>", "", text)
    return len(clean_text.split())


def format_transcript_for_prompt(transcript_data: dict) -> str:
    """Parses the raw JSON transcript into a readable format for the LLM prompt."""
    formatted_text = ""
    for segment in transcript_data.get("segments", []):
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "")
        formatted_text += f"[{speaker}]: {text}\n"
    return formatted_text


def load_transcript(path: str) -> dict:
    """Load transcript from a JSON file (resolved relative to the *original* cwd)."""
    original_cwd = hydra.utils.get_original_cwd()
    full_path = os.path.join(original_cwd, path)
    with open(full_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)
