"""
Audio2Script + Summarizer Pipeline

An end-to-end pipeline that ingests raw podcast audio,
performs speaker diarization, and outputs a summarized script
with speaker-aware emotion annotations.

Usage with uv:
    uv run --extra audio2script python -m audio2script_and_summarizer \
        --input "path/to/podcast.wav" --openai-key "sk-..."
"""

# Package exports are lazy-loaded to avoid import errors when running as a module
# The heavy imports happen in the actual scripts (diarize.py, summarizer.py)

__version__ = "0.1.0"
__all__ = ["run_pipeline", "diarize", "summarizer"]
