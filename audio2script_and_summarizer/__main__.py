"""
Entry point for running the audio2script_and_summarizer package as a module.

Usage:
    uv run --extra audio2script python -m audio2script_and_summarizer [args]
"""

from .run_pipeline import main

if __name__ == "__main__":
    main()
