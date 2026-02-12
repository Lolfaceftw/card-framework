"""Constants for IndexTTS2 voice cloning benchmark."""

from __future__ import annotations

from typing import Final

WAVLM_SPEAKER_MODEL_ID: Final[str] = "microsoft/wavlm-base-plus-sv"
DEFAULT_BOOTSTRAP_SAMPLES: Final[int] = 1000
RESEARCH_CUTOFF_DATE: Final[str] = "2026-02-12"
MOS_SCORE_MIN: Final[int] = 1
MOS_SCORE_MAX: Final[int] = 5
CMOS_SCORE_MIN: Final[int] = -3
CMOS_SCORE_MAX: Final[int] = 3

