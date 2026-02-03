"""
Speech Separation Module.

This module provides functionality to separate overlapping speech sources
from mixed audio recordings using:
1. Targeted separation (CARD methodology - recommended for overlapping speech)
2. Diarization-guided extraction (recommended when diarization data is available)
3. Blind source separation using DPRNN-TasNet, Conv-TasNet, or SepFormer
"""

from .separator import SpeechSeparator
from .diarization_separator import DiarizationGuidedSeparator, separate_with_diarization
from .targeted_separator import TargetedSpeakerSeparator
from .enrollment import EnrollmentEmbeddingExtractor, SpeakerAssigner
from .crossfade import apply_crossfade, concatenate_with_crossfade
from .utils import load_audio, save_separated_audio, extract_speaker_count

__all__ = [
    'SpeechSeparator',
    'DiarizationGuidedSeparator',
    'separate_with_diarization',
    'TargetedSpeakerSeparator',
    'EnrollmentEmbeddingExtractor',
    'SpeakerAssigner',
    'apply_crossfade',
    'concatenate_with_crossfade',
    'load_audio',
    'save_separated_audio',
    'extract_speaker_count'
]
