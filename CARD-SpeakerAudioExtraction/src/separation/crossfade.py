"""
Cross-fade utilities for seamless audio concatenation.

This module provides functions to apply linear cross-fades between audio segments
to avoid clicks and pops when joining speaker segments.
"""

import logging
from typing import List

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def apply_crossfade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    fade_duration_ms: float = 25.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Apply linear cross-fade between two audio segments.
    
    The end of audio1 fades out while the beginning of audio2 fades in,
    creating a smooth transition.
    
    Args:
        audio1: First audio segment.
        audio2: Second audio segment to fade in.
        fade_duration_ms: Duration of the cross-fade in milliseconds.
        sample_rate: Sample rate of the audio.
        
    Returns:
        Concatenated audio with cross-fade applied.
    """
    # Convert fade duration to samples
    fade_samples = int(fade_duration_ms * sample_rate / 1000.0)
    
    # Handle edge cases
    if len(audio1) == 0:
        return audio2.copy()
    if len(audio2) == 0:
        return audio1.copy()
    
    # If fade is longer than either segment, use the shorter segment length
    fade_samples = min(fade_samples, len(audio1), len(audio2))
    
    # If segments are too short for cross-fade, just concatenate
    if fade_samples <= 1:
        return np.concatenate([audio1, audio2])
    
    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    
    # Apply cross-fade
    # Part 1: audio1 before fade region
    part1 = audio1[:-fade_samples]
    
    # Part 2: Cross-fade region
    fade_region = (
        audio1[-fade_samples:] * fade_out +
        audio2[:fade_samples] * fade_in
    )
    
    # Part 3: audio2 after fade region
    part3 = audio2[fade_samples:]
    
    # Concatenate all parts
    result = np.concatenate([part1, fade_region, part3])
    
    return result


def concatenate_with_crossfade(
    segments: List[np.ndarray],
    fade_duration_ms: float = 25.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Concatenate multiple audio segments with cross-fade at boundaries.
    
    Args:
        segments: List of audio segments to concatenate.
        fade_duration_ms: Duration of cross-fade in milliseconds.
        sample_rate: Sample rate of the audio.
        
    Returns:
        Concatenated audio with cross-fades between all segments.
    """
    if not segments:
        return np.array([], dtype=np.float32)
    
    if len(segments) == 1:
        return segments[0].copy()
    
    # Start with the first segment
    result = segments[0].copy()
    
    # Apply cross-fade for each subsequent segment
    for i in range(1, len(segments)):
        result = apply_crossfade(
            result,
            segments[i],
            fade_duration_ms=fade_duration_ms,
            sample_rate=sample_rate
        )
    
    logger.debug(
        f"Concatenated {len(segments)} segments with {fade_duration_ms}ms cross-fades"
    )
    
    return result
