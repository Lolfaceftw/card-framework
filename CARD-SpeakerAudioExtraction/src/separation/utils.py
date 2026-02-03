"""
Utility functions for audio I/O and helper operations.

This module provides helper functions for loading audio, saving separated
audio sources, extracting speaker counts from diarization JSON, and
ensuring output directories exist.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load and resample audio to target sample rate.

    Args:
        path: Path to the audio file.
        target_sr: Target sample rate (default: 16000).

    Returns:
        Tuple of (waveform tensor, sample rate).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If the audio file cannot be loaded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # Try torchaudio first
        waveform, sample_rate = torchaudio.load(path)
    except Exception:
        # Fallback to soundfile
        try:
            data, sample_rate = sf.read(path)
            # Convert to tensor and add channel dimension
            if data.ndim == 1:
                waveform = torch.from_numpy(data).float().unsqueeze(0)
            elif data.ndim == 2:
                # soundfile returns (samples, channels), transpose to (channels, samples)
                waveform = torch.from_numpy(data.T).float()
            else:
                # Handle unexpected dimensions by flattening to mono
                waveform = torch.from_numpy(data.flatten()).float().unsqueeze(0)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {path}. Error: {e}")

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr
        )
        waveform = resampler(waveform)

    return waveform, target_sr


def save_separated_audio(
    sources: np.ndarray,
    output_dir: str,
    base_filename: str,
    sample_rate: int = 16000
) -> list:
    """
    Save N separated audio sources as WAV files.

    Args:
        sources: Array of separated sources with shape (num_sources, num_samples).
        output_dir: Output directory path.
        base_filename: Base filename for output files.
        sample_rate: Sample rate for output files (default: 16000).

    Returns:
        List of paths to saved files.

    Raises:
        ValueError: If sources array has invalid shape.
    """
    if sources.ndim != 2:
        raise ValueError(f"Expected 2D array of sources, got shape {sources.shape}")

    ensure_output_directory(output_dir)

    saved_paths = []
    num_sources = sources.shape[0]

    for i in range(num_sources):
        filename = f"speaker_{i:02d}.wav"
        output_path = os.path.join(output_dir, filename)

        # Get the source audio
        source_audio = sources[i]

        # Normalize to prevent clipping
        max_val = np.max(np.abs(source_audio))
        if max_val > 0:
            source_audio = source_audio / max_val * 0.95

        # Save as WAV file
        sf.write(output_path, source_audio, sample_rate)
        saved_paths.append(output_path)

    return saved_paths


def extract_speaker_count(diarization_json_path: str) -> int:
    """
    Parse diarization JSON and count unique speakers.

    Args:
        diarization_json_path: Path to the diarization JSON file.

    Returns:
        Number of unique speakers found in the diarization.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the JSON file is invalid.
    """
    if not os.path.exists(diarization_json_path):
        raise FileNotFoundError(f"Diarization JSON not found: {diarization_json_path}")

    with open(diarization_json_path, 'r') as f:
        diarization_data = json.load(f)

    # Extract unique speakers
    speakers = set()
    for segment in diarization_data:
        if 'speaker' in segment:
            speakers.add(segment['speaker'])

    return len(speakers)


def ensure_output_directory(output_path: str) -> None:
    """
    Create output directory if it does not exist.

    Args:
        output_path: Path to the output directory.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)


def get_audio_duration(path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        # Fallback to soundfile
        try:
            data, sample_rate = sf.read(path)
            return len(data) / sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to get audio duration: {path}. Error: {e}")
