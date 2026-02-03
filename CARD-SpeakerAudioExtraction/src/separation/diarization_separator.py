"""
Diarization-guided speaker extraction module.

This module provides the DiarizationGuidedSeparator class that extracts
individual speaker audio tracks using diarization timestamps rather than
blind source separation.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .utils import ensure_output_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TIME_TOLERANCE = 0.001  # Tolerance for time comparison in seconds
NORMALIZATION_FACTOR = 0.95  # Factor for audio normalization to prevent clipping


@dataclass
class DiarizationSegment:
    """Represents a single diarization segment."""
    start: float
    end: float
    speaker: str
    text: Optional[str] = None

    @property
    def duration(self) -> float:
        """Return duration of the segment in seconds."""
        return self.end - self.start


@dataclass
class OverlapRegion:
    """Represents an overlapping speech region."""
    start: float
    end: float
    speakers: List[str]

    @property
    def duration(self) -> float:
        """Return duration of the overlap in seconds."""
        return self.end - self.start


class DiarizationGuidedSeparator:
    """
    Extract individual speakers using diarization timestamps.

    This approach:
    1. Loads diarization JSON with speaker timestamps
    2. Extracts non-overlapping segments for each speaker
    3. For overlapping segments, applies configurable handling strategy
    4. Stitches segments together into complete speaker tracks

    Attributes:
        sample_rate: Target sample rate for output audio (default: 16000).
        handle_overlap: Strategy for handling overlapping speech
            ('skip', 'mix', or 'both').
        preserve_timing: If True, maintains original timing with silence gaps.
        min_segment_duration: Minimum segment duration in seconds to include.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        handle_overlap: str = 'skip',
        preserve_timing: bool = True,
        min_segment_duration: float = 0.0
    ):
        """
        Initialize the DiarizationGuidedSeparator.

        Args:
            sample_rate: Target sample rate for output audio.
            handle_overlap: Strategy for overlapping speech:
                - 'skip': Leave silence in overlapping regions
                - 'mix': Include audio for all speakers in overlaps
                - 'both': Include overlap audio for all involved speakers
            preserve_timing: If True, maintain original timing (with silence).
                If False, create compact audio without gaps.
            min_segment_duration: Minimum duration for segments to include.
        """
        if handle_overlap not in ('skip', 'mix', 'both'):
            raise ValueError(
                f"Invalid handle_overlap: {handle_overlap}. "
                f"Choose from: 'skip', 'mix', 'both'"
            )

        self.sample_rate = sample_rate
        self.handle_overlap = handle_overlap
        self.preserve_timing = preserve_timing
        self.min_segment_duration = min_segment_duration

        logger.info(f"Initialized DiarizationGuidedSeparator")
        logger.info(f"  Sample rate: {sample_rate}")
        logger.info(f"  Overlap handling: {handle_overlap}")
        logger.info(f"  Preserve timing: {preserve_timing}")
        logger.info(f"  Min segment duration: {min_segment_duration}s")

    def load_diarization(self, diarization_path: str) -> List[DiarizationSegment]:
        """
        Load diarization data from JSON file.

        Args:
            diarization_path: Path to the diarization JSON file.

        Returns:
            List of DiarizationSegment objects.

        Raises:
            FileNotFoundError: If the diarization file doesn't exist.
            ValueError: If the JSON format is invalid.
        """
        if not os.path.exists(diarization_path):
            raise FileNotFoundError(
                f"Diarization file not found: {diarization_path}"
            )

        logger.info(f"Loading diarization from: {diarization_path}")

        with open(diarization_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                "Invalid diarization format: expected a list of segments"
            )

        segments = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid segment format at index {i}: expected dict"
                )

            required_fields = ['start', 'end', 'speaker']
            for field in required_fields:
                if field not in item:
                    raise ValueError(
                        f"Missing required field '{field}' in segment {i}"
                    )

            segment = DiarizationSegment(
                start=float(item['start']),
                end=float(item['end']),
                speaker=str(item['speaker']),
                text=item.get('text')
            )
            segments.append(segment)

        logger.info(f"Loaded {len(segments)} diarization segments")
        return segments

    def get_speakers(self, segments: List[DiarizationSegment]) -> List[str]:
        """
        Extract unique speaker IDs from diarization segments.

        Args:
            segments: List of diarization segments.

        Returns:
            Sorted list of unique speaker IDs.
        """
        speakers = sorted(set(seg.speaker for seg in segments))
        logger.info(f"Found {len(speakers)} unique speakers: {speakers}")
        return speakers

    def detect_overlaps(
        self,
        segments: List[DiarizationSegment]
    ) -> List[OverlapRegion]:
        """
        Detect overlapping speech regions where multiple speakers talk.

        Args:
            segments: List of diarization segments.

        Returns:
            List of OverlapRegion objects.
        """
        if not segments:
            return []

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.start)

        overlaps = []
        active_segments: List[DiarizationSegment] = []

        # Create events for segment starts and ends
        events = []
        for seg in sorted_segments:
            events.append((seg.start, 'start', seg))
            events.append((seg.end, 'end', seg))

        # Sort events by time
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'start' else 1))

        for time, event_type, segment in events:
            if event_type == 'start':
                # Check for overlap with active segments
                for active in active_segments:
                    if active.speaker != segment.speaker:
                        overlap_start = max(segment.start, active.start)
                        overlap_end = min(segment.end, active.end)
                        if overlap_end > overlap_start:
                            # Check if this overlap already exists
                            existing = False
                            for existing_overlap in overlaps:
                                if (abs(existing_overlap.start - overlap_start) < TIME_TOLERANCE and
                                        abs(existing_overlap.end - overlap_end) < TIME_TOLERANCE):
                                    if segment.speaker not in existing_overlap.speakers:
                                        existing_overlap.speakers.append(segment.speaker)
                                    if active.speaker not in existing_overlap.speakers:
                                        existing_overlap.speakers.append(active.speaker)
                                    existing = True
                                    break
                            if not existing:
                                overlaps.append(OverlapRegion(
                                    start=overlap_start,
                                    end=overlap_end,
                                    speakers=[active.speaker, segment.speaker]
                                ))
                active_segments.append(segment)
            else:
                # Remove from active segments
                active_segments = [s for s in active_segments if s != segment]

        # Sort overlaps by start time
        overlaps.sort(key=lambda o: o.start)

        if overlaps:
            logger.info(f"Detected {len(overlaps)} overlapping regions")
            total_overlap_duration = sum(o.duration for o in overlaps)
            logger.info(f"Total overlap duration: {total_overlap_duration:.2f}s")
        else:
            logger.info("No overlapping speech detected")

        return overlaps

    def is_in_overlap(
        self,
        time: float,
        overlaps: List[OverlapRegion]
    ) -> Optional[OverlapRegion]:
        """
        Check if a time point falls within an overlap region.

        Args:
            time: Time point in seconds.
            overlaps: List of overlap regions.

        Returns:
            The OverlapRegion if time is within one, None otherwise.
        """
        for overlap in overlaps:
            if overlap.start <= time < overlap.end:
                return overlap
        return None

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to target sample rate.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Tuple of (audio_data as numpy array, sample_rate).
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Loading audio: {audio_path}")

        # Try soundfile first as it's more reliable for WAV files
        try:
            data, sample_rate = sf.read(audio_path)
            if data.ndim == 1:
                waveform = torch.from_numpy(data).float().unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T).float()
        except (RuntimeError, OSError, IOError, sf.SoundFileError) as e:
            # Fallback to torchaudio for other formats
            logger.debug(f"soundfile failed ({e}), trying torchaudio")
            waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        audio_data = waveform.squeeze(0).numpy()
        duration = len(audio_data) / self.sample_rate

        logger.info(f"Audio loaded: {duration:.2f}s at {self.sample_rate}Hz")
        return audio_data, self.sample_rate

    def extract_segment(
        self,
        audio: np.ndarray,
        start: float,
        end: float
    ) -> np.ndarray:
        """
        Extract an audio segment by time range.

        Args:
            audio: Full audio array.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            Audio segment as numpy array.
        """
        start_sample = int(start * self.sample_rate)
        end_sample = int(end * self.sample_rate)

        # Ensure we don't exceed audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        return audio[start_sample:end_sample]

    def separate(
        self,
        audio_path: str,
        diarization_path: str
    ) -> Dict[str, np.ndarray]:
        """
        Separate audio into speaker-specific tracks using diarization.

        Args:
            audio_path: Path to the input audio file.
            diarization_path: Path to the diarization JSON file.

        Returns:
            Dictionary mapping speaker IDs to audio arrays.
        """
        # Load diarization and audio
        segments = self.load_diarization(diarization_path)
        audio, _ = self.load_audio(audio_path)

        # Get unique speakers
        speakers = self.get_speakers(segments)

        # Detect overlaps
        overlaps = self.detect_overlaps(segments)

        # Get audio duration
        audio_duration = len(audio) / self.sample_rate

        # Initialize output arrays for each speaker
        if self.preserve_timing:
            speaker_audio = {
                speaker: np.zeros(len(audio), dtype=np.float32)
                for speaker in speakers
            }
        else:
            speaker_audio = {
                speaker: []
                for speaker in speakers
            }

        # Process each segment
        logger.info("Processing segments...")
        processed_count = 0

        for segment in segments:
            # Skip very short segments
            if segment.duration < self.min_segment_duration:
                continue

            # Check for overlap handling
            seg_start = segment.start
            seg_end = segment.end

            # Split segment into non-overlapping and overlapping parts
            current_time = seg_start
            while current_time < seg_end:
                # Check if current position is in an overlap
                overlap = self.is_in_overlap(current_time, overlaps)

                if overlap:
                    # Handle overlapping region
                    overlap_end = min(overlap.end, seg_end)

                    if self.handle_overlap == 'skip':
                        # Skip overlapping audio (leave silence)
                        pass
                    elif self.handle_overlap in ('mix', 'both'):
                        # Include audio for this speaker in overlap
                        chunk = self.extract_segment(audio, current_time, overlap_end)
                        if self.preserve_timing:
                            start_idx = int(current_time * self.sample_rate)
                            end_idx = start_idx + len(chunk)
                            if end_idx <= len(speaker_audio[segment.speaker]):
                                speaker_audio[segment.speaker][start_idx:end_idx] = chunk
                        else:
                            speaker_audio[segment.speaker].append(chunk)

                    current_time = overlap_end
                else:
                    # Find the end of this non-overlapping region
                    next_overlap_start = seg_end
                    for o in overlaps:
                        if o.start > current_time:
                            next_overlap_start = min(next_overlap_start, o.start)
                            break

                    chunk_end = min(next_overlap_start, seg_end)

                    # Extract non-overlapping audio
                    chunk = self.extract_segment(audio, current_time, chunk_end)

                    if self.preserve_timing:
                        start_idx = int(current_time * self.sample_rate)
                        end_idx = start_idx + len(chunk)
                        if end_idx <= len(speaker_audio[segment.speaker]):
                            speaker_audio[segment.speaker][start_idx:end_idx] = chunk
                    else:
                        speaker_audio[segment.speaker].append(chunk)

                    current_time = chunk_end

            processed_count += 1

        logger.info(f"Processed {processed_count} segments")

        # Concatenate segments for compact mode
        if not self.preserve_timing:
            for speaker in speakers:
                if speaker_audio[speaker]:
                    speaker_audio[speaker] = np.concatenate(speaker_audio[speaker])
                else:
                    speaker_audio[speaker] = np.array([], dtype=np.float32)

        return speaker_audio

    def process_and_save(
        self,
        audio_path: str,
        diarization_path: str,
        output_dir: str
    ) -> List[str]:
        """
        Separate audio and save speaker tracks to output directory.

        Args:
            audio_path: Path to the input audio file.
            diarization_path: Path to the diarization JSON file.
            output_dir: Directory to save separated audio files.

        Returns:
            List of paths to saved audio files.
        """
        # Get base filename
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        separation_output_dir = os.path.join(
            output_dir, f"{base_filename}_separation"
        )

        # Ensure output directory exists
        ensure_output_directory(separation_output_dir)

        # Run separation
        speaker_audio = self.separate(audio_path, diarization_path)

        # Save each speaker's audio
        saved_paths = []
        for speaker, audio_data in speaker_audio.items():
            if len(audio_data) == 0:
                logger.warning(f"No audio for speaker {speaker}, skipping")
                continue

            # Normalize audio
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * NORMALIZATION_FACTOR

            # Create output filename
            output_path = os.path.join(separation_output_dir, f"{speaker}.wav")

            # Save audio
            sf.write(output_path, audio_data, self.sample_rate)
            saved_paths.append(output_path)

            duration = len(audio_data) / self.sample_rate
            logger.info(f"Saved {speaker}: {output_path} ({duration:.2f}s)")

        logger.info(f"Saved {len(saved_paths)} speaker tracks to: {separation_output_dir}")

        return saved_paths


def separate_with_diarization(
    audio_path: str,
    diarization_path: str,
    output_dir: str,
    sample_rate: int = 16000,
    handle_overlap: str = 'skip',
    preserve_timing: bool = True,
    min_segment_duration: float = 0.0
) -> List[str]:
    """
    Convenience function to separate audio using diarization.

    Args:
        audio_path: Path to the input audio file.
        diarization_path: Path to the diarization JSON file.
        output_dir: Directory to save separated audio files.
        sample_rate: Target sample rate for output audio.
        handle_overlap: Strategy for overlapping speech ('skip', 'mix', 'both').
        preserve_timing: If True, maintain original timing with silence gaps.
        min_segment_duration: Minimum segment duration to include.

    Returns:
        List of paths to saved audio files.
    """
    separator = DiarizationGuidedSeparator(
        sample_rate=sample_rate,
        handle_overlap=handle_overlap,
        preserve_timing=preserve_timing,
        min_segment_duration=min_segment_duration
    )

    return separator.process_and_save(
        audio_path=audio_path,
        diarization_path=diarization_path,
        output_dir=output_dir
    )
