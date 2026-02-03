"""
Targeted speaker separator using diarization-guided SepFormer.

This module implements the complete CARD methodology for speaker separation:
1. Load diarization and detect overlaps
2. Create enrollment embeddings from clean segments
3. Extract non-overlapping segments directly
4. Process overlapping regions with SepFormer
5. Assign separated sources using embeddings
6. Concatenate segments with cross-fade
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .crossfade import concatenate_with_crossfade
from .diarization_separator import DiarizationSegment, OverlapRegion
from .enrollment import EnrollmentEmbeddingExtractor, SpeakerAssigner
from .separator import SpeechSeparator
from .utils import ensure_output_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
# Normalization factor to prevent clipping while maintaining loudness
# 0.95 leaves 5% headroom for safety
NORMALIZATION_FACTOR = 0.95


class TargetedSpeakerSeparator:
    """
    Targeted speaker separator using diarization-guided SepFormer.
    
    This class implements the CARD methodology for speaker separation by:
    - Using direct extraction for non-overlapping segments (artifact-free)
    - Running SepFormer only on overlapping regions (memory efficient)
    - Matching sources to speakers using enrollment embeddings
    
    Attributes:
        sample_rate: Sample rate for audio processing.
        similarity_threshold: Threshold for speaker assignment confidence.
        crossfade_ms: Cross-fade duration in milliseconds.
        overlap_padding_s: Padding around overlap windows in seconds.
        enrollment_duration_range: (min, max) duration for enrollment snippets.
        device: Device for inference.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        similarity_threshold: float = 0.7,
        crossfade_ms: float = 25.0,
        overlap_padding_s: float = 0.5,
        enrollment_duration_range: Tuple[float, float] = (3.0, 6.0),
        device: str = 'auto'
    ):
        """
        Initialize the TargetedSpeakerSeparator.
        
        Args:
            sample_rate: Sample rate for audio processing.
            similarity_threshold: Cosine similarity threshold for assignment.
            crossfade_ms: Cross-fade duration in milliseconds.
            overlap_padding_s: Padding for overlap windows in seconds.
            enrollment_duration_range: Duration range for enrollment snippets.
            device: Device for inference ('cpu', 'cuda', or 'auto').
        """
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.crossfade_ms = crossfade_ms
        self.overlap_padding_s = overlap_padding_s
        self.enrollment_duration_range = enrollment_duration_range
        self.device = self._resolve_device(device)
        
        # Initialize components
        self.enrollment_extractor = EnrollmentEmbeddingExtractor(
            sample_rate=sample_rate,
            device=device
        )
        self.speaker_assigner = SpeakerAssigner(threshold=similarity_threshold)
        
        # Lazy-load SepFormer only when needed
        self.sepformer = None
        
        logger.info(f"Initialized TargetedSpeakerSeparator")
        logger.info(f"  Sample rate: {sample_rate}Hz")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        logger.info(f"  Cross-fade: {crossfade_ms}ms")
        logger.info(f"  Overlap padding: {overlap_padding_s}s")
        logger.info(f"  Device: {self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def _load_sepformer(self) -> None:
        """Load SepFormer model (lazy loading)."""
        if self.sepformer is None:
            logger.info("Loading SepFormer model...")
            self.sepformer = SpeechSeparator(
                model_name='sepformer',
                device=self.device
            )
            self.sepformer.load_model()
            logger.info("SepFormer loaded successfully")
    
    def load_diarization(self, path: str) -> List[dict]:
        """
        Load and parse diarization JSON.
        
        Args:
            path: Path to diarization JSON file.
            
        Returns:
            List of segment dictionaries.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Diarization file not found: {path}")
        
        logger.info(f"Loading diarization from: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Invalid diarization format: expected a list")
        
        segments = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid segment at index {i}: expected dict")
            
            required_fields = ['start', 'end', 'speaker']
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing field '{field}' in segment {i}")
            
            segments.append({
                'start': float(item['start']),
                'end': float(item['end']),
                'speaker': str(item['speaker']),
                'text': item.get('text')
            })
        
        logger.info(f"Loaded {len(segments)} diarization segments")
        return segments
    
    def detect_overlaps(self, segments: List[dict]) -> List[dict]:
        """
        Detect overlapping speech regions.
        
        Args:
            segments: List of diarization segments.
            
        Returns:
            List of overlap region dictionaries.
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s['start'])
        
        overlaps = []
        active_segments: List[dict] = []
        
        # Create events for segment starts and ends
        events = []
        for seg in sorted_segments:
            events.append((seg['start'], 'start', seg))
            events.append((seg['end'], 'end', seg))
        
        # Sort events by time
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'start' else 1))
        
        for time, event_type, segment in events:
            if event_type == 'start':
                # Check for overlap with active segments
                for active in active_segments:
                    if active['speaker'] != segment['speaker']:
                        overlap_start = max(segment['start'], active['start'])
                        overlap_end = min(segment['end'], active['end'])
                        
                        if overlap_end > overlap_start:
                            # Check if this overlap already exists
                            existing = False
                            for existing_overlap in overlaps:
                                if (abs(existing_overlap['start'] - overlap_start) < 0.001 and
                                    abs(existing_overlap['end'] - overlap_end) < 0.001):
                                    if segment['speaker'] not in existing_overlap['speakers']:
                                        existing_overlap['speakers'].append(segment['speaker'])
                                    if active['speaker'] not in existing_overlap['speakers']:
                                        existing_overlap['speakers'].append(active['speaker'])
                                    existing = True
                                    break
                            
                            if not existing:
                                overlaps.append({
                                    'start': overlap_start,
                                    'end': overlap_end,
                                    'speakers': [active['speaker'], segment['speaker']]
                                })
                
                active_segments.append(segment)
            else:
                # Remove from active segments
                active_segments = [s for s in active_segments if s != segment]
        
        # Sort overlaps by start time
        overlaps.sort(key=lambda o: o['start'])
        
        if overlaps:
            logger.info(f"Detected {len(overlaps)} overlapping regions")
            total_duration = sum(o['end'] - o['start'] for o in overlaps)
            logger.info(f"Total overlap duration: {total_duration:.2f}s")
        else:
            logger.info("No overlapping speech detected")
        
        return overlaps
    
    def extract_non_overlapping(
        self,
        audio: np.ndarray,
        segments: List[dict],
        overlaps: List[dict]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Extract non-overlapping segments for each speaker.
        
        Args:
            audio: Full audio array.
            segments: List of diarization segments.
            overlaps: List of overlap regions.
            
        Returns:
            Dictionary mapping speaker IDs to lists of audio segments.
        """
        logger.info("Extracting non-overlapping segments...")
        
        speaker_segments = {}
        
        for segment in segments:
            speaker = segment['speaker']
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Split segment into non-overlapping parts
            current_time = seg_start
            
            while current_time < seg_end:
                # Check if current position is in an overlap
                in_overlap = False
                overlap_end = seg_end
                
                for overlap in overlaps:
                    if overlap['start'] <= current_time < overlap['end']:
                        in_overlap = True
                        overlap_end = overlap['end']
                        break
                
                if in_overlap:
                    # Skip the overlap region
                    current_time = overlap_end
                else:
                    # Find the next overlap start or segment end
                    next_overlap_start = seg_end
                    for overlap in overlaps:
                        if overlap['start'] > current_time:
                            next_overlap_start = min(next_overlap_start, overlap['start'])
                            break
                    
                    chunk_end = min(next_overlap_start, seg_end)
                    
                    # Extract non-overlapping audio
                    start_sample = int(current_time * self.sample_rate)
                    end_sample = int(chunk_end * self.sample_rate)
                    
                    # Ensure we don't exceed audio bounds
                    start_sample = max(0, start_sample)
                    end_sample = min(len(audio), end_sample)
                    
                    chunk = audio[start_sample:end_sample]
                    
                    if len(chunk) > 0:
                        if speaker not in speaker_segments:
                            speaker_segments[speaker] = []
                        speaker_segments[speaker].append(chunk)
                    
                    current_time = chunk_end
        
        # Log statistics
        for speaker, segs in speaker_segments.items():
            total_duration = sum(len(s) for s in segs) / self.sample_rate
            logger.info(
                f"Speaker {speaker}: {len(segs)} non-overlap segments "
                f"({total_duration:.2f}s total)"
            )
        
        return speaker_segments
    
    def process_overlap_window(
        self,
        audio: np.ndarray,
        overlap: dict,
        enrollment_embeddings: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Process a single overlap window with SepFormer.
        
        Args:
            audio: Full audio array.
            overlap: Overlap region dictionary.
            enrollment_embeddings: Enrollment embeddings for all speakers.
            
        Returns:
            Tuple of (assigned_sources dict, assignment_info dict).
        """
        # Load SepFormer if not already loaded
        self._load_sepformer()
        
        # Extract overlap window with padding
        overlap_start = overlap['start']
        overlap_end = overlap['end']
        
        padded_start = max(0, overlap_start - self.overlap_padding_s)
        padded_end = overlap_end + self.overlap_padding_s
        
        start_sample = int(padded_start * self.sample_rate)
        end_sample = int(padded_end * self.sample_rate)
        
        # Ensure we don't exceed audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        window = audio[start_sample:end_sample]
        
        # Run SepFormer on the window
        logger.info(
            f"Processing overlap [{overlap_start:.2f}s - {overlap_end:.2f}s] "
            f"with padding ({len(window)/self.sample_rate:.2f}s total)"
        )
        
        # Convert to torch tensor
        window_tensor = torch.from_numpy(window).float().unsqueeze(0)
        
        # Run separation using the public method
        separated = self.sepformer._separate_chunk(window_tensor)
        
        # Extract embeddings for each separated source
        source_embeddings = []
        for src_idx in range(separated.shape[0]):
            src_audio = separated[src_idx]
            src_embedding = self.enrollment_extractor.compute_embedding(src_audio)
            src_embedding = self.enrollment_extractor.normalize_embedding(src_embedding)
            source_embeddings.append(src_embedding)
        
        # Assign sources to speakers
        assignments = self.speaker_assigner.assign_sources(
            [separated[i] for i in range(separated.shape[0])],
            source_embeddings,
            enrollment_embeddings
        )
        
        # Calculate padding samples
        padding_start_samples = int((overlap_start - padded_start) * self.sample_rate)
        padding_end_samples = int((padded_end - overlap_end) * self.sample_rate)
        
        # Trim padding from assigned sources
        assigned_sources = {}
        for assignment in assignments:
            src_idx = assignment['source_idx']
            speaker = assignment['speaker']
            
            # Trim padding
            trimmed = separated[src_idx][padding_start_samples:]
            if padding_end_samples > 0:
                trimmed = trimmed[:-padding_end_samples]
            
            if speaker not in assigned_sources:
                assigned_sources[speaker] = trimmed
            else:
                # If multiple sources assigned to same speaker, take the one with higher similarity
                # This shouldn't happen often, but handle it gracefully
                logger.warning(
                    f"Multiple sources assigned to {speaker} in same overlap window"
                )
                # Keep the current one (already assigned)
        
        # Create assignment info for reporting
        assignment_info = {
            'overlap_start': overlap_start,
            'overlap_end': overlap_end,
            'assignments': assignments
        }
        
        return assigned_sources, assignment_info
    
    def separate(
        self,
        audio_path: str,
        diarization_path: str
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Separate audio into speaker-specific tracks.
        
        Args:
            audio_path: Path to input audio file.
            diarization_path: Path to diarization JSON file.
            
        Returns:
            Tuple of (speaker_audio dict, separation_stats dict).
        """
        logger.info("=" * 80)
        logger.info("TARGETED SPEAKER SEPARATION")
        logger.info("=" * 80)
        
        # Load audio
        logger.info(f"Loading audio: {audio_path}")
        data, sr = sf.read(audio_path)
        
        # Convert to mono if needed
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        
        # Resample if needed
        if sr != self.sample_rate:
            data_tensor = torch.from_numpy(data).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate
            )
            data_tensor = resampler(data_tensor)
            data = data_tensor.squeeze(0).numpy()
        
        audio_duration = len(data) / self.sample_rate
        logger.info(f"Audio duration: {audio_duration:.2f}s at {self.sample_rate}Hz")
        
        # Load diarization
        segments = self.load_diarization(diarization_path)
        
        # Get unique speakers
        speakers = sorted(set(seg['speaker'] for seg in segments))
        logger.info(f"Found {len(speakers)} unique speakers: {speakers}")
        
        # Detect overlaps
        overlaps = self.detect_overlaps(segments)
        
        # Create enrollment embeddings
        logger.info("=" * 80)
        logger.info("Creating enrollment embeddings...")
        enrollment_embeddings = self.enrollment_extractor.create_enrollment_embeddings(
            audio=data,
            segments=segments,
            overlaps=overlaps,
            duration_range=self.enrollment_duration_range
        )
        
        if not enrollment_embeddings:
            raise RuntimeError("Failed to create enrollment embeddings for any speaker")
        
        # Extract non-overlapping segments
        logger.info("=" * 80)
        speaker_segments = self.extract_non_overlapping(data, segments, overlaps)
        
        # Process overlapping segments
        logger.info("=" * 80)
        logger.info("Processing overlapping segments...")
        
        uncertain_assignments = []
        
        for overlap_idx, overlap in enumerate(overlaps):
            logger.info(f"\nProcessing overlap {overlap_idx + 1}/{len(overlaps)}")
            
            assigned_sources, assignment_info = self.process_overlap_window(
                audio=data,
                overlap=overlap,
                enrollment_embeddings=enrollment_embeddings
            )
            
            # Add to speaker segments
            for speaker, source_audio in assigned_sources.items():
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append(source_audio)
            
            # Track uncertain assignments
            for assignment in assignment_info['assignments']:
                if assignment['uncertain']:
                    uncertain_assignments.append({
                        'overlap_start': assignment_info['overlap_start'],
                        'overlap_end': assignment_info['overlap_end'],
                        'source_index': assignment['source_idx'],
                        'best_match': assignment['speaker'],
                        'similarity': assignment['similarity']
                    })
        
        # Construct speaker tracks with cross-fade
        logger.info("=" * 80)
        logger.info("Constructing final speaker tracks...")
        
        speaker_audio = {}
        per_speaker_duration = {}
        
        for speaker in speakers:
            if speaker in speaker_segments and speaker_segments[speaker]:
                # Concatenate with cross-fade
                track = concatenate_with_crossfade(
                    segments=speaker_segments[speaker],
                    fade_duration_ms=self.crossfade_ms,
                    sample_rate=self.sample_rate
                )
                speaker_audio[speaker] = track
                duration = len(track) / self.sample_rate
                per_speaker_duration[speaker] = duration
                
                logger.info(f"{speaker}: {duration:.2f}s")
            else:
                logger.warning(f"{speaker}: No audio segments found")
                speaker_audio[speaker] = np.array([], dtype=np.float32)
                per_speaker_duration[speaker] = 0.0
        
        # Compile statistics
        stats = {
            'speakers': speakers,
            'total_duration_s': audio_duration,
            'per_speaker_duration_s': per_speaker_duration,
            'overlaps_detected': len(overlaps),
            'overlaps_processed': len(overlaps),
            'uncertain_assignments': uncertain_assignments
        }
        
        logger.info("=" * 80)
        logger.info("SEPARATION COMPLETE")
        logger.info(f"Processed {len(overlaps)} overlaps")
        logger.info(f"Uncertain assignments: {len(uncertain_assignments)}")
        logger.info("=" * 80)
        
        return speaker_audio, stats
    
    def process_and_save(
        self,
        audio_path: str,
        diarization_path: str,
        output_dir: str,
        save_embeddings: bool = False
    ) -> List[str]:
        """
        Separate audio and save results to output directory.
        
        Args:
            audio_path: Path to input audio file.
            diarization_path: Path to diarization JSON file.
            output_dir: Directory to save outputs.
            save_embeddings: Whether to save enrollment embeddings.
            
        Returns:
            List of paths to saved files.
        """
        # Get base filename
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        separation_output_dir = os.path.join(
            output_dir, f"{base_filename}_separation"
        )
        
        # Ensure output directory exists
        ensure_output_directory(separation_output_dir)
        
        # Run separation
        speaker_audio, stats = self.separate(audio_path, diarization_path)
        
        # Save speaker audio files
        saved_paths = []
        
        for speaker, audio_data in speaker_audio.items():
            if len(audio_data) == 0:
                logger.warning(f"No audio for {speaker}, skipping")
                continue
            
            # Normalize audio
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * NORMALIZATION_FACTOR
            
            # Save audio
            output_path = os.path.join(separation_output_dir, f"{speaker}.wav")
            sf.write(output_path, audio_data, self.sample_rate)
            saved_paths.append(output_path)
            
            logger.info(f"Saved: {output_path}")
        
        # Save enrollment embeddings if requested
        if save_embeddings:
            # Need to recreate embeddings since we don't store them
            segments = self.load_diarization(diarization_path)
            overlaps = self.detect_overlaps(segments)
            
            # Load audio for embeddings
            data, sr = sf.read(audio_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != self.sample_rate:
                data_tensor = torch.from_numpy(data).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sample_rate
                )
                data_tensor = resampler(data_tensor)
                data = data_tensor.squeeze(0).numpy()
            
            enrollment_embeddings = self.enrollment_extractor.create_enrollment_embeddings(
                audio=data,
                segments=segments,
                overlaps=overlaps,
                duration_range=self.enrollment_duration_range
            )
            
            embeddings_path = os.path.join(
                separation_output_dir,
                "enrollment_embeddings.npy"
            )
            self.enrollment_extractor.save_embeddings(
                enrollment_embeddings,
                embeddings_path
            )
            saved_paths.append(embeddings_path)
        
        # Save separation report
        report_path = os.path.join(separation_output_dir, "separation_report.json")
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        saved_paths.append(report_path)
        
        logger.info(f"Saved separation report: {report_path}")
        logger.info(f"Total files saved: {len(saved_paths)}")
        
        return saved_paths
