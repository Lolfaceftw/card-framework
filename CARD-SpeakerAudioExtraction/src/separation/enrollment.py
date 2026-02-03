"""
Enrollment embedding system for speaker identification.

This module provides functionality to:
1. Select clean non-overlapping speech segments for enrollment
2. Extract speaker embeddings using ECAPA-TDNN from SpeechBrain
3. Assign separated sources to speakers using cosine similarity
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class EnrollmentEmbeddingExtractor:
    """
    Extract enrollment embeddings from clean speech segments.
    
    This class uses ECAPA-TDNN from SpeechBrain to create reference embeddings
    for each speaker from non-overlapping speech segments.
    
    Attributes:
        sample_rate: Sample rate for audio processing (default: 16000).
        device: Device for inference ('cpu', 'cuda', or 'auto').
        model: ECAPA-TDNN model for embedding extraction.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        device: str = 'auto'
    ):
        """
        Initialize the EnrollmentEmbeddingExtractor.
        
        Args:
            sample_rate: Sample rate for audio processing.
            device: Device for inference ('cpu', 'cuda', or 'auto').
        """
        self.sample_rate = sample_rate
        self.device = self._resolve_device(device)
        self.model = None
        self._load_model()
        
        logger.info(f"Initialized EnrollmentEmbeddingExtractor")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Sample rate: {sample_rate}Hz")
    
    def _resolve_device(self, device: str) -> str:
        """
        Resolve the device string to an actual device.
        
        Args:
            device: Device specification ('cpu', 'cuda', 'auto').
            
        Returns:
            Resolved device string.
        """
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def _load_model(self) -> None:
        """
        Load the ECAPA-TDNN model from SpeechBrain.
        
        Raises:
            RuntimeError: If the model fails to load.
        """
        # Apply torchaudio compatibility fix before importing SpeechBrain
        self._apply_torchaudio_compat()
        
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise RuntimeError(
                f"SpeechBrain is required but could not be imported: {e}. "
                f"Please install speechbrain: pip install speechbrain"
            )
        
        try:
            # Load ECAPA-TDNN model from SpeechBrain
            savedir = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "speechbrain",
                "spkrec-ecapa-voxceleb"
            )
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=savedir,
                run_opts={"device": self.device}
            )
            logger.info("ECAPA-TDNN model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load ECAPA-TDNN model: {e}")
    
    def _apply_torchaudio_compat(self) -> None:
        """
        Apply torchaudio compatibility fix for newer versions.
        
        Newer torchaudio versions (>=2.1) removed `list_audio_backends` which
        SpeechBrain may require. This adds a compatibility shim.
        """
        import torchaudio
        try:
            # Check if the function exists
            _ = torchaudio.list_audio_backends
        except AttributeError:
            # Add compatibility shim for newer torchaudio versions
            # Return a reasonable default set of backends that are commonly available
            # Note: This is a compatibility workaround and doesn't affect actual audio loading
            # which is handled by the libraries' internal mechanisms
            torchaudio.list_audio_backends = lambda: ['soundfile', 'sox', 'ffmpeg']
            logger.debug("Applied torchaudio compatibility fix for list_audio_backends")
    
    def select_enrollment_snippets(
        self,
        segments: List[dict],
        overlaps: List[dict],
        duration_range: Tuple[float, float] = (3.0, 6.0)
    ) -> Dict[str, List[dict]]:
        """
        Select clean non-overlapping segments for enrollment.
        
        Args:
            segments: List of diarization segments with 'start', 'end', 'speaker'.
            overlaps: List of overlap regions with 'start', 'end', 'speakers'.
            duration_range: (min_duration, max_duration) in seconds for snippets.
            
        Returns:
            Dictionary mapping speaker IDs to lists of enrollment segments.
        """
        min_duration, max_duration = duration_range
        
        # Group segments by speaker
        speaker_segments = {}
        for seg in segments:
            speaker = seg['speaker']
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)
        
        # For each speaker, find clean segments
        enrollment_snippets = {}
        
        for speaker, segs in speaker_segments.items():
            clean_segments = []
            
            for seg in segs:
                seg_start = seg['start']
                seg_end = seg['end']
                seg_duration = seg_end - seg_start
                
                # Skip segments that are too short
                if seg_duration < min_duration:
                    continue
                
                # Check if segment overlaps with any overlap region
                is_clean = True
                for overlap in overlaps:
                    overlap_start = overlap['start']
                    overlap_end = overlap['end']
                    
                    # Check for any overlap
                    if not (seg_end <= overlap_start or seg_start >= overlap_end):
                        is_clean = False
                        break
                
                if is_clean:
                    # Truncate to max_duration if needed
                    if seg_duration > max_duration:
                        # Take the middle portion for better quality
                        mid_point = (seg_start + seg_end) / 2
                        use_start = mid_point - max_duration / 2
                        use_end = mid_point + max_duration / 2
                        clean_segments.append({
                            'start': use_start,
                            'end': use_end,
                            'speaker': speaker,
                            'duration': max_duration
                        })
                    else:
                        clean_segments.append({
                            'start': seg_start,
                            'end': seg_end,
                            'speaker': speaker,
                            'duration': seg_duration
                        })
            
            # Sort by duration (prefer longer segments for better embeddings)
            clean_segments.sort(key=lambda s: s['duration'], reverse=True)
            
            enrollment_snippets[speaker] = clean_segments
            
            total_duration = sum(s['duration'] for s in clean_segments)
            logger.info(
                f"Speaker {speaker}: Found {len(clean_segments)} clean segments "
                f"({total_duration:.2f}s total)"
            )
        
        return enrollment_snippets
    
    def compute_embedding(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Compute speaker embedding for an audio segment.
        
        Args:
            audio_segment: Audio data as numpy array.
            
        Returns:
            Speaker embedding as numpy array.
        """
        # Convert to torch tensor
        if isinstance(audio_segment, np.ndarray):
            audio_tensor = torch.from_numpy(audio_segment).float()
        else:
            audio_tensor = audio_segment.float()
        
        # Ensure correct shape: (batch, samples) or (samples,)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Move to device
        audio_tensor = audio_tensor.to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model.encode_batch(audio_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to an embedding.
        
        Args:
            embedding: Embedding vector.
            
        Returns:
            L2-normalized embedding.
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def create_enrollment_embeddings(
        self,
        audio: np.ndarray,
        segments: List[dict],
        overlaps: List[dict],
        duration_range: Tuple[float, float] = (3.0, 6.0)
    ) -> Dict[str, np.ndarray]:
        """
        Create enrollment embeddings for all speakers.
        
        Args:
            audio: Full audio as numpy array.
            segments: List of diarization segments.
            overlaps: List of overlap regions.
            duration_range: Duration range for enrollment snippets.
            
        Returns:
            Dictionary mapping speaker IDs to normalized embeddings.
        """
        logger.info("Creating enrollment embeddings...")
        
        # Select clean segments for each speaker
        enrollment_snippets = self.select_enrollment_snippets(
            segments, overlaps, duration_range
        )
        
        # Compute embeddings for each speaker
        speaker_embeddings = {}
        
        for speaker, snippets in enrollment_snippets.items():
            if not snippets:
                logger.warning(f"No clean segments found for {speaker}, skipping")
                continue
            
            # Use the first (longest) clean segment for enrollment
            # Could average multiple segments, but one good segment is often sufficient
            snippet = snippets[0]
            
            # Extract audio segment
            start_sample = int(snippet['start'] * self.sample_rate)
            end_sample = int(snippet['end'] * self.sample_rate)
            audio_segment = audio[start_sample:end_sample]
            
            # Compute and normalize embedding
            embedding = self.compute_embedding(audio_segment)
            normalized_embedding = self.normalize_embedding(embedding)
            
            speaker_embeddings[speaker] = normalized_embedding
            
            logger.info(
                f"Created embedding for {speaker} "
                f"({snippet['duration']:.2f}s segment, dim={len(normalized_embedding)})"
            )
        
        return speaker_embeddings
    
    def save_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        path: str
    ) -> None:
        """
        Save embeddings to .npy file.
        
        Args:
            embeddings: Dictionary of speaker embeddings.
            path: Output file path.
        """
        np.save(path, embeddings)
        logger.info(f"Saved embeddings to: {path}")
    
    def load_embeddings(self, path: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from .npy file.
        
        Args:
            path: Path to embeddings file.
            
        Returns:
            Dictionary of speaker embeddings.
        """
        embeddings = np.load(path, allow_pickle=True).item()
        logger.info(f"Loaded embeddings from: {path}")
        return embeddings


class SpeakerAssigner:
    """
    Assign separated sources to speakers using embedding similarity.
    
    This class uses cosine similarity between enrollment embeddings and
    source embeddings to match SepFormer outputs to known speakers.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize the SpeakerAssigner.
        
        Args:
            threshold: Minimum cosine similarity for confident assignment.
        """
        self.threshold = threshold
        logger.info(f"Initialized SpeakerAssigner (threshold={threshold})")
    
    def cosine_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.
            
        Returns:
            Cosine similarity score (0-1).
        """
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        
        # Clip to [0, 1] range (embeddings should be normalized, but be safe)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def assign_sources(
        self,
        separated_sources: List[np.ndarray],
        source_embeddings: List[np.ndarray],
        enrollment_embeddings: Dict[str, np.ndarray]
    ) -> List[dict]:
        """
        Assign separated sources to speakers.
        
        Args:
            separated_sources: List of separated audio sources.
            source_embeddings: List of embeddings for each source.
            enrollment_embeddings: Dictionary of enrollment embeddings per speaker.
            
        Returns:
            List of assignment dicts with 'source_idx', 'speaker', 'similarity', 'uncertain'.
        """
        assignments = []
        
        for src_idx, src_emb in enumerate(source_embeddings):
            best_speaker = None
            best_similarity = -1.0
            
            # Compare with each enrollment embedding
            for speaker, enroll_emb in enrollment_embeddings.items():
                similarity = self.cosine_similarity(src_emb, enroll_emb)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = speaker
            
            # Determine if assignment is uncertain
            is_uncertain = best_similarity < self.threshold
            
            assignment = {
                'source_idx': src_idx,
                'speaker': best_speaker,
                'similarity': best_similarity,
                'uncertain': is_uncertain
            }
            
            assignments.append(assignment)
            
            status = "UNCERTAIN" if is_uncertain else "confident"
            logger.info(
                f"Source {src_idx} -> {best_speaker} "
                f"(similarity={best_similarity:.3f}, {status})"
            )
        
        return assignments
