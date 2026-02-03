import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import whisper
import ssl
import urllib.request

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# Use new SpeechBrain import (v1.0+)
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier


class SpeakerDiarization:
    """
    Unsupervised speaker diarization pipeline using Whisper for segmentation
    and SpeechBrain ECAPA-TDNN for speaker embeddings.
    Optimized for Apple Silicon (M4 MacBook Pro).
    """
    
    def __init__(
        self,
        whisper_model: str = "large",
        similarity_threshold: float = 0.65,
        ema_alpha: float = 0.3,
        device: Optional[str] = None,
        min_speakers: int = 2,
        max_speakers: int = 10
    ):
        """
        Initialize the diarization pipeline.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            similarity_threshold: Cosine similarity threshold for speaker matching
            ema_alpha: Exponential moving average alpha for speaker profiles
            device: Device to run models on (mps/cpu/cuda)
            min_speakers: Expected minimum number of speakers
            max_speakers: Expected maximum number of speakers
        """
        # Use CPU due to MPS FFT limitations
        if device is None:
            self.device = "cpu"
            print("[INFO] Using CPU (optimized for Apple Silicon)")
        else:
            self.device = device
            
        self.similarity_threshold = similarity_threshold
        self.ema_alpha = ema_alpha
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Load Whisper for temporal segmentation
        print(f"[INFO] Loading Whisper model: {whisper_model}")
        try:
            self.whisper = whisper.load_model(whisper_model, device="cpu")
        except Exception as e:
            print(f"[WARNING] Failed to download Whisper model: {e}")
            print("[INFO] Attempting to load from cache...")
            self.whisper = whisper.load_model(
                whisper_model, 
                device="cpu", 
                download_root=str(Path.home() / ".cache" / "whisper")
            )
        
        # Load SpeechBrain ECAPA-TDNN for speaker embeddings
        print("[INFO] Loading SpeechBrain ECAPA-TDNN model")
        model_dir = Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"
        
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(model_dir),
            run_opts={"device": self.device}
        )
        
        # Speaker profile storage
        self.speaker_profiles = []
        self.speaker_counts = []
        self.all_embeddings = []
        
    def extract_segments(self, audio_path: str) -> List[Dict]:
        """
        Stage 1: Temporal segmentation using Whisper ASR.
        """
        print("[Stage 1] Extracting speech segments with Whisper...")
        
        result = self.whisper.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
            fp16=False
        )
        
        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'audio_path': audio_path
            })
        
        print(f"[Stage 1] Extracted {len(segments)} segments")
        return segments
    
    def extract_embedding(self, audio_path: str, start: float, end: float) -> np.ndarray:
        """
        Extract speaker embedding for an audio segment using ECAPA-TDNN.
        """
        import torchaudio
        
        # Load audio segment
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract segment
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment = waveform[:, start_sample:end_sample]
        
        # Ensure minimum segment length
        min_samples = int(0.5 * sample_rate)
        if segment.shape[1] < min_samples:
            padding = min_samples - segment.shape[1]
            segment = torch.nn.functional.pad(segment, (0, padding))
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            segment = resampler(segment.cpu())
        
        # Extract embedding
        with torch.no_grad():
            segment_input = segment.cpu()
            embedding = self.speaker_model.encode_batch(segment_input)
            embedding = embedding.squeeze().cpu().numpy()
        
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        """
        return np.dot(emb1, emb2)
    
    def assign_speaker(self, embedding: np.ndarray) -> int:
        """
        Stage 3: Assign speaker to segment using cosine similarity matching.
        """
        if len(self.speaker_profiles) == 0:
            self.speaker_profiles.append(embedding)
            self.speaker_counts.append(1)
            return 0
        
        # Compute similarities with all existing speakers
        similarities = [
            self.cosine_similarity(embedding, profile)
            for profile in self.speaker_profiles
        ]
        
        max_similarity = max(similarities)
        best_speaker_id = np.argmax(similarities)
        
        # Dynamic threshold: lower threshold if we have too many speakers
        dynamic_threshold = self.similarity_threshold
        if len(self.speaker_profiles) > self.max_speakers:
            dynamic_threshold = max(0.60, self.similarity_threshold - 0.05)
        
        if max_similarity >= dynamic_threshold:
            # Match found - assign to existing speaker
            speaker_id = best_speaker_id
            
            # Update speaker profile with EMA
            self.speaker_profiles[speaker_id] = (
                self.ema_alpha * embedding +
                (1 - self.ema_alpha) * self.speaker_profiles[speaker_id]
            )
            # Re-normalize
            self.speaker_profiles[speaker_id] = (
                self.speaker_profiles[speaker_id] / 
                (np.linalg.norm(self.speaker_profiles[speaker_id]) + 1e-8)
            )
            self.speaker_counts[speaker_id] += 1
            
            return speaker_id
        else:
            # Check if we should create a new speaker
            if len(self.speaker_profiles) < self.max_speakers:
                # Create new speaker
                speaker_id = len(self.speaker_profiles)
                self.speaker_profiles.append(embedding)
                self.speaker_counts.append(1)
                return speaker_id
            else:
                # Too many speakers, assign to closest one anyway
                return best_speaker_id
    
    def post_process_clustering(self, results: List[Dict]) -> List[Dict]:
        """
        Post-process to merge similar speakers using hierarchical clustering.
        """
        print("[Post-processing] Refining speaker clusters...")
        
        if len(self.speaker_profiles) <= self.min_speakers:
            return results
        
        # Compute pairwise similarities between speaker profiles
        n_speakers = len(self.speaker_profiles)
        similarity_matrix = np.zeros((n_speakers, n_speakers))
        
        for i in range(n_speakers):
            for j in range(i+1, n_speakers):
                sim = self.cosine_similarity(
                    self.speaker_profiles[i],
                    self.speaker_profiles[j]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Merge speakers with high similarity
        merge_threshold = 0.85
        speaker_mapping = list(range(n_speakers))
        
        for i in range(n_speakers):
            for j in range(i+1, n_speakers):
                if similarity_matrix[i, j] > merge_threshold:
                    old_id = speaker_mapping[j]
                    new_id = speaker_mapping[i]
                    for k in range(n_speakers):
                        if speaker_mapping[k] == old_id:
                            speaker_mapping[k] = new_id
        
        # Renumber speakers sequentially
        unique_speakers = sorted(set(speaker_mapping))
        final_mapping = {old: new for new, old in enumerate(unique_speakers)}
        
        # Update results
        for result in results:
            if result['speaker'] != "SPEAKER_UNKNOWN":
                old_id = int(result['speaker'].split('_')[1])
                new_id = final_mapping[speaker_mapping[old_id]]
                result['speaker'] = f"SPEAKER_{new_id:02d}"
        
        # Count speakers in final results
        final_speaker_counts = {}
        for result in results:
            speaker = result['speaker']
            final_speaker_counts[speaker] = final_speaker_counts.get(speaker, 0) + 1
        
        print(f"[Post-processing] Reduced from {n_speakers} to {len(final_speaker_counts)} speakers")
        
        return results
    
    def diarize(self, audio_path: str, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Complete diarization pipeline.
        """
        import time
        start_time = time.time()
        
        self.speaker_profiles = []
        self.speaker_counts = []
        self.all_embeddings = []
        
        # Stage 1: Temporal segmentation
        segments = self.extract_segments(audio_path)
        
        # Stage 2 & 3: Extract embeddings and assign speakers
        print("[Stage 2 & 3] Extracting embeddings and assigning speakers...")
        
        results = []
        for i, segment in enumerate(segments):
            print(f"[Progress] Processing segment {i+1}/{len(segments)}", end='\r')
            
            try:
                # Stage 2: Extract embedding
                embedding = self.extract_embedding(
                    segment['audio_path'],
                    segment['start'],
                    segment['end']
                )
                self.all_embeddings.append(embedding)
                
                # Stage 3: Assign speaker
                speaker_id = self.assign_speaker(embedding)
                
                results.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker': f"SPEAKER_{speaker_id:02d}",
                    'text': segment['text'],
                    'embedding_idx': i
                })
            except Exception as e:
                print(f"\n[WARNING] Failed to process segment {i+1}: {e}")
                results.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker': "SPEAKER_UNKNOWN",
                    'text': segment['text'],
                    'embedding_idx': -1
                })
        
        print(f"\n[Complete] Initial clustering: {len(self.speaker_profiles)} speakers detected")
        
        # Post-process to merge similar speakers
        results = self.post_process_clustering(results)
        
        elapsed_time = time.time() - start_time
        audio_duration = results[-1]['end'] if results else 0
        rtf = elapsed_time / audio_duration if audio_duration > 0 else 0
        
        # Print final speaker statistics
        final_speaker_counts = {}
        for result in results:
            speaker = result['speaker']
            final_speaker_counts[speaker] = final_speaker_counts.get(speaker, 0) + 1
        
        print("\n[Statistics] Final speaker distribution:")
        for speaker in sorted(final_speaker_counts.keys()):
            count = final_speaker_counts[speaker]
            print(f"  {speaker}: {count} segments")
        
        print(f"\n[Performance]")
        print(f"  Processing time: {elapsed_time:.1f}s")
        print(f"  Audio duration: {audio_duration:.1f}s")
        print(f"  Real-time factor: {rtf:.2f}x")
        
        # Save results if output directory provided
        if output_dir:
            self.save_results(results, audio_path, output_dir)
        
        return results
    
    def format_output(self, results: List[Dict]) -> str:
        """Format diarization results as readable text."""
        output = []
        for segment in results:
            start_time = f"{int(segment['start']//60):02d}:{segment['start']%60:05.2f}"
            end_time = f"{int(segment['end']//60):02d}:{segment['end']%60:05.2f}"
            output.append(
                f"[{start_time} - {end_time}] {segment['speaker']}: {segment['text']}"
            )
        return "\n".join(output)
    
    def save_results(self, results: List[Dict], audio_path: str, output_dir: str):
        """Save diarization results to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_name = Path(audio_path).stem
        
        # Remove embedding_idx from results before saving
        clean_results = []
        for result in results:
            clean_result = {k: v for k, v in result.items() if k != 'embedding_idx'}
            clean_results.append(clean_result)
        
        # Save formatted text
        txt_file = output_path / f"{audio_name}_diarization.txt"
        with open(txt_file, 'w') as f:
            f.write(self.format_output(clean_results))
        print(f"[INFO] Saved text output to: {txt_file}")
        
        # Save JSON format
        import json
        json_file = output_path / f"{audio_name}_diarization.json"
        with open(json_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"[INFO] Saved JSON output to: {json_file}")
        
        # Save RTTM format
        rttm_file = output_path / f"{audio_name}_diarization.rttm"
        with open(rttm_file, 'w') as f:
            for segment in clean_results:
                duration = segment['end'] - segment['start']
                speaker_id = segment['speaker'].replace('SPEAKER_', '')
                f.write(f"SPEAKER {audio_name} 1 {segment['start']:.3f} {duration:.3f} "
                       f"<NA> <NA> {speaker_id} <NA> <NA>\n")
        print(f"[INFO] Saved RTTM output to: {rttm_file}")


def diarize_audio(
    audio_path: str,
    output_dir: str,
    whisper_model: str = "medium",
    similarity_threshold: float = 0.65,
    ema_alpha: float = 0.3,
    min_speakers: int = 2,
    max_speakers: int = 10
) -> List[Dict]:
    """Convenience function to diarize audio file."""
    diarizer = SpeakerDiarization(
        whisper_model=whisper_model,
        similarity_threshold=similarity_threshold,
        ema_alpha=ema_alpha,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    
    return diarizer.diarize(audio_path, output_dir=output_dir)


__all__ = ['SpeakerDiarization', 'diarize_audio']