#!/usr/bin/env python3
"""
Main script to run speaker diarization with Whisper medium model.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.diarization.diarization import SpeakerDiarization


def main():
    # Project paths
    project_root = Path(__file__).parent.parent.parent
    audio_file = project_root / "data" / "custom" / "PrimeagenLex.wav"
    output_dir = project_root / "outputs"
    
    print("="*80)
    print("Unsupervised Speaker Diarization Pipeline")
    print("Using Whisper Large Model")
    print("="*80)
    print(f"Audio file: {audio_file}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Check if audio file exists
    if not audio_file.exists():
        print(f"[ERROR] Audio file not found: {audio_file}")
        return
    
    # Initialize diarizer with medium Whisper model
    diarizer = SpeakerDiarization(
        whisper_model="large",          # Using large
        similarity_threshold=0.45,      # Lower = fewer unique speakers
        ema_alpha=0.3,                  # Lower = more stable profiles
        min_speakers=2,                 # Expected minimum speakers
        max_speakers=2                  # Expected maximum speakers
    )
    
    # Run diarization
    results = diarizer.diarize(str(audio_file), output_dir=str(output_dir))
    
    # Print summary
    print("\n" + "="*80)
    print("DIARIZATION SUMMARY")
    print("="*80)
    
    speakers = set(r['speaker'] for r in results if r['speaker'] != "SPEAKER_UNKNOWN")
    print(f"Total segments: {len(results)}")
    print(f"Unique speakers: {len(speakers)}")
    if results:
        print(f"Duration: {results[-1]['end']:.2f} seconds ({results[-1]['end']/60:.1f} minutes)")
    
    print("\n" + "="*80)
    print("PREVIEW (First 10 segments)")
    print("="*80)
    
    preview = diarizer.format_output(results[:10])
    print(preview)
    
    if len(results) > 10:
        print(f"\n... ({len(results) - 10} more segments)")
    
    print("\n" + "="*80)
    print(f"Full results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()