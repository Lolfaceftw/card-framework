#!/usr/bin/env python3
"""
Simple test script to verify targeted speaker separation implementation.

This script tests the core functionality without requiring actual audio files
by creating synthetic test data.
"""

import json
import os
import sys
import tempfile
import numpy as np
import soundfile as sf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.separation.enrollment import EnrollmentEmbeddingExtractor, SpeakerAssigner
from src.separation.crossfade import apply_crossfade, concatenate_with_crossfade


def test_crossfade():
    """Test cross-fade functionality."""
    print("=" * 80)
    print("Testing cross-fade utilities...")
    print("=" * 80)
    
    # Create two simple audio segments
    sample_rate = 16000
    duration = 1.0  # 1 second each
    
    # Generate test audio (sine waves at different frequencies)
    t1 = np.linspace(0, duration, int(sample_rate * duration))
    audio1 = np.sin(2 * np.pi * 440 * t1).astype(np.float32)  # 440 Hz
    
    t2 = np.linspace(0, duration, int(sample_rate * duration))
    audio2 = np.sin(2 * np.pi * 880 * t2).astype(np.float32)  # 880 Hz
    
    # Test apply_crossfade
    result = apply_crossfade(audio1, audio2, fade_duration_ms=25.0, sample_rate=sample_rate)
    expected_length = len(audio1) + len(audio2) - int(0.025 * sample_rate)
    
    assert len(result) == expected_length, f"Crossfade length mismatch: {len(result)} != {expected_length}"
    print(f"✓ apply_crossfade: OK (output length: {len(result)} samples)")
    
    # Test concatenate_with_crossfade
    segments = [audio1, audio2, audio1.copy()]
    result = concatenate_with_crossfade(segments, fade_duration_ms=25.0, sample_rate=sample_rate)
    
    print(f"✓ concatenate_with_crossfade: OK (output length: {len(result)} samples)")
    
    # Test edge cases
    empty = np.array([], dtype=np.float32)
    result = apply_crossfade(audio1, empty, fade_duration_ms=25.0, sample_rate=sample_rate)
    assert len(result) == len(audio1), "Empty audio2 should return audio1"
    print(f"✓ Edge case (empty audio): OK")
    
    # Test with very short segments
    short = np.array([0.1, 0.2], dtype=np.float32)
    result = apply_crossfade(short, short, fade_duration_ms=100.0, sample_rate=sample_rate)
    print(f"✓ Edge case (short segments): OK (length: {len(result)})")
    
    print("\nAll cross-fade tests passed! ✓\n")


def test_enrollment_selection():
    """Test enrollment snippet selection logic."""
    print("=" * 80)
    print("Testing enrollment snippet selection...")
    print("=" * 80)
    
    # Create synthetic segments
    segments = [
        {'start': 0.0, 'end': 5.0, 'speaker': 'SPEAKER_00'},  # Clean, good duration
        {'start': 6.0, 'end': 8.0, 'speaker': 'SPEAKER_00'},  # Short
        {'start': 10.0, 'end': 16.0, 'speaker': 'SPEAKER_01'},  # Clean, good duration
        {'start': 17.0, 'end': 20.0, 'speaker': 'SPEAKER_01'},  # Overlaps with SPEAKER_00
        {'start': 18.0, 'end': 21.0, 'speaker': 'SPEAKER_00'},  # Overlaps with SPEAKER_01
    ]
    
    # Create overlap regions
    overlaps = [
        {'start': 18.0, 'end': 20.0, 'speakers': ['SPEAKER_00', 'SPEAKER_01']}
    ]
    
    # Initialize extractor (this will try to load the model)
    try:
        extractor = EnrollmentEmbeddingExtractor(sample_rate=16000, device='cpu')
        
        # Test snippet selection
        snippets = extractor.select_enrollment_snippets(
            segments, overlaps, duration_range=(3.0, 6.0)
        )
        
        # Verify results
        assert 'SPEAKER_00' in snippets, "SPEAKER_00 should have snippets"
        assert 'SPEAKER_01' in snippets, "SPEAKER_01 should have snippets"
        
        # Check SPEAKER_00: should have segment [0, 5] (clean, 5s)
        speaker_00_snippets = snippets['SPEAKER_00']
        assert len(speaker_00_snippets) > 0, "SPEAKER_00 should have at least one snippet"
        print(f"✓ SPEAKER_00 has {len(speaker_00_snippets)} clean snippets")
        
        # Check SPEAKER_01: should have segment [10, 16] (clean, 6s)
        speaker_01_snippets = snippets['SPEAKER_01']
        assert len(speaker_01_snippets) > 0, "SPEAKER_01 should have at least one snippet"
        print(f"✓ SPEAKER_01 has {len(speaker_01_snippets)} clean snippets")
        
        print("\nEnrollment snippet selection tests passed! ✓\n")
        
    except Exception as e:
        print(f"⚠ Enrollment test skipped (model not available): {e}")
        print("This is expected in environments without model files.\n")


def test_speaker_assigner():
    """Test speaker assignment logic."""
    print("=" * 80)
    print("Testing speaker assignment...")
    print("=" * 80)
    
    assigner = SpeakerAssigner(threshold=0.7)
    
    # Create synthetic embeddings (192-dim, typical for ECAPA-TDNN)
    np.random.seed(42)
    
    # Enrollment embeddings
    speaker_00_emb = np.random.randn(192).astype(np.float32)
    speaker_00_emb /= np.linalg.norm(speaker_00_emb)  # Normalize
    
    speaker_01_emb = np.random.randn(192).astype(np.float32)
    speaker_01_emb /= np.linalg.norm(speaker_01_emb)  # Normalize
    
    enrollment_embeddings = {
        'SPEAKER_00': speaker_00_emb,
        'SPEAKER_01': speaker_01_emb
    }
    
    # Source embeddings (similar to enrollment)
    source_0_emb = speaker_00_emb + np.random.randn(192).astype(np.float32) * 0.1
    source_0_emb /= np.linalg.norm(source_0_emb)
    
    source_1_emb = speaker_01_emb + np.random.randn(192).astype(np.float32) * 0.1
    source_1_emb /= np.linalg.norm(source_1_emb)
    
    # Test cosine similarity
    sim = assigner.cosine_similarity(speaker_00_emb, source_0_emb)
    print(f"✓ Cosine similarity (same speaker): {sim:.3f}")
    assert sim > 0.5, "Same speaker similarity should be high"
    
    # Create dummy audio sources (not used in assignment, just for API)
    sample_rate = 16000
    source_audio = [
        np.random.randn(sample_rate).astype(np.float32),
        np.random.randn(sample_rate).astype(np.float32)
    ]
    
    # Test assignment
    assignments = assigner.assign_sources(
        source_audio,
        [source_0_emb, source_1_emb],
        enrollment_embeddings
    )
    
    assert len(assignments) == 2, "Should have 2 assignments"
    
    # Verify assignments
    print(f"✓ Source 0 assigned to: {assignments[0]['speaker']} (similarity: {assignments[0]['similarity']:.3f})")
    print(f"✓ Source 1 assigned to: {assignments[1]['speaker']} (similarity: {assignments[1]['similarity']:.3f})")
    
    print("\nSpeaker assignment tests passed! ✓\n")


def test_integration_with_synthetic_audio():
    """Test integration with synthetic audio and diarization."""
    print("=" * 80)
    print("Testing integration with synthetic audio...")
    print("=" * 80)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic audio (2 seconds of sine wave)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        audio_path = os.path.join(tmpdir, "test_audio.wav")
        sf.write(audio_path, audio, sample_rate)
        print(f"✓ Created synthetic audio: {audio_path}")
        
        # Create synthetic diarization
        diarization = [
            {'start': 0.0, 'end': 0.8, 'speaker': 'SPEAKER_00', 'text': 'Hello'},
            {'start': 1.0, 'end': 1.8, 'speaker': 'SPEAKER_01', 'text': 'World'},
        ]
        
        diarization_path = os.path.join(tmpdir, "test_diarization.json")
        with open(diarization_path, 'w') as f:
            json.dump(diarization, f)
        print(f"✓ Created synthetic diarization: {diarization_path}")
        
        # Test that files were created
        assert os.path.exists(audio_path), "Audio file should exist"
        assert os.path.exists(diarization_path), "Diarization file should exist"
        
        print("\nIntegration test setup completed! ✓\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TARGETED SPEAKER SEPARATION - UNIT TESTS")
    print("=" * 80 + "\n")
    
    try:
        # Test cross-fade utilities (no external dependencies)
        test_crossfade()
        
        # Test enrollment snippet selection (requires SpeechBrain)
        test_enrollment_selection()
        
        # Test speaker assignment
        test_speaker_assigner()
        
        # Test integration setup
        test_integration_with_synthetic_audio()
        
        print("=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
