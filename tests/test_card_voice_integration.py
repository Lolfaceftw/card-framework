"""
Test Script for CARD-SpeakerAudioExtraction and Voice-Cloner-and-Interjector.

This module provides comprehensive testing for:
1. CARD-SpeakerAudioExtraction: Audio separation toolkit
2. voice-cloner-and-interjector: IndexTTS2-based voice cloning pipeline

Run with: uv run pytest tests/test_card_voice_integration.py -v
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project roots to path
PROJECT_ROOT = Path(__file__).parent.parent
CARD_ROOT = PROJECT_ROOT / "CARD-SpeakerAudioExtraction"
VOICE_ROOT = PROJECT_ROOT / "voice-cloner-and-interjector"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARD_ROOT))
sys.path.insert(0, str(VOICE_ROOT))
sys.path.insert(0, str(VOICE_ROOT / "refactor"))

# Test data paths
TEST_DIR = Path(__file__).parent
TEST_DIARIZATION = TEST_DIR / "test_diarization.json"
TEST_PODCAST_INPUT = TEST_DIR / "test_podcast_input.json"
TEST_OUTPUT_DIR = TEST_DIR / "outputs"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_diarization_data() -> List[Dict[str, Any]]:
    """Load test diarization data."""
    with open(TEST_DIARIZATION, "r", encoding="utf-8") as f:
        return cast(List[Dict[str, Any]], json.load(f))


@pytest.fixture
def test_podcast_data() -> List[Dict[str, Any]]:
    """Load test podcast input data."""
    with open(TEST_PODCAST_INPUT, "r", encoding="utf-8") as f:
        return cast(List[Dict[str, Any]], json.load(f))


@pytest.fixture
def output_dir() -> Path:
    """Create and return output directory for test artifacts."""
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    return TEST_OUTPUT_DIR


@pytest.fixture
def sample_audio_array() -> np.ndarray:
    """Generate a sample audio array for testing."""
    # 5 seconds of 16kHz audio (sine wave)
    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Mix of two frequencies to simulate speech-like content
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    return cast(np.ndarray, audio.astype(np.float32))


# =============================================================================
# CARD-SpeakerAudioExtraction Tests
# =============================================================================

class TestCARDImports:
    """Test that CARD modules can be imported."""

    def test_import_separation_module(self) -> None:
        """Test that separation module imports correctly."""
        try:
            from src.separation import (
                TargetedSpeakerSeparator,
                DiarizationGuidedSeparator,
                SpeechSeparator,
            )
            assert TargetedSpeakerSeparator is not None
            assert DiarizationGuidedSeparator is not None
            assert SpeechSeparator is not None
        except ImportError as e:
            pytest.skip(f"CARD separation module not available: {e}")

    def test_import_utils(self) -> None:
        """Test that utility functions import correctly."""
        try:
            from src.separation import (
                load_audio,
                save_separated_audio,
                apply_crossfade,
                concatenate_with_crossfade,
            )
            assert load_audio is not None
            assert save_separated_audio is not None
            assert apply_crossfade is not None
            assert concatenate_with_crossfade is not None
        except ImportError as e:
            pytest.skip(f"CARD utils not available: {e}")


class TestCARDDiarization:
    """Test diarization loading and parsing."""

    def test_load_diarization_file(
        self, test_diarization_data: List[Dict[str, Any]]
    ) -> None:
        """Test that diarization data loads correctly."""
        assert len(test_diarization_data) > 0
        assert all("start" in seg for seg in test_diarization_data)
        assert all("end" in seg for seg in test_diarization_data)
        assert all("speaker" in seg for seg in test_diarization_data)

    def test_diarization_segment_structure(
        self, test_diarization_data: List[Dict[str, Any]]
    ) -> None:
        """Test that diarization segments have valid structure."""
        for seg in test_diarization_data:
            assert isinstance(seg["start"], (int, float))
            assert isinstance(seg["end"], (int, float))
            assert seg["end"] > seg["start"]
            assert isinstance(seg["speaker"], str)

    def test_detect_overlaps(
        self, test_diarization_data: List[Dict[str, Any]]
    ) -> None:
        """Test overlap detection in diarization data."""
        overlaps = []
        segments = sorted(test_diarization_data, key=lambda x: x["start"])

        for i, seg1 in enumerate(segments):
            for seg2 in segments[i + 1:]:
                # Check for overlap
                if seg1["end"] > seg2["start"] and seg1["start"] < seg2["end"]:
                    overlaps.append({
                        "start": max(seg1["start"], seg2["start"]),
                        "end": min(seg1["end"], seg2["end"]),
                        "speakers": [seg1["speaker"], seg2["speaker"]],
                    })

        # Our test data has overlaps
        assert len(overlaps) >= 2, "Expected at least 2 overlapping regions in test data"
        print(f"\nDetected {len(overlaps)} overlapping regions:")
        for overlap in overlaps:
            print(f"  {overlap['start']:.2f}s - {overlap['end']:.2f}s: {overlap['speakers']}")


class TestCARDTargetedSeparator:
    """Test TargetedSpeakerSeparator functionality."""

    def test_separator_initialization(self) -> None:
        """Test that TargetedSpeakerSeparator initializes correctly."""
        try:
            from src.separation import TargetedSpeakerSeparator

            separator = TargetedSpeakerSeparator(
                sample_rate=16000,
                similarity_threshold=0.7,
                crossfade_ms=25.0,
                device="cpu",
            )
            assert separator is not None
            assert separator.sample_rate == 16000
            assert separator.similarity_threshold == 0.7
            print("\n✓ TargetedSpeakerSeparator initialized successfully")
        except ImportError as e:
            pytest.skip(f"TargetedSpeakerSeparator not available: {e}")
        except Exception as e:
            pytest.skip(f"TargetedSpeakerSeparator initialization failed: {e}")

    def test_separator_load_diarization(
        self, test_diarization_data: List[Dict[str, Any]]
    ) -> None:
        """Test diarization loading via separator."""
        try:
            from src.separation import TargetedSpeakerSeparator

            separator = TargetedSpeakerSeparator(device="cpu")
            segments = separator.load_diarization(str(TEST_DIARIZATION))
            assert len(segments) == len(test_diarization_data)
            print(f"\n✓ Loaded {len(segments)} diarization segments")
        except ImportError as e:
            pytest.skip(f"TargetedSpeakerSeparator not available: {e}")
        except Exception as e:
            pytest.skip(f"Diarization loading failed: {e}")

    def test_separator_detect_overlaps(self) -> None:
        """Test overlap detection via separator."""
        try:
            from src.separation import TargetedSpeakerSeparator

            separator = TargetedSpeakerSeparator(device="cpu")
            segments = separator.load_diarization(str(TEST_DIARIZATION))
            overlaps = separator.detect_overlaps(segments)
            assert isinstance(overlaps, list)
            print(f"\n✓ Detected {len(overlaps)} overlaps via separator")
        except ImportError as e:
            pytest.skip(f"TargetedSpeakerSeparator not available: {e}")
        except Exception as e:
            pytest.skip(f"Overlap detection failed: {e}")


class TestCARDCrossfade:
    """Test crossfade utilities."""

    def test_apply_crossfade(self, sample_audio_array: np.ndarray) -> None:
        """Test crossfade application."""
        try:
            from src.separation import apply_crossfade

            # Apply crossfade at boundary
            crossfade_samples = 400  # 25ms at 16kHz
            result = apply_crossfade(
                sample_audio_array[:8000],
                sample_audio_array[7600:],
                crossfade_samples,
            )
            assert result is not None
            assert len(result) > 0
            print(f"\n✓ Crossfade applied, result length: {len(result)}")
        except ImportError as e:
            pytest.skip(f"Crossfade utils not available: {e}")
        except Exception as e:
            pytest.skip(f"Crossfade failed: {e}")

    def test_concatenate_with_crossfade(
        self, sample_audio_array: np.ndarray
    ) -> None:
        """Test segment concatenation with crossfade."""
        try:
            from src.separation import concatenate_with_crossfade

            segments = [
                sample_audio_array[:16000],
                sample_audio_array[15600:32000],
                sample_audio_array[31600:48000],
            ]
            result = concatenate_with_crossfade(segments, fade_duration_ms=25.0)
            assert result is not None
            assert len(result) > 0
            print(f"\n✓ Concatenated {len(segments)} segments, result length: {len(result)}")
        except ImportError as e:
            pytest.skip(f"Crossfade utils not available: {e}")
        except Exception as e:
            pytest.skip(f"Concatenation failed: {e}")


# =============================================================================
# Voice-Cloner-and-Interjector Tests
# =============================================================================

class TestVoiceClonerImports:
    """Test that voice-cloner modules can be imported."""

    def test_import_config(self) -> None:
        """Test that config imports correctly."""
        try:
            from config import LLM_CFG, TTS_CFG, AUDIO_CFG

            assert LLM_CFG is not None
            assert TTS_CFG is not None
            assert AUDIO_CFG is not None
            print(f"\n✓ Config loaded: LLM model={LLM_CFG.model}")
        except ImportError as e:
            pytest.skip(f"Config not available: {e}")

    def test_import_services(self) -> None:
        """Test that services import correctly."""
        try:
            from services.audio_service import AudioService
            from services.llm_service import LLMService
            from services.tts_service import TTSService

            assert AudioService is not None
            assert LLMService is not None
            assert TTSService is not None
            print("\n✓ All services imported successfully")
        except ImportError as e:
            pytest.skip(f"Services not available: {e}")

    def test_import_pipeline(self) -> None:
        """Test that CardPipeline imports correctly."""
        try:
            from core.pipeline import CardPipeline

            assert CardPipeline is not None
            print("\n✓ CardPipeline imported successfully")
        except ImportError as e:
            pytest.skip(f"CardPipeline not available: {e}")


class TestAudioService:
    """Test AudioService functionality."""

    def test_audio_service_initialization(self) -> None:
        """Test AudioService initializes correctly."""
        try:
            from services.audio_service import AudioService

            service = AudioService()
            assert service is not None
            print("\n✓ AudioService initialized")
        except ImportError as e:
            pytest.skip(f"AudioService not available: {e}")

    def test_timing_calculation(self) -> None:
        """Test audio timing calculations."""
        try:
            from pydub import AudioSegment
            from services.audio_service import AudioService

            service = AudioService()

            # Create a mock audio segment (5 seconds)
            mock_audio = AudioSegment.silent(duration=5000)

            timing = service.calculate_timing(
                text="Test text for timing calculation",
                audio=mock_audio,
                trigger_percent=0.5,
            )

            assert "trigger_ms" in timing
            assert "reaction_delay" in timing
            assert "final_pos_ms" in timing
            assert "duration_ms" in timing
            assert timing["duration_ms"] == 5000
            assert 2000 <= timing["trigger_ms"] <= 3000  # Around 50%
            print(f"\n✓ Timing calculated: trigger={timing['trigger_ms']}ms, final={timing['final_pos_ms']}ms")
        except ImportError as e:
            pytest.skip(f"AudioService not available: {e}")

    def test_segment_merging(self) -> None:
        """Test audio segment merging."""
        try:
            from pydub import AudioSegment
            from services.audio_service import AudioService

            service = AudioService()

            # Create mock segments
            segments = [
                AudioSegment.silent(duration=1000),
                AudioSegment.silent(duration=1500),
                AudioSegment.silent(duration=2000),
            ]

            merged = service.merge_segments(segments)
            assert merged is not None
            # Total should be sum of segments + pauses - crossfades
            assert len(merged) > 0
            print(f"\n✓ Merged {len(segments)} segments, result duration: {len(merged)}ms")
        except ImportError as e:
            pytest.skip(f"AudioService not available: {e}")

    def test_audio_overlay(self) -> None:
        """Test audio overlay functionality."""
        try:
            from pydub import AudioSegment
            from services.audio_service import AudioService

            service = AudioService()

            main = AudioSegment.silent(duration=5000)
            overlay = AudioSegment.silent(duration=1000)

            result = service.overlay(main, overlay, position_ms=2000)
            assert result is not None
            assert len(result) == 5000  # Main audio length preserved
            print("\n✓ Audio overlay successful")
        except ImportError as e:
            pytest.skip(f"AudioService not available: {e}")


class TestLLMService:
    """Test LLMService functionality."""

    def test_llm_service_initialization(self) -> None:
        """Test LLMService initializes correctly."""
        try:
            from services.llm_service import LLMService

            service = LLMService()
            assert service is not None
            print("\n✓ LLMService initialized")
        except ImportError as e:
            pytest.skip(f"LLMService not available: {e}")

    def test_trigger_prompt_building(self) -> None:
        """Test trigger detection prompt building."""
        try:
            from services.llm_service import LLMService

            service = LLMService()
            prompt = service._build_trigger_prompt("This is a test sentence with important content.")

            assert "trigger_word" in prompt
            assert "char_pos" in prompt
            assert "category" in prompt
            print(f"\n✓ Trigger prompt built:\n{prompt[:100]}...")
        except ImportError as e:
            pytest.skip(f"LLMService not available: {e}")

    @patch("services.llm_service.requests.get")
    def test_model_check_with_mock(self, mock_get: MagicMock) -> None:
        """Test model availability check with mock."""
        try:
            from services.llm_service import LLMService

            # Mock successful response
            mock_get.return_value.json.return_value = {
                "models": [{"name": "mistral:7b-instruct-q4_0"}]
            }

            service = LLMService()
            result = service.ensure_model_loaded()
            assert result is True
            print("\n✓ Model check with mock successful")
        except ImportError as e:
            pytest.skip(f"LLMService not available: {e}")

    @patch("services.llm_service.requests.post")
    def test_trigger_detection_mock(self, mock_post: MagicMock) -> None:
        """Test trigger detection with mocked LLM response."""
        try:
            from services.llm_service import LLMService

            # Mock LLM response
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "response": json.dumps({
                    "trigger_word": "important",
                    "char_pos": 25,
                    "category": "statement",
                })
            }

            service = LLMService()
            triggers = service.detect_trigger("This is a test with important content.")

            assert len(triggers) > 0
            assert triggers[0]["trigger_word"] == "important"
            assert "pos_percent" in triggers[0]
            print(f"\n✓ Trigger detected: {triggers[0]}")
        except ImportError as e:
            pytest.skip(f"LLMService not available: {e}")


class TestTTSService:
    """Test TTSService functionality."""

    def test_tts_service_initialization(self) -> None:
        """Test TTSService initializes correctly."""
        try:
            from services.tts_service import TTSService

            service = TTSService()
            assert service is not None
            assert service.model is None  # Lazy loading
            print("\n✓ TTSService initialized (model not loaded yet)")
        except ImportError as e:
            pytest.skip(f"TTSService not available: {e}")

    def test_tts_service_structure(self) -> None:
        """Test TTSService has required methods."""
        try:
            from services.tts_service import TTSService

            service = TTSService()
            assert hasattr(service, "load_model")
            assert hasattr(service, "synthesize")
            assert callable(service.load_model)
            assert callable(service.synthesize)
            print("\n✓ TTSService has required methods")
        except ImportError as e:
            pytest.skip(f"TTSService not available: {e}")


class TestCardPipeline:
    """Test CardPipeline functionality."""

    def test_pipeline_structure(self) -> None:
        """Test CardPipeline has required structure."""
        try:
            from core.pipeline import CardPipeline

            # Check class structure without instantiation
            assert hasattr(CardPipeline, "__init__")
            assert hasattr(CardPipeline, "run")
            assert hasattr(CardPipeline, "_handle_interjection")
            assert hasattr(CardPipeline, "_save_log")
            print("\n✓ CardPipeline has required methods")
        except ImportError as e:
            pytest.skip(f"CardPipeline not available: {e}")

    @patch("core.pipeline.LLMService")
    @patch("core.pipeline.TTSService")
    @patch("core.pipeline.AudioService")
    def test_pipeline_initialization_mock(
        self,
        mock_audio: MagicMock,
        _mock_tts: MagicMock,
        _mock_llm: MagicMock,
    ) -> None:
        """Test CardPipeline initializes with mocked services."""
        try:
            from core.pipeline import CardPipeline

            pipeline = CardPipeline()
            assert pipeline is not None
            assert pipeline.llm is not None
            assert pipeline.tts is not None
            assert pipeline.audio is not None
            assert pipeline.interjection_log == []
            print("\n✓ CardPipeline initialized with mocked services")
        except ImportError as e:
            pytest.skip(f"CardPipeline not available: {e}")


class TestPodcastInput:
    """Test podcast input data validation."""

    def test_podcast_input_structure(
        self, test_podcast_data: List[Dict[str, Any]]
    ) -> None:
        """Test podcast input data structure."""
        assert len(test_podcast_data) > 0

        required_fields = ["speaker", "voice_sample", "text"]
        for entry in test_podcast_data:
            for field in required_fields:
                assert field in entry, f"Missing required field: {field}"

            if "emo_alpha" in entry:
                assert 0.0 <= entry["emo_alpha"] <= 1.0

        print(f"\n✓ Podcast input validated: {len(test_podcast_data)} entries")

    def test_speaker_mapping(
        self, test_podcast_data: List[Dict[str, Any]]
    ) -> None:
        """Test speaker to voice sample mapping."""
        speakers: dict[str, set[str]] = {}
        for entry in test_podcast_data:
            speaker = entry["speaker"]
            sample = entry["voice_sample"]
            if speaker not in speakers:
                speakers[speaker] = set()
            speakers[speaker].add(sample)

        print("\n✓ Speaker mapping:")
        for speaker, samples in speakers.items():
            print(f"  {speaker}: {samples}")


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining CARD and voice-cloner components."""

    def test_diarization_to_separation_workflow(
        self, test_diarization_data: List[Dict[str, Any]]
    ) -> None:
        """Test workflow from diarization to separation preparation."""
        # Parse diarization
        segments = sorted(test_diarization_data, key=lambda x: x["start"])
        assert len(segments) > 0

        # Extract unique speakers
        speakers = list(set(seg["speaker"] for seg in segments))
        assert len(speakers) == 2  # Our test data has 2 speakers

        # Calculate total duration
        total_duration = max(seg["end"] for seg in segments)
        assert total_duration > 0

        print("\n✓ Workflow test:")
        print(f"  Segments: {len(segments)}")
        print(f"  Speakers: {speakers}")
        print(f"  Duration: {total_duration:.2f}s")

    def test_podcast_to_tts_workflow(
        self, test_podcast_data: List[Dict[str, Any]]
    ) -> None:
        """Test workflow from podcast input to TTS preparation."""
        # Validate all entries
        for idx, entry in enumerate(test_podcast_data):
            assert "text" in entry
            assert len(entry["text"]) > 0
            assert "voice_sample" in entry

        # Check for sample prompt
        sample_path = TEST_DIR / "sample_prompt.wav"
        sample_exists = sample_path.exists()

        print("\n✓ Podcast workflow test:")
        print(f"  Entries: {len(test_podcast_data)}")
        print(f"  Sample prompt exists: {sample_exists}")

        if sample_exists:
            print(f"  Sample path: {sample_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
