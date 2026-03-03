"""Typed contracts for the audio-to-transcript pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

from audio_pipeline.eta import StageProgressCallback


class TranscriptSegmentPayload(TypedDict):
    """Serialized segment payload consumed by downstream retrieval/LLM stages."""

    speaker: str
    start_time: int
    end_time: int
    text: str


class TranscriptMetadataPayload(TypedDict, total=False):
    """Optional serialized metadata emitted by the audio stage."""

    source_audio_path: str
    vocals_audio_path: str
    separator_model: str
    transcriber_model: str
    diarizer_backend: str
    device: str
    generated_at_utc: str
    warnings: list[str]
    speaker_samples_manifest_path: str
    speaker_samples_dir: str
    speaker_sample_count: int
    speaker_samples_generated_at_utc: str


class TranscriptPayload(TypedDict, total=False):
    """Transcript document payload persisted as JSON."""

    segments: list[TranscriptSegmentPayload]
    metadata: TranscriptMetadataPayload


@dataclass(slots=True, frozen=True)
class TimedTextSegment:
    """Text span with millisecond timing from ASR output."""

    start_time_ms: int
    end_time_ms: int
    text: str

    def __post_init__(self) -> None:
        if self.start_time_ms < 0:
            raise ValueError("start_time_ms must be >= 0")
        if self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms must be >= start_time_ms")
        if not self.text.strip():
            raise ValueError("text must be non-empty")


@dataclass(slots=True, frozen=True)
class DiarizationTurn:
    """Speaker activity span with millisecond timing."""

    speaker: str
    start_time_ms: int
    end_time_ms: int

    def __post_init__(self) -> None:
        if self.start_time_ms < 0:
            raise ValueError("start_time_ms must be >= 0")
        if self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms must be >= start_time_ms")
        if not self.speaker.strip():
            raise ValueError("speaker must be non-empty")


@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    """Normalized transcript segment produced by alignment."""

    speaker: str
    start_time: int
    end_time: int
    text: str

    def __post_init__(self) -> None:
        if self.start_time < 0:
            raise ValueError("start_time must be >= 0")
        if self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")
        if not self.speaker.strip():
            raise ValueError("speaker must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")


@runtime_checkable
class SourceSeparator(Protocol):
    """Port for source-separation adapters."""

    def separate_vocals(
        self,
        input_audio_path: Path,
        output_dir: Path,
        *,
        device: str,
        progress_callback: StageProgressCallback | None = None,
    ) -> Path:
        """Return path to separated vocals audio."""


@runtime_checkable
class SpeechTranscriber(Protocol):
    """Port for speech-to-text adapters."""

    def transcribe(
        self,
        audio_path: Path,
        *,
        device: str,
        progress_callback: StageProgressCallback | None = None,
    ) -> list[TimedTextSegment]:
        """Return timed text segments."""


@runtime_checkable
class SpeakerDiarizer(Protocol):
    """Port for speaker-diarization adapters."""

    def diarize(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        device: str,
        progress_callback: StageProgressCallback | None = None,
    ) -> list[DiarizationTurn]:
        """Return diarization turns."""
