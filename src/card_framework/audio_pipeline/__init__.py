"""Audio-to-script stage package."""

from card_framework.audio_pipeline.config import should_use_audio_stage
from card_framework.audio_pipeline.factory import (
    build_audio_to_script_orchestrator,
    build_interjector_orchestrator,
    build_speaker_diarizer,
    build_speaker_sample_generator,
    build_voice_clone_orchestrator,
)
from card_framework.audio_pipeline.interjector import InterjectorOrchestrator
from card_framework.audio_pipeline.orchestrator import AudioStageOutput, AudioToScriptOrchestrator
from card_framework.audio_pipeline.voice_clone_orchestrator import VoiceCloneOrchestrator

__all__ = [
    "AudioStageOutput",
    "AudioToScriptOrchestrator",
    "InterjectorOrchestrator",
    "VoiceCloneOrchestrator",
    "build_audio_to_script_orchestrator",
    "build_interjector_orchestrator",
    "build_speaker_diarizer",
    "build_speaker_sample_generator",
    "build_voice_clone_orchestrator",
    "should_use_audio_stage",
]

