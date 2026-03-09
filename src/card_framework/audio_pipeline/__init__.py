"""Audio-to-script stage package."""

from __future__ import annotations

from typing import Any

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


def __getattr__(name: str) -> Any:
    """Resolve audio-pipeline exports lazily to avoid eager heavy imports."""
    if name == "should_use_audio_stage":
        from card_framework.audio_pipeline.config import should_use_audio_stage

        return should_use_audio_stage
    if name in {
        "build_audio_to_script_orchestrator",
        "build_interjector_orchestrator",
        "build_speaker_diarizer",
        "build_speaker_sample_generator",
        "build_voice_clone_orchestrator",
    }:
        from card_framework.audio_pipeline.factory import (
            build_audio_to_script_orchestrator,
            build_interjector_orchestrator,
            build_speaker_diarizer,
            build_speaker_sample_generator,
            build_voice_clone_orchestrator,
        )

        exported = {
            "build_audio_to_script_orchestrator": build_audio_to_script_orchestrator,
            "build_interjector_orchestrator": build_interjector_orchestrator,
            "build_speaker_diarizer": build_speaker_diarizer,
            "build_speaker_sample_generator": build_speaker_sample_generator,
            "build_voice_clone_orchestrator": build_voice_clone_orchestrator,
        }
        return exported[name]
    if name in {"InterjectorOrchestrator", "VoiceCloneOrchestrator"}:
        from card_framework.audio_pipeline.interjector import InterjectorOrchestrator
        from card_framework.audio_pipeline.voice_clone_orchestrator import (
            VoiceCloneOrchestrator,
        )

        exported = {
            "InterjectorOrchestrator": InterjectorOrchestrator,
            "VoiceCloneOrchestrator": VoiceCloneOrchestrator,
        }
        return exported[name]
    if name in {"AudioStageOutput", "AudioToScriptOrchestrator"}:
        from card_framework.audio_pipeline.orchestrator import (
            AudioStageOutput,
            AudioToScriptOrchestrator,
        )

        exported = {
            "AudioStageOutput": AudioStageOutput,
            "AudioToScriptOrchestrator": AudioToScriptOrchestrator,
        }
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
