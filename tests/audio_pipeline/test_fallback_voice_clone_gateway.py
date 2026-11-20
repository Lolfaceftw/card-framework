from pathlib import Path

import pytest

from card_framework.audio_pipeline.errors import NonRetryableAudioStageError
from card_framework.audio_pipeline.gateways.fallback_voice_clone_gateway import (
    PassthroughVoiceCloneGateway,
)


def test_passthrough_voice_clone_gateway_copies_reference_audio(tmp_path: Path) -> None:
    gateway = PassthroughVoiceCloneGateway()
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"audio")
    output_audio = tmp_path / "output.wav"

    rendered_path = gateway.synthesize(
        reference_audio_path=reference_audio,
        text="hello world",
        output_audio_path=output_audio,
    )

    assert rendered_path == output_audio
    assert output_audio.read_bytes() == b"audio"


def test_passthrough_voice_clone_gateway_requires_reference_audio(tmp_path: Path) -> None:
    gateway = PassthroughVoiceCloneGateway()

    with pytest.raises(NonRetryableAudioStageError, match="does not exist"):
        gateway.synthesize(
            reference_audio_path=tmp_path / "missing.wav",
            text="hello world",
            output_audio_path=tmp_path / "output.wav",
        )

