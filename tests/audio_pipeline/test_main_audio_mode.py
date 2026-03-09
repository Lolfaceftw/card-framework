from card_framework.audio_pipeline.config import should_use_audio_stage


def test_should_use_audio_stage_true_for_default_audio_first() -> None:
    assert should_use_audio_stage({}) is True


def test_should_use_audio_stage_true_for_audio_first() -> None:
    assert (
        should_use_audio_stage(
            {
                "input_mode": "audio_first",
                "audio_path": "",
            }
        )
        is True
    )


def test_should_use_audio_stage_auto_detect_requires_audio_path() -> None:
    assert (
        should_use_audio_stage(
            {
                "input_mode": "auto_detect",
                "audio_path": "",
            }
        )
        is False
    )
    assert (
        should_use_audio_stage(
            {
                "input_mode": "auto_detect",
                "audio_path": "sample.wav",
            }
        )
        is True
    )


def test_should_use_audio_stage_raises_for_invalid_mode() -> None:
    import pytest

    with pytest.raises(ValueError):
        should_use_audio_stage({"input_mode": "invalid"})

