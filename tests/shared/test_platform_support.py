"""Tests for packaged-runtime platform capability helpers."""

from __future__ import annotations

from card_framework.shared import platform_support


def test_ensure_supported_packaged_runtime_accepts_cpu_on_macos_arm64() -> None:
    """Allow packaged CPU inference on Apple Silicon macOS."""
    runtime_platform = platform_support.RuntimePlatform(
        system="Darwin",
        machine="arm64",
    )

    resolved_platform = platform_support.ensure_supported_packaged_runtime(
        device="cpu",
        runtime_platform=runtime_platform,
    )

    assert resolved_platform is runtime_platform


def test_ensure_supported_packaged_runtime_rejects_cuda_on_macos_arm64() -> None:
    """Reject packaged CUDA inference on macOS."""
    runtime_platform = platform_support.RuntimePlatform(
        system="Darwin",
        machine="arm64",
    )

    try:
        platform_support.ensure_supported_packaged_runtime(
            device="cuda",
            runtime_platform=runtime_platform,
        )
    except platform_support.UnsupportedPackagedRuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected packaged CUDA rejection on macOS.")

    assert "Windows x86_64 and Linux x86_64" in message
    assert "macOS remains CPU-only" in message


def test_apply_runtime_platform_fallbacks_downgrades_nemo_on_windows() -> None:
    """Fall back to single-speaker diarization on unsupported Windows hosts."""
    config = {"audio": {"diarization": {"provider": "nemo"}}}
    reporter_messages: list[str] = []

    modified = platform_support.apply_runtime_platform_fallbacks(
        config,
        reporter=reporter_messages.append,
        runtime_platform=platform_support.RuntimePlatform(
            system="Windows",
            machine="AMD64",
        ),
    )

    assert modified is True
    assert config["audio"]["diarization"]["provider"] == "single_speaker"
    assert reporter_messages
    assert "not supported on Windows x86_64" in reporter_messages[0]


def test_apply_runtime_platform_fallbacks_downgrades_nemo_when_missing(
    monkeypatch,
) -> None:
    """Fall back when NeMo is configured but the optional dependency is absent."""
    monkeypatch.setattr(
        platform_support.importlib.util,
        "find_spec",
        lambda module_name: None if module_name == "nemo" else object(),
    )
    config = {"audio": {"diarization": {"provider": "nemo"}}}
    reporter_messages: list[str] = []

    modified = platform_support.apply_runtime_platform_fallbacks(
        config,
        reporter=reporter_messages.append,
        runtime_platform=platform_support.RuntimePlatform(
            system="Linux",
            machine="x86_64",
        ),
    )

    assert modified is True
    assert config["audio"]["diarization"]["provider"] == "single_speaker"
    assert "not installed" in reporter_messages[0]


def test_apply_runtime_platform_fallbacks_keeps_nemo_when_supported_and_installed(
    monkeypatch,
) -> None:
    """Leave the configured NeMo backend in place when it is usable."""
    monkeypatch.setattr(
        platform_support.importlib.util,
        "find_spec",
        lambda module_name: object(),
    )
    config = {"audio": {"diarization": {"provider": "nemo"}}}

    modified = platform_support.apply_runtime_platform_fallbacks(
        config,
        runtime_platform=platform_support.RuntimePlatform(
            system="Linux",
            machine="x86_64",
        ),
    )

    assert modified is False
    assert config["audio"]["diarization"]["provider"] == "nemo"
