"""Startup loading-status coverage for the runtime CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

import card_framework.cli.main as runtime_main
from card_framework.shared.events import event_bus


class _FakeAudioOrchestrator:
    """Minimal audio-orchestrator stub for startup tests."""

    eta_profile_filename = "eta_profile.json"
    eta_strategy = None
    eta_update_interval_seconds = 10.0
    eta_progress_smoothing = 0.25
    eta_overrun_factor = 1.15
    eta_headroom_seconds = 1.0

    def _build_eta_profile_context(self, *, device: str) -> dict[str, str]:
        del device
        return {}

    def _load_eta_profile(self, *, profile_path: Path, context: dict[str, str]) -> None:
        del profile_path, context

    def _save_eta_profile(self, *, profile_path: Path, context: dict[str, str]) -> None:
        del profile_path, context


class _FakeProvider:
    """Minimal LLM provider stub."""


class _FakeAgentClient:
    """Minimal shared agent client stub."""

    def __init__(self, *, event_bus) -> None:
        self.event_bus = event_bus


class _FakeExecutor:
    """Minimal A2A executor stub."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs


class _FakeHealthChecker:
    """Minimal health checker that reports immediate readiness."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def wait_for_many(self, *args, **kwargs) -> bool:
        del args, kwargs
        return True


class _FakeTranscriptIndex:
    """Minimal transcript-index stub."""

    def __init__(self, *, embedding_provider) -> None:
        self.embedding_provider = embedding_provider


class _FakeOrchestrator:
    """Minimal loop orchestrator stub."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs


class _FakeStageOrchestrator:
    """Minimal stage orchestrator stub."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    async def run(self, *, transcript, retrieval_enabled: bool) -> None:
        del transcript, retrieval_enabled


def test_main_emits_timed_loading_messages_for_startup_phases(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Emit timed system messages for transcript, calibration, providers, and readiness."""
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        (
            '{"segments": [{"speaker": "SPEAKER_00", "start_time": 0, '
            '"end_time": 1000, "text": "hello world"}], "metadata": {}}\n'
        ),
        encoding="utf-8",
    )

    cfg = OmegaConf.create(
        {
            "logging": {"print_to_terminal": False, "enabled": False},
            "audio": {
                "device": "cpu",
                "work_dir": "artifacts/audio_stage",
                "audio_path": "",
                "voice_clone": {
                    "enabled": False,
                    "live_drafting": {"enabled": False},
                    "gpu_heartbeat": {},
                    "emo_presets": {},
                },
                "interjector": {"enabled": False},
                "speaker_samples": {"enabled": False},
            },
            "orchestrator": {"target_seconds": 60, "timeouts": {}, "max_iterations": 1},
            "pipeline": {"start_stage": "stage-2"},
            "transcript_path": "transcript.json",
            "llm": {"_target_": "fake.llm"},
            "stage_llm": {},
            "embedding": {
                "_target_": "card_framework.providers.null_provider.NoOpEmbeddingProvider"
            },
            "ports": {"retrieval": 9101, "summarizer": 9102, "critic": 9103},
            "agents": {"summarizer": {}, "critic": {}},
        }
    )

    monkeypatch.setattr("card_framework.cli.main.configure_logger", lambda _: None)
    monkeypatch.setattr("card_framework.cli.main.hydra.utils.get_original_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(
        "card_framework.cli.main._load_audio_pipeline_builders",
        lambda **_: (
            lambda _: _FakeAudioOrchestrator(),
            lambda *args, **kwargs: None,
            lambda _: None,
            lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setattr(
        "card_framework.cli.main._run_post_transcript_speaker_sample_step",
        lambda **kwargs: kwargs["transcript"],
    )
    monkeypatch.setattr(
        "card_framework.cli.main.ensure_voice_clone_calibration",
        lambda **_: SimpleNamespace(artifact_path=tmp_path / "voice_clone_calibration.json"),
    )
    monkeypatch.setattr(
        "card_framework.cli.main._load_agent_runtime_helpers",
        lambda **_: (
            _FakeAgentClient,
            _FakeExecutor,
            _FakeHealthChecker,
            _FakeExecutor,
            _FakeExecutor,
            _FakeTranscriptIndex,
        ),
    )
    monkeypatch.setattr(
        "card_framework.cli.main._instantiate_llm_provider",
        lambda llm_cfg, *, enable_logging: _FakeProvider(),
    )
    monkeypatch.setattr(
        "card_framework.cli.main._resolve_stage_llm",
        lambda cfg, *, shared_llm, enable_logging: (shared_llm, "shared"),
    )
    monkeypatch.setattr("card_framework.cli.main._build_a2a_app", lambda **_: object())
    monkeypatch.setattr(
        "card_framework.cli.main._run_server_in_thread",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("card_framework.cli.main.Orchestrator", _FakeOrchestrator)
    monkeypatch.setattr("card_framework.cli.main.StageOrchestrator", _FakeStageOrchestrator)

    system_messages: list[str] = []

    def _capture(message: str) -> None:
        system_messages.append(message)

    event_bus.subscribe("system_message", _capture)
    try:
        runtime_main.main.__wrapped__(cfg)
    finally:
        event_bus.unsubscribe("system_message", _capture)

    assert any(message.startswith("Loading transcript from ") for message in system_messages)
    assert any(message.startswith("Loaded transcript from ") for message in system_messages)
    assert "Loading voice-clone calibration artifact..." in system_messages
    assert any(
        message.startswith("Loaded voice-clone calibration artifact in ")
        for message in system_messages
    )
    assert "Loading LLM providers..." in system_messages
    assert any(
        message.startswith("Loaded LLM providers in ")
        for message in system_messages
    )
    assert "Loading A2A server readiness..." in system_messages
    assert any(
        message.startswith("Loaded A2A server readiness in ")
        for message in system_messages
    )
