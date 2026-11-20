"""Startup loading-status coverage for the runtime CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

import card_framework.cli.main as runtime_main
from card_framework.runtime.pipeline_sessions import PipelineSessionStore
from card_framework.shared.events import event_bus


class _FakeAudioOrchestrator:
    """Minimal audio-orchestrator stub for startup tests."""

    eta_profile_filename = "eta_history.sqlite3"
    eta_strategy = None
    eta_update_interval_seconds = 10.0
    eta_progress_smoothing = 0.25
    eta_overrun_factor = 1.15
    eta_headroom_seconds = 1.0
    eta_profile_path: Path | None = None
    loaded_profile_paths: list[Path]
    saved_profile_paths: list[Path]

    def __init__(self) -> None:
        self.loaded_profile_paths = []
        self.saved_profile_paths = []

    def _build_eta_profile_context(self, *, device: str) -> dict[str, str]:
        del device
        return {}

    def _load_eta_profile(self, *, profile_path: Path, context: dict[str, str]) -> None:
        del context
        self.loaded_profile_paths.append(profile_path)

    def _save_eta_profile(self, *, profile_path: Path, context: dict[str, str]) -> None:
        del context
        self.saved_profile_paths.append(profile_path)


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


class _FakeUnreadyHealthChecker:
    """Minimal health checker that never reaches readiness."""

    def wait_for_many(self, *args, **kwargs) -> bool:
        del args, kwargs
        return False


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


class _InterruptingStageOrchestrator(_FakeStageOrchestrator):
    async def run(self, *, transcript, retrieval_enabled: bool) -> None:
        del transcript, retrieval_enabled
        raise KeyboardInterrupt


class _CapturingStageOrchestrator(_FakeStageOrchestrator):
    """Minimal stage orchestrator stub that captures construction kwargs."""

    captured_kwargs: dict[str, object] | None = None

    def __init__(self, *args, **kwargs) -> None:
        del args
        type(self).captured_kwargs = dict(kwargs)


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
    audio_orchestrator = _FakeAudioOrchestrator()
    monkeypatch.setattr(
        "card_framework.cli.main._load_audio_pipeline_builders",
        lambda **_: (
            lambda _: audio_orchestrator,
            lambda *args, **kwargs: None,
            lambda _: None,
            lambda *args, **kwargs: None,
            lambda *args, **kwargs: None,
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


def test_main_passes_unbounded_max_iterations_to_stage_orchestrator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Null max_iterations config should reach StageOrchestrator as None."""
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
            "orchestrator": {
                "target_seconds": 60,
                "timeouts": {},
                "max_iterations": None,
            },
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
    audio_orchestrator = _FakeAudioOrchestrator()
    monkeypatch.setattr(
        "card_framework.cli.main._load_audio_pipeline_builders",
        lambda **_: (
            lambda _: audio_orchestrator,
            lambda *args, **kwargs: None,
            lambda _: None,
            lambda *args, **kwargs: None,
            lambda *args, **kwargs: None,
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
    monkeypatch.setattr(
        "card_framework.cli.main.StageOrchestrator",
        _CapturingStageOrchestrator,
    )

    _CapturingStageOrchestrator.captured_kwargs = None
    runtime_main.main.__wrapped__(cfg)

    assert _CapturingStageOrchestrator.captured_kwargs is not None
    assert _CapturingStageOrchestrator.captured_kwargs["max_iterations"] is None


def test_main_preserves_zero_max_iterations_when_configured(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Explicit zero max_iterations should stay zero unless null is requested."""
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
            "orchestrator": {
                "target_seconds": 60,
                "timeouts": {},
                "max_iterations": 0,
            },
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
    audio_orchestrator = _FakeAudioOrchestrator()
    monkeypatch.setattr(
        "card_framework.cli.main._load_audio_pipeline_builders",
        lambda **_: (
            lambda _: audio_orchestrator,
            lambda *args, **kwargs: None,
            lambda _: None,
            lambda *args, **kwargs: None,
            lambda *args, **kwargs: None,
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
    monkeypatch.setattr(
        "card_framework.cli.main.StageOrchestrator",
        _CapturingStageOrchestrator,
    )

    _CapturingStageOrchestrator.captured_kwargs = None
    runtime_main.main.__wrapped__(cfg)

    assert _CapturingStageOrchestrator.captured_kwargs is not None
    assert _CapturingStageOrchestrator.captured_kwargs["max_iterations"] == 0


def test_instantiate_llm_provider_strips_runtime_metadata(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_instantiate(cfg):  # noqa: ANN001
        captured.update(runtime_main._to_plain_dict(cfg))
        return _FakeProvider()

    monkeypatch.setattr("card_framework.cli.main.hydra.utils.instantiate", _fake_instantiate)

    provider = runtime_main._instantiate_llm_provider(
        OmegaConf.create(
            {
                "_target_": "fake.llm",
                "api_key": "test-key",
                "context_window_tokens": 131072,
                "compaction_target_ratio": 0.65,
            }
        ),
        enable_logging=False,
    )

    assert isinstance(provider, _FakeProvider)
    assert captured["_target_"] == "fake.llm"
    assert captured["api_key"] == "test-key"
    assert "context_window_tokens" not in captured
    assert "compaction_target_ratio" not in captured


def test_main_uses_shared_eta_store_path_across_run_local_work_dirs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Resolve ETA persistence to one shared store instead of a run-local work dir."""
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
                "work_dir": "artifacts/audio_stage/runs/run123",
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
    audio_orchestrator = _FakeAudioOrchestrator()
    monkeypatch.setattr(
        "card_framework.cli.main._load_audio_pipeline_builders",
        lambda **_: (
            lambda _: audio_orchestrator,
            lambda *args, **kwargs: None,
            lambda _: None,
            lambda *args, **kwargs: None,
            lambda *args, **kwargs: None,
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

    runtime_main.main.__wrapped__(cfg)

    expected_store_path = (
        tmp_path / "artifacts" / "audio_stage" / "eta_history.sqlite3"
    ).resolve()
    assert audio_orchestrator.eta_profile_path == expected_store_path
    assert audio_orchestrator.loaded_profile_paths == [expected_store_path]
    assert audio_orchestrator.saved_profile_paths == [expected_store_path]


def test_wait_for_agent_servers_raises_runtime_error_when_readiness_times_out() -> None:
    """Raise the readiness failure instead of terminating the process."""
    with pytest.raises(
        RuntimeError,
        match="A2A servers did not become healthy in time.",
    ):
        runtime_main._wait_for_agent_servers(
            checker=_FakeUnreadyHealthChecker(),
            servers=[("Summarizer", 9102)],
            overall_timeout_seconds=0.01,
            poll_interval_seconds=0.01,
            request_timeout_seconds=0.01,
        )


def test_main_marks_incomplete_session_failed_and_raises_runtime_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Fail loudly when required session checkpoints are still missing."""
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        (
            '{"segments": [{"speaker": "SPEAKER_00", "start_time": 0, '
            '"end_time": 1000, "text": "hello world"}], "metadata": {}}\n'
        ),
        encoding="utf-8",
    )
    session_store_path = tmp_path / "sessions.sqlite3"
    session_summary_path = tmp_path / "session_summary.xml"
    PipelineSessionStore(session_store_path).create_or_replace_session(
        session_id="session-1",
        requested_start_stage="stage-2",
        prepared_overrides=["pipeline.start_stage=stage-2"],
        artifacts={
            "work_dir": str(tmp_path / "artifacts" / "audio_stage" / "runs" / "session-1"),
            "transcript_path": str(transcript_path),
            "workspace_summary_path": str(tmp_path / "summary.xml"),
            "session_summary_path": str(session_summary_path),
            "voice_clone_manifest_path": str(
                tmp_path
                / "artifacts"
                / "audio_stage"
                / "runs"
                / "session-1"
                / "voice_clone"
                / "manifest.json"
            ),
            "interjector_manifest_path": str(
                tmp_path
                / "artifacts"
                / "audio_stage"
                / "runs"
                / "session-1"
                / "interjector"
                / "interjector_manifest.json"
            ),
        },
        options={
            "voice_clone_enabled": False,
            "interjector_enabled": False,
        },
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
            "session": {
                "enabled": True,
                "session_id": "session-1",
                "store_path": str(session_store_path),
                "summary_output_path": str(session_summary_path),
            },
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
    monkeypatch.setattr(
        "card_framework.cli.main.hydra.utils.get_original_cwd",
        lambda: str(tmp_path),
    )
    audio_orchestrator = _FakeAudioOrchestrator()
    monkeypatch.setattr(
        "card_framework.cli.main._load_audio_pipeline_builders",
        lambda **_: (
            lambda _: audio_orchestrator,
            lambda *args, **kwargs: None,
            lambda _: None,
            lambda *args, **kwargs: None,
            lambda *args, **kwargs: None,
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

    with pytest.raises(
        RuntimeError,
        match="Pipeline exited without completing all required stages.",
    ):
        runtime_main.main.__wrapped__(cfg)

    record = PipelineSessionStore(session_store_path).get_session("session-1")
    assert record is not None
    assert record.status == "failed"
    assert record.resume_stage == "stage-2"
    assert "Pipeline exited without completing all required stages." in record.last_error


def test_main_converts_keyboard_interrupt_into_exit_130(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exit cleanly on Ctrl+C after persisting a paused session state."""
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        (
            '{"segments": [{"speaker": "SPEAKER_00", "start_time": 0, '
            '"end_time": 1000, "text": "hello world"}], "metadata": {}}\n'
        ),
        encoding="utf-8",
    )
    session_store_path = tmp_path / "sessions.sqlite3"
    session_summary_path = tmp_path / "session_summary.xml"
    PipelineSessionStore(session_store_path).create_or_replace_session(
        session_id="session-2",
        requested_start_stage="stage-2",
        prepared_overrides=["pipeline.start_stage=stage-2"],
        artifacts={
            "work_dir": str(tmp_path / "artifacts" / "audio_stage" / "runs" / "session-2"),
            "transcript_path": str(transcript_path),
            "workspace_summary_path": str(tmp_path / "summary.xml"),
            "session_summary_path": str(session_summary_path),
            "voice_clone_manifest_path": str(
                tmp_path
                / "artifacts"
                / "audio_stage"
                / "runs"
                / "session-2"
                / "voice_clone"
                / "manifest.json"
            ),
            "interjector_manifest_path": str(
                tmp_path
                / "artifacts"
                / "audio_stage"
                / "runs"
                / "session-2"
                / "interjector"
                / "interjector_manifest.json"
            ),
        },
        options={
            "voice_clone_enabled": False,
            "interjector_enabled": False,
        },
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
            "session": {
                "enabled": True,
                "session_id": "session-2",
                "store_path": str(session_store_path),
                "summary_output_path": str(session_summary_path),
            },
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
    monkeypatch.setattr(
        "card_framework.cli.main.hydra.utils.get_original_cwd",
        lambda: str(tmp_path),
    )
    audio_orchestrator = _FakeAudioOrchestrator()
    monkeypatch.setattr(
        "card_framework.cli.main._load_audio_pipeline_builders",
        lambda **_: (
            lambda _: audio_orchestrator,
            lambda *args, **kwargs: None,
            lambda _: None,
            lambda *args, **kwargs: None,
            lambda *args, **kwargs: None,
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
    monkeypatch.setattr(
        "card_framework.cli.main.StageOrchestrator",
        _InterruptingStageOrchestrator,
    )

    with pytest.raises(SystemExit) as exc_info:
        runtime_main.main.__wrapped__(cfg)

    assert exc_info.value.code == 130
    record = PipelineSessionStore(session_store_path).get_session("session-2")
    assert record is not None
    assert record.status == "paused"
    assert record.resume_stage == "stage-2"
