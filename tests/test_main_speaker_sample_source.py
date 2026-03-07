from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
import types

import pytest
from audio_pipeline.eta import LinearStageEtaStrategy, StageSpeedProfile
from orchestration.transcript import Transcript

pytest.importorskip("a2a")


def _import_app_main():
    """Import ``main`` while tolerating environments where numpy is unavailable to pytest."""
    if "main" in sys.modules:
        return sys.modules["main"]

    stubbed_numpy = False
    if "numpy" not in sys.modules:
        try:
            importlib.import_module("numpy")
        except ModuleNotFoundError:
            numpy_stub = types.ModuleType("numpy")
            numpy_stub.ndarray = object
            sys.modules["numpy"] = numpy_stub
            stubbed_numpy = True

    stubbed_jinja2 = False
    if "jinja2" not in sys.modules:
        try:
            importlib.import_module("jinja2")
        except ModuleNotFoundError:
            jinja2_stub = types.ModuleType("jinja2")

            class _FileSystemLoader:
                def __init__(self, _path: str) -> None:
                    self.path = _path

            class _Environment:
                def __init__(self, *args, **kwargs) -> None:
                    self.args = args
                    self.kwargs = kwargs

                def get_template(self, template_name: str):  # noqa: ANN204
                    class _Template:
                        def __init__(self, name: str) -> None:
                            self.name = name

                        def render(self, **kwargs) -> str:
                            return f"{self.name}:{kwargs}"

                    return _Template(template_name)

            def _select_autoescape() -> bool:
                return False

            jinja2_stub.Environment = _Environment
            jinja2_stub.FileSystemLoader = _FileSystemLoader
            jinja2_stub.select_autoescape = _select_autoescape
            sys.modules["jinja2"] = jinja2_stub
            stubbed_jinja2 = True

    module = importlib.import_module("main")
    if stubbed_numpy:
        sys.modules.pop("numpy", None)
    if stubbed_jinja2:
        sys.modules.pop("jinja2", None)
    return module

@dataclass(slots=True, frozen=True)
class _SampleResult:
    output_dir: Path
    manifest_path: Path
    generated_at_utc: str
    artifacts: tuple[object, ...]


class _StubSampleGenerator:
    def generate(
        self,
        *,
        transcript_payload: dict[str, object],
        source_audio_path: Path,
        output_dir: Path,
        progress_callback=None,
    ) -> _SampleResult:
        del transcript_payload, source_audio_path, progress_callback
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return _SampleResult(
            output_dir=output_dir,
            manifest_path=manifest_path,
            generated_at_utc="2026-01-01T00:00:00+00:00",
            artifacts=(),
        )


class _LearningSampleGenerator:
    def generate(
        self,
        *,
        transcript_payload: dict[str, object],
        source_audio_path: Path,
        output_dir: Path,
        progress_callback=None,
    ) -> _SampleResult:
        del transcript_payload, source_audio_path, progress_callback
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return _SampleResult(
            output_dir=output_dir,
            manifest_path=manifest_path,
            generated_at_utc="2026-01-01T00:00:00+00:00",
            artifacts=(object(),),
        )


def test_post_transcript_step_enforces_vocals_source_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_main = _import_app_main()
    captured_source_modes: list[str] = []

    def _capture_source_mode(
        *,
        source_mode: str,
        transcript_metadata: dict[str, object] | None,
        configured_audio_path: str,
        base_dir: Path,
    ) -> Path:
        del transcript_metadata, configured_audio_path, base_dir
        captured_source_modes.append(source_mode)
        vocals_path = tmp_path / "vocals.wav"
        vocals_path.write_bytes(b"vocals")
        return vocals_path

    monkeypatch.setattr(app_main, "resolve_sample_source_audio_path", _capture_source_mode)
    monkeypatch.setattr(
        app_main,
        "build_speaker_sample_generator",
        lambda cfg: _StubSampleGenerator(),
    )
    monkeypatch.setattr(app_main, "write_transcript_atomic", lambda payload, path: None)

    transcript = Transcript.from_mapping(
        {
            "segments": [{"speaker": "SPEAKER_00", "text": "text"}],
            "metadata": {},
        }
    )
    updated_transcript = app_main._run_post_transcript_speaker_sample_step(
        stage_start="stage-1",
        audio_cfg_dict={
            "audio_path": "audio.wav",
            "work_dir": "artifacts/audio_stage",
            "speaker_samples": {
                "enabled": True,
                "source_audio": "source",
                "output_dir_name": "speaker_samples",
            },
        },
        project_root=tmp_path,
        transcript_path="artifacts/transcripts/latest.transcript.json",
        transcript=transcript,
    )

    assert captured_source_modes == ["vocals"]
    assert (
        updated_transcript.metadata.get("speaker_samples_manifest_path")
        == str(tmp_path / "artifacts" / "audio_stage" / "speaker_samples" / "manifest.json")
    )


def test_post_transcript_step_hides_first_run_eta_and_persists_learning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_main = _import_app_main()
    messages: list[str] = []

    def _capture(
        event_type: str,
        message: str,
        **kwargs,
    ) -> None:
        del event_type, kwargs
        messages.append(message)

    monkeypatch.setattr(
        app_main,
        "resolve_sample_source_audio_path",
        lambda **kwargs: tmp_path / "vocals.wav",
    )
    monkeypatch.setattr(
        app_main,
        "build_speaker_sample_generator",
        lambda cfg: _LearningSampleGenerator(),
    )
    monkeypatch.setattr(app_main, "write_transcript_atomic", lambda payload, path: None)
    monkeypatch.setattr(app_main.event_bus, "publish", _capture)

    vocals_path = tmp_path / "vocals.wav"
    vocals_path.write_bytes(b"vocals")
    profile_path = tmp_path / "eta_profile.json"
    transcript = Transcript.from_mapping(
        {
            "segments": [{"speaker": "SPEAKER_00", "text": "text"}],
            "metadata": {},
        }
    )
    strategy = LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=1.0, cuda=1.0),
        transcription=StageSpeedProfile(cpu=1.0, cuda=1.0),
        diarization=StageSpeedProfile(cpu=1.0, cuda=1.0),
    )

    app_main._run_post_transcript_speaker_sample_step(
        stage_start="stage-1",
        audio_cfg_dict={
            "audio_path": "audio.wav",
            "work_dir": "artifacts/audio_stage",
            "speaker_samples": {
                "enabled": True,
                "output_dir_name": "speaker_samples",
            },
        },
        project_root=tmp_path,
        transcript_path="artifacts/transcripts/latest.transcript.json",
        transcript=transcript,
        eta_strategy=strategy,
        eta_profile_path=profile_path,
        eta_profile_context={"device": "cpu"},
        eta_update_interval_seconds=0.0,
    )

    assert not any("estimated time left" in message for message in messages)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert payload["unit_stages"]["speaker_samples"]["samples"] == 1


def test_post_transcript_stage_two_reuses_existing_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_main = _import_app_main()
    manifest_path = tmp_path / "speaker_samples" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"samples": []}', encoding="utf-8")

    monkeypatch.setattr(
        app_main,
        "resolve_sample_source_audio_path",
        lambda **kwargs: pytest.fail("speaker samples should be reused, not regenerated"),
    )
    monkeypatch.setattr(
        app_main,
        "build_speaker_sample_generator",
        lambda cfg: pytest.fail("speaker sample generator should not run"),
    )
    monkeypatch.setattr(
        app_main,
        "write_transcript_atomic",
        lambda payload, path: pytest.fail("transcript should not be rewritten"),
    )

    transcript = Transcript.from_mapping(
        {
            "segments": [{"speaker": "SPEAKER_00", "text": "text"}],
            "metadata": {"speaker_samples_manifest_path": str(manifest_path)},
        }
    )

    reused_transcript = app_main._run_post_transcript_speaker_sample_step(
        stage_start="stage-2",
        audio_cfg_dict={
            "audio_path": "audio.wav",
            "work_dir": "artifacts/audio_stage",
            "speaker_samples": {
                "enabled": True,
                "output_dir_name": "speaker_samples",
            },
        },
        project_root=tmp_path,
        transcript_path="artifacts/transcripts/latest.transcript.json",
        transcript=transcript,
    )

    assert reused_transcript == transcript


def test_wait_for_agent_servers_uses_shared_wait_strategy(monkeypatch) -> None:
    app_main = _import_app_main()
    calls: list[dict[str, object]] = []
    sleep_calls: list[float] = []

    class _FakeChecker:
        def wait_for_many(
            self,
            servers,
            *,
            overall_timeout_seconds: float,
            poll_interval_seconds: float,
            request_timeout_seconds: float,
        ) -> bool:
            calls.append(
                {
                    "servers": list(servers),
                    "overall_timeout_seconds": overall_timeout_seconds,
                    "poll_interval_seconds": poll_interval_seconds,
                    "request_timeout_seconds": request_timeout_seconds,
                }
            )
            return True

    monkeypatch.setattr(app_main.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    app_main._wait_for_agent_servers(
        checker=_FakeChecker(),  # type: ignore[arg-type]
        servers=[("Summarizer", 9010), ("Critic", 9011)],
        overall_timeout_seconds=8.0,
        poll_interval_seconds=0.1,
        request_timeout_seconds=0.5,
    )

    assert sleep_calls == []
    assert calls == [
        {
            "servers": [("Summarizer", 9010), ("Critic", 9011)],
            "overall_timeout_seconds": 8.0,
            "poll_interval_seconds": 0.1,
            "request_timeout_seconds": 0.5,
        }
    ]
