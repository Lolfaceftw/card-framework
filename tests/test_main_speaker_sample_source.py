from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
import types

import pytest
from audio_pipeline.errors import NonRetryableAudioStageError
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


def _write_stage_two_manifest(
    *,
    manifest_path: Path,
    speakers: list[str],
) -> None:
    """Persist a minimal speaker-sample manifest for stage-2 bootstrap tests."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    for speaker in speakers:
        sample_path = manifest_path.parent / f"{speaker}.wav"
        sample_path.write_bytes(b"sample")
        samples.append(
            {
                "speaker": speaker,
                "path": str(sample_path),
                "duration_ms": 30_000,
            }
        )
    manifest_path.write_text(
        json.dumps({"samples": samples}),
        encoding="utf-8",
    )


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


@dataclass(slots=True, frozen=True)
class _BootstrapResult:
    transcript_path: Path
    generation_result: _SampleResult


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


def test_post_transcript_stage_two_bootstraps_missing_manifest_from_audio(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_main = _import_app_main()
    source_audio_path = tmp_path / "audio.wav"
    source_audio_path.write_bytes(b"audio")
    written_payloads: list[tuple[dict[str, object], Path]] = []
    captured: dict[str, Path] = {}

    def _bootstrap_speaker_samples(
        *,
        project_root: Path,
        audio_cfg: dict[str, object],
        audio_path: Path,
        bootstrap_transcript_path: Path,
        bootstrap_audio_work_dir: Path,
        speaker_samples_output_dir: Path,
    ) -> _BootstrapResult:
        del project_root, audio_cfg
        captured["audio_path"] = audio_path
        captured["bootstrap_transcript_path"] = bootstrap_transcript_path
        captured["bootstrap_audio_work_dir"] = bootstrap_audio_work_dir
        captured["speaker_samples_output_dir"] = speaker_samples_output_dir
        bootstrap_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_transcript_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker": "SPEAKER_00",
                            "start_time": 0,
                            "end_time": 10_000,
                            "text": "alpha beta gamma delta",
                        }
                    ],
                    "metadata": {"vocals_audio_path": "vocals.wav"},
                }
            ),
            encoding="utf-8",
        )
        speaker_samples_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = speaker_samples_output_dir / "manifest.json"
        _write_stage_two_manifest(
            manifest_path=manifest_path,
            speakers=["SPEAKER_00"],
        )
        return _BootstrapResult(
            transcript_path=bootstrap_transcript_path,
            generation_result=_SampleResult(
                output_dir=speaker_samples_output_dir,
                manifest_path=manifest_path,
                generated_at_utc="2026-01-01T00:00:00+00:00",
                artifacts=(object(), object()),
            ),
        )

    monkeypatch.setattr(
        app_main,
        "bootstrap_speaker_samples_from_audio",
        _bootstrap_speaker_samples,
    )
    monkeypatch.setattr(
        app_main,
        "resolve_sample_source_audio_path",
        lambda **kwargs: pytest.fail("stage-2 should bootstrap from audio, not clip transcript"),
    )
    monkeypatch.setattr(
        app_main,
        "build_speaker_sample_generator",
        lambda cfg: pytest.fail("direct speaker sample generator should not run in stage-2 bootstrap"),
    )
    monkeypatch.setattr(
        app_main,
        "write_transcript_atomic",
        lambda payload, path: written_payloads.append((payload, path)),
    )

    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 10_000,
                    "text": "alpha beta gamma delta",
                }
            ],
            "metadata": {"source_audio_path": str(source_audio_path)},
        }
    )

    updated_transcript = app_main._run_post_transcript_speaker_sample_step(
        stage_start="stage-2",
        audio_cfg_dict={
            "work_dir": "artifacts/audio_stage/runs/test_run",
            "speaker_samples": {
                "enabled": True,
                "output_dir_name": "speaker_samples",
            },
        },
        project_root=tmp_path,
        transcript_path="transcript.json",
        transcript=transcript,
    )

    expected_work_dir = tmp_path / "artifacts" / "audio_stage" / "runs" / "test_run"
    assert captured["audio_path"] == source_audio_path.resolve()
    assert captured["bootstrap_transcript_path"] == (
        expected_work_dir / "speaker_sample_bootstrap.transcript.json"
    )
    assert captured["bootstrap_audio_work_dir"] == (
        expected_work_dir / "speaker_sample_bootstrap_audio"
    )
    assert captured["speaker_samples_output_dir"] == (
        expected_work_dir / "speaker_samples"
    )
    assert updated_transcript.metadata["speaker_samples_manifest_path"] == str(
        expected_work_dir / "speaker_samples" / "manifest.json"
    )
    assert written_payloads[0][1] == (tmp_path / "transcript.json")
    assert (
        written_payloads[0][0]["metadata"]["speaker_samples_manifest_path"]
        == str(expected_work_dir / "speaker_samples" / "manifest.json")
    )


def test_post_transcript_stage_two_bootstrap_remaps_speaker_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_main = _import_app_main()
    source_audio_path = tmp_path / "audio.wav"
    source_audio_path.write_bytes(b"audio")
    manifest_path = (
        tmp_path
        / "artifacts"
        / "audio_stage"
        / "runs"
        / "test_run"
        / "speaker_samples"
        / "manifest.json"
    )

    def _bootstrap_speaker_samples(
        *,
        project_root: Path,
        audio_cfg: dict[str, object],
        audio_path: Path,
        bootstrap_transcript_path: Path,
        bootstrap_audio_work_dir: Path,
        speaker_samples_output_dir: Path,
    ) -> _BootstrapResult:
        del project_root, audio_cfg, audio_path, bootstrap_audio_work_dir
        bootstrap_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_transcript_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker": "BOOT_A",
                            "start_time": 0,
                            "end_time": 15_000,
                            "text": "alpha beta gamma delta",
                        },
                        {
                            "speaker": "BOOT_B",
                            "start_time": 15_000,
                            "end_time": 30_000,
                            "text": "theta lambda mu nu",
                        },
                    ],
                    "metadata": {"vocals_audio_path": "vocals.wav"},
                }
            ),
            encoding="utf-8",
        )
        _write_stage_two_manifest(
            manifest_path=manifest_path,
            speakers=["BOOT_A", "BOOT_B"],
        )
        return _BootstrapResult(
            transcript_path=bootstrap_transcript_path,
            generation_result=_SampleResult(
                output_dir=speaker_samples_output_dir,
                manifest_path=manifest_path,
                generated_at_utc="2026-01-01T00:00:00+00:00",
                artifacts=(object(), object()),
            ),
        )

    monkeypatch.setattr(
        app_main,
        "bootstrap_speaker_samples_from_audio",
        _bootstrap_speaker_samples,
    )
    monkeypatch.setattr(app_main, "write_transcript_atomic", lambda payload, path: None)

    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 15_000,
                    "text": "alpha beta gamma delta",
                },
                {
                    "speaker": "SPEAKER_01",
                    "start_time": 15_000,
                    "end_time": 30_000,
                    "text": "theta lambda mu nu",
                },
            ],
            "metadata": {"source_audio_path": str(source_audio_path)},
        }
    )

    app_main._run_post_transcript_speaker_sample_step(
        stage_start="stage-2",
        audio_cfg_dict={
            "work_dir": "artifacts/audio_stage/runs/test_run",
            "speaker_samples": {
                "enabled": True,
                "output_dir_name": "speaker_samples",
            },
        },
        project_root=tmp_path,
        transcript_path="transcript.json",
        transcript=transcript,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [sample["speaker"] for sample in payload["samples"]] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert [sample["bootstrap_speaker"] for sample in payload["samples"]] == [
        "BOOT_A",
        "BOOT_B",
    ]
    assert payload["stage_two_bootstrap_speaker_mapping"] == {
        "BOOT_A": "SPEAKER_00",
        "BOOT_B": "SPEAKER_01",
    }


def test_post_transcript_stage_two_bootstrap_rejects_incompatible_audio(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_main = _import_app_main()
    source_audio_path = tmp_path / "audio.wav"
    source_audio_path.write_bytes(b"audio")
    write_calls: list[tuple[dict[str, object], Path]] = []

    def _bootstrap_speaker_samples(
        *,
        project_root: Path,
        audio_cfg: dict[str, object],
        audio_path: Path,
        bootstrap_transcript_path: Path,
        bootstrap_audio_work_dir: Path,
        speaker_samples_output_dir: Path,
    ) -> _BootstrapResult:
        del project_root, audio_cfg, audio_path, bootstrap_audio_work_dir
        bootstrap_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_transcript_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker": "SPEAKER_00",
                            "start_time": 0,
                            "end_time": 10_000,
                            "text": "zebra yak xylophone walrus",
                        },
                        {
                            "speaker": "SPEAKER_00",
                            "start_time": 30_000,
                            "end_time": 40_000,
                            "text": "violet umber topaz saffron",
                        },
                        {
                            "speaker": "SPEAKER_00",
                            "start_time": 60_000,
                            "end_time": 70_000,
                            "text": "quartz ruby silver titanium",
                        },
                    ],
                    "metadata": {"vocals_audio_path": "vocals.wav"},
                }
            ),
            encoding="utf-8",
        )
        manifest_path = speaker_samples_output_dir / "manifest.json"
        _write_stage_two_manifest(
            manifest_path=manifest_path,
            speakers=["SPEAKER_00"],
        )
        return _BootstrapResult(
            transcript_path=bootstrap_transcript_path,
            generation_result=_SampleResult(
                output_dir=speaker_samples_output_dir,
                manifest_path=manifest_path,
                generated_at_utc="2026-01-01T00:00:00+00:00",
                artifacts=(object(),),
            ),
        )

    monkeypatch.setattr(
        app_main,
        "bootstrap_speaker_samples_from_audio",
        _bootstrap_speaker_samples,
    )
    monkeypatch.setattr(
        app_main,
        "write_transcript_atomic",
        lambda payload, path: write_calls.append((payload, path)),
    )

    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 10_000,
                    "text": "alpha beta gamma delta",
                },
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 30_000,
                    "end_time": 40_000,
                    "text": "epsilon zeta eta theta",
                },
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 60_000,
                    "end_time": 70_000,
                    "text": "iota kappa lambda mu",
                },
            ],
            "metadata": {"source_audio_path": str(source_audio_path)},
        }
    )

    with pytest.raises(
        NonRetryableAudioStageError,
        match="does not appear to match the reusable transcript",
    ):
        app_main._run_post_transcript_speaker_sample_step(
            stage_start="stage-2",
            audio_cfg_dict={
                "work_dir": "artifacts/audio_stage/runs/test_run",
                "speaker_samples": {
                    "enabled": True,
                    "output_dir_name": "speaker_samples",
                },
            },
            project_root=tmp_path,
            transcript_path="transcript.json",
            transcript=transcript,
        )

    assert write_calls == []


def test_post_transcript_stage_two_bootstrap_rejects_missing_speaker_coverage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_main = _import_app_main()
    source_audio_path = tmp_path / "audio.wav"
    source_audio_path.write_bytes(b"audio")

    def _bootstrap_speaker_samples(
        *,
        project_root: Path,
        audio_cfg: dict[str, object],
        audio_path: Path,
        bootstrap_transcript_path: Path,
        bootstrap_audio_work_dir: Path,
        speaker_samples_output_dir: Path,
    ) -> _BootstrapResult:
        del project_root, audio_cfg, audio_path, bootstrap_audio_work_dir
        bootstrap_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_transcript_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker": "SPEAKER_00",
                            "start_time": 0,
                            "end_time": 15_000,
                            "text": "alpha beta gamma delta",
                        },
                        {
                            "speaker": "SPEAKER_01",
                            "start_time": 15_000,
                            "end_time": 30_000,
                            "text": "theta lambda mu nu",
                        },
                    ],
                    "metadata": {"vocals_audio_path": "vocals.wav"},
                }
            ),
            encoding="utf-8",
        )
        manifest_path = speaker_samples_output_dir / "manifest.json"
        _write_stage_two_manifest(
            manifest_path=manifest_path,
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        return _BootstrapResult(
            transcript_path=bootstrap_transcript_path,
            generation_result=_SampleResult(
                output_dir=speaker_samples_output_dir,
                manifest_path=manifest_path,
                generated_at_utc="2026-01-01T00:00:00+00:00",
                artifacts=(object(), object()),
            ),
        )

    monkeypatch.setattr(
        app_main,
        "bootstrap_speaker_samples_from_audio",
        _bootstrap_speaker_samples,
    )
    monkeypatch.setattr(app_main, "write_transcript_atomic", lambda payload, path: None)

    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 15_000,
                    "text": "alpha beta gamma delta",
                },
                {
                    "speaker": "SPEAKER_01",
                    "start_time": 15_000,
                    "end_time": 30_000,
                    "text": "theta lambda mu nu",
                },
                {
                    "speaker": "SPEAKER_02",
                    "start_time": 30_000,
                    "end_time": 45_000,
                    "text": "omicron pi rho sigma",
                },
            ],
            "metadata": {"source_audio_path": str(source_audio_path)},
        }
    )

    with pytest.raises(
        NonRetryableAudioStageError,
        match="speaker coverage does not match the reusable transcript",
    ):
        app_main._run_post_transcript_speaker_sample_step(
            stage_start="stage-2",
            audio_cfg_dict={
                "work_dir": "artifacts/audio_stage/runs/test_run",
                "speaker_samples": {
                    "enabled": True,
                    "output_dir_name": "speaker_samples",
                },
            },
            project_root=tmp_path,
            transcript_path="transcript.json",
            transcript=transcript,
        )


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
