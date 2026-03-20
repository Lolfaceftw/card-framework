from __future__ import annotations

import json
from pathlib import Path
import subprocess
import pytest

from card_framework.audio_pipeline.calibration import (
    bootstrap_speaker_samples_from_diarization,
)
from card_framework.audio_pipeline.contracts import DiarizationTurn
from card_framework.audio_pipeline.errors import NonRetryableAudioStageError
from card_framework.audio_pipeline.speaker_samples import (
    AudioSlice,
    SpeakerSampleArtifact,
    SpeakerSampleGenerationResult,
)


@pytest.mark.parametrize(
    ("source_mode", "expected_output_name"),
    [
        ("source", "source.wav"),
        ("vocals", "generated_vocals.wav"),
    ],
)
def test_bootstrap_speaker_samples_from_diarization_skips_asr_and_respects_source_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_mode: str,
    expected_output_name: str,
) -> None:
    source_audio_path = (tmp_path / "source.wav").resolve()
    source_audio_path.write_bytes(b"RIFF")
    expected_vocals_path = (tmp_path / "generated_vocals.wav").resolve()
    captured: dict[str, object] = {}

    class FakeSeparator:
        def separate_vocals(self, input_audio_path: Path, output_dir: Path, *, device: str, progress_callback=None) -> Path:  # type: ignore[no-untyped-def]
            del progress_callback
            captured["separator_input_audio_path"] = input_audio_path.resolve()
            captured["separator_device"] = device
            output_dir.mkdir(parents=True, exist_ok=True)
            expected_vocals_path.write_bytes(b"RIFF")
            return expected_vocals_path

    class FakeDiarizer:
        def diarize(self, audio_path: Path, output_dir: Path, *, device: str, progress_callback=None) -> list[DiarizationTurn]:  # type: ignore[no-untyped-def]
            del progress_callback
            captured["diarizer_audio_path"] = audio_path.resolve()
            captured["diarizer_device"] = device
            output_dir.mkdir(parents=True, exist_ok=True)
            return [
                DiarizationTurn(
                    speaker="BOOT_00",
                    start_time_ms=0,
                    end_time_ms=2_000,
                ),
                DiarizationTurn(
                    speaker="BOOT_01",
                    start_time_ms=2_500,
                    end_time_ms=5_000,
                ),
            ]

        def release_cached_resources(self, *, device: str) -> None:
            captured["released_device"] = device

    class FakeGenerator:
        def generate(self, *, transcript_payload, source_audio_path: Path, output_dir: Path, progress_callback=None) -> SpeakerSampleGenerationResult:  # type: ignore[no-untyped-def]
            del progress_callback
            captured["generator_transcript_payload"] = transcript_payload
            captured["generator_source_audio_path"] = source_audio_path.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = (output_dir / "manifest.json").resolve()
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_audio_path": str(source_audio_path),
                        "samples": [
                            {"speaker": "BOOT_00"},
                            {"speaker": "BOOT_01"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            artifact_path_0 = (output_dir / "speaker_00.wav").resolve()
            artifact_path_1 = (output_dir / "speaker_01.wav").resolve()
            artifact_path_0.write_bytes(b"RIFF")
            artifact_path_1.write_bytes(b"RIFF")
            return SpeakerSampleGenerationResult(
                output_dir=output_dir.resolve(),
                manifest_path=manifest_path,
                generated_at_utc="2026-03-20T00:00:00+00:00",
                artifacts=(
                    SpeakerSampleArtifact(
                        speaker="BOOT_00",
                        path=artifact_path_0,
                        requested_duration_ms=30_000,
                        actual_duration_ms=2_000,
                        status="shorter_than_requested",
                        dropped_slice_count=0,
                        ranges_ms=(AudioSlice(start_time_ms=0, end_time_ms=2_000),),
                    ),
                    SpeakerSampleArtifact(
                        speaker="BOOT_01",
                        path=artifact_path_1,
                        requested_duration_ms=30_000,
                        actual_duration_ms=2_500,
                        status="shorter_than_requested",
                        dropped_slice_count=0,
                        ranges_ms=(AudioSlice(start_time_ms=2_500, end_time_ms=5_000),),
                    ),
                ),
            )

    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.build_audio_to_script_orchestrator",
        lambda audio_cfg: (_ for _ in ()).throw(
            AssertionError("full audio orchestrator should not be used")
        ),
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration._should_delegate_diarization_bootstrap_to_docker",
        lambda *, audio_cfg: False,
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.ensure_module_available",
        lambda module_name: None,
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.build_source_separator",
        lambda audio_cfg: FakeSeparator(),
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.build_speaker_diarizer",
        lambda audio_cfg: FakeDiarizer(),
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.build_speaker_sample_generator",
        lambda audio_cfg: FakeGenerator(),
    )

    bootstrap_transcript_path = (tmp_path / "bootstrap.transcript.json").resolve()
    result = bootstrap_speaker_samples_from_diarization(
        project_root=tmp_path,
        audio_cfg={"device": "cpu", "speaker_samples": {"source_audio": source_mode}},
        audio_path=source_audio_path,
        bootstrap_transcript_path=bootstrap_transcript_path,
        bootstrap_audio_work_dir=(tmp_path / "audio_stage").resolve(),
        speaker_samples_output_dir=(tmp_path / "speaker_samples").resolve(),
    )

    assert result.transcript_path == bootstrap_transcript_path
    assert captured["separator_input_audio_path"] == source_audio_path
    assert captured["diarizer_audio_path"] == expected_vocals_path
    assert captured["released_device"] == "cpu"
    assert Path(captured["generator_source_audio_path"]).name == expected_output_name

    payload = json.loads(bootstrap_transcript_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["speaker_sample_bootstrap_mode"] == "diarization_only"
    assert payload["metadata"]["diarization_turns"] == [
        {"speaker": "BOOT_00", "start_time": 0, "end_time": 2_000},
        {"speaker": "BOOT_01", "start_time": 2_500, "end_time": 5_000},
    ]
    assert [segment["speaker"] for segment in payload["segments"]] == [
        "BOOT_00",
        "BOOT_01",
    ]
    assert all(segment["text"] == "" for segment in payload["segments"])


def test_bootstrap_speaker_samples_from_diarization_fails_when_no_usable_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_audio_path = (tmp_path / "source.wav").resolve()
    source_audio_path.write_bytes(b"RIFF")

    class FakeSeparator:
        def separate_vocals(self, input_audio_path: Path, output_dir: Path, *, device: str, progress_callback=None) -> Path:  # type: ignore[no-untyped-def]
            del input_audio_path, device, progress_callback
            output_dir.mkdir(parents=True, exist_ok=True)
            vocals_path = (output_dir / "vocals.wav").resolve()
            vocals_path.write_bytes(b"RIFF")
            return vocals_path

    class FakeDiarizer:
        def diarize(self, audio_path: Path, output_dir: Path, *, device: str, progress_callback=None) -> list[DiarizationTurn]:  # type: ignore[no-untyped-def]
            del audio_path, output_dir, device, progress_callback
            return [
                DiarizationTurn(
                    speaker="BOOT_00",
                    start_time_ms=1_000,
                    end_time_ms=1_000,
                )
            ]

    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.build_source_separator",
        lambda audio_cfg: FakeSeparator(),
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration._should_delegate_diarization_bootstrap_to_docker",
        lambda *, audio_cfg: False,
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.ensure_module_available",
        lambda module_name: None,
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.build_speaker_diarizer",
        lambda audio_cfg: FakeDiarizer(),
    )

    with pytest.raises(
        NonRetryableAudioStageError,
        match="produced no usable speaker turns",
    ):
        bootstrap_speaker_samples_from_diarization(
            project_root=tmp_path,
            audio_cfg={"device": "cpu", "speaker_samples": {"source_audio": "source"}},
            audio_path=source_audio_path,
            bootstrap_transcript_path=tmp_path / "bootstrap.transcript.json",
            bootstrap_audio_work_dir=(tmp_path / "audio_stage").resolve(),
            speaker_samples_output_dir=(tmp_path / "speaker_samples").resolve(),
        )


def test_bootstrap_speaker_samples_from_diarization_fails_fast_when_nemo_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_audio_path = (tmp_path / "source.wav").resolve()
    source_audio_path.write_bytes(b"RIFF")

    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.ensure_module_available",
        lambda module_name: (_ for _ in ()).throw(
            ModuleNotFoundError(module_name)
        ),
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration._should_delegate_diarization_bootstrap_to_docker",
        lambda *, audio_cfg: False,
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.build_source_separator",
        lambda audio_cfg: (_ for _ in ()).throw(
            AssertionError("separator should not run when diarizer dependency is missing")
        ),
    )

    with pytest.raises(ModuleNotFoundError, match="nemo"):
        bootstrap_speaker_samples_from_diarization(
            project_root=tmp_path,
            audio_cfg={
                "device": "cpu",
                "diarization": {"provider": "nemo"},
                "speaker_samples": {"source_audio": "source"},
            },
            audio_path=source_audio_path,
            bootstrap_transcript_path=tmp_path / "bootstrap.transcript.json",
            bootstrap_audio_work_dir=(tmp_path / "audio_stage").resolve(),
            speaker_samples_output_dir=(tmp_path / "speaker_samples").resolve(),
        )


def test_bootstrap_speaker_samples_from_diarization_uses_docker_when_nemo_is_unavailable_locally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path.resolve()
    source_audio_path = (project_root / "audio" / "source.wav").resolve()
    source_audio_path.parent.mkdir(parents=True, exist_ok=True)
    source_audio_path.write_bytes(b"RIFF")
    bootstrap_transcript_path = (
        project_root / "episode_assets" / "bootstrap.transcript.json"
    ).resolve()
    bootstrap_audio_work_dir = (project_root / "episode_assets" / "audio_stage").resolve()
    speaker_samples_output_dir = (
        project_root / "episode_assets" / "speaker_samples"
    ).resolve()
    captured: dict[str, object] = {}

    def fake_run(command: list[str], *, cwd: str, check: bool) -> subprocess.CompletedProcess[str]:
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["check"] = check
        bootstrap_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_transcript_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker": "BOOT_00",
                            "text": "",
                            "start_time": 0,
                            "end_time": 2_000,
                        }
                    ],
                    "metadata": {
                        "speaker_sample_bootstrap_mode": "diarization_only",
                        "diarization_turns": [
                            {
                                "speaker": "BOOT_00",
                                "start_time": 0,
                                "end_time": 2_000,
                            }
                        ],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        speaker_samples_output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = (speaker_samples_output_dir / "speaker_00.wav").resolve()
        artifact_path.write_bytes(b"RIFF")
        manifest_path = (speaker_samples_output_dir / "manifest.json").resolve()
        manifest_path.write_text(
            json.dumps(
                {
                    "generated_at_utc": "2026-03-20T00:00:00+00:00",
                    "source_audio_path": str(source_audio_path),
                    "samples": [
                        {
                            "speaker": "BOOT_00",
                            "path": str(artifact_path),
                            "requested_duration_ms": 30_000,
                            "duration_ms": 2_000,
                            "status": "shorter_than_requested",
                            "dropped_slice_count": 0,
                            "ranges_ms": [
                                {
                                    "start_time_ms": 0,
                                    "end_time_ms": 2_000,
                                }
                            ],
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.supports_nemo_backend",
        lambda runtime_platform: False,
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.ensure_command_available",
        lambda command_name: None,
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "card_framework.audio_pipeline.calibration.build_source_separator",
        lambda audio_cfg: (_ for _ in ()).throw(
            AssertionError("local source separation should not run in Docker mode")
        ),
    )

    result = bootstrap_speaker_samples_from_diarization(
        project_root=project_root,
        audio_cfg={
            "device": "cuda",
            "diarization": {"provider": "nemo"},
            "speaker_samples": {"source_audio": "vocals"},
        },
        audio_path=source_audio_path,
        bootstrap_transcript_path=bootstrap_transcript_path,
        bootstrap_audio_work_dir=bootstrap_audio_work_dir,
        speaker_samples_output_dir=speaker_samples_output_dir,
    )

    docker_command = captured["command"]
    assert isinstance(docker_command, list)
    assert docker_command[:4] == ["docker", "compose", "--profile", "gpu"]
    assert "setup-and-run-gpu" in docker_command
    assert "card_framework.audio_pipeline.bootstrap_speaker_samples_cli" in docker_command
    assert "CARD_FRAMEWORK_DISABLE_DOCKER_SPEAKER_BOOTSTRAP=1" in docker_command
    assert captured["cwd"] == str(project_root)
    assert captured["check"] is False

    docker_cfg_path = (bootstrap_audio_work_dir / "docker_audio_cfg.json").resolve()
    assert docker_cfg_path.exists()
    docker_cfg_payload = json.loads(docker_cfg_path.read_text(encoding="utf-8"))
    assert docker_cfg_payload["device"] == "cuda"

    assert result.transcript_path == bootstrap_transcript_path
    assert result.generation_result.manifest_path == (
        speaker_samples_output_dir / "manifest.json"
    ).resolve()
    assert len(result.generation_result.artifacts) == 1
    assert result.generation_result.artifacts[0].speaker == "BOOT_00"
