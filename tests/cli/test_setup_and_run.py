"""Tests for packaged-safe setup-and-run path handling."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import card_framework.cli.setup_and_run as setup_and_run


def test_build_run_overrides_uses_explicit_output_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Build flat infer-style artifact overrides when an output root is provided."""
    output_root = tmp_path / "outputs"
    runner_dir = tmp_path / "runtime_vendor"
    checkpoints_dir = tmp_path / "runtime_checkpoints"

    monkeypatch.setattr(setup_and_run, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(setup_and_run, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        setup_and_run,
        "resolve_index_tts_runner_project_dir",
        lambda: runner_dir,
    )
    monkeypatch.setattr(
        setup_and_run,
        "resolve_index_tts_checkpoints_dir",
        lambda: checkpoints_dir,
    )

    overrides = setup_and_run.build_run_overrides(run_id="run123")

    assert f"audio.work_dir={output_root.joinpath('audio_stage').resolve().as_posix()}" in overrides
    assert f"audio.output_transcript_path={output_root.joinpath('transcript.json').resolve().as_posix()}" in overrides
    assert f"transcript_path={output_root.joinpath('transcript.json').resolve().as_posix()}" in overrides
    assert f"audio.voice_clone.runner_project_dir={runner_dir.resolve().as_posix()}" in overrides
    assert f"audio.voice_clone.cfg_path={checkpoints_dir.joinpath('config.yaml').resolve().as_posix()}" in overrides
    assert f"audio.voice_clone.model_dir={checkpoints_dir.resolve().as_posix()}" in overrides


def test_resolve_audio_input_uses_workspace_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Resolve relative audio input paths against the caller workspace root."""
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    monkeypatch.setattr(setup_and_run, "WORKSPACE_ROOT", tmp_path)

    resolved = setup_and_run.resolve_audio_input("audio.wav")

    assert resolved == audio_path.resolve()


def test_resolve_transcript_metadata_path_uses_workspace_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Resolve transcript metadata paths relative to the workspace root."""
    manifest_path = tmp_path / "artifacts" / "audio_stage" / "runs" / "r1" / "speaker_samples" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"artifacts": []}\n', encoding="utf-8")
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [],
                "metadata": {
                    "speaker_samples_manifest_path": "artifacts/audio_stage/runs/r1/speaker_samples/manifest.json",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(setup_and_run, "WORKSPACE_ROOT", tmp_path)

    resolved = setup_and_run.resolve_transcript_metadata_path(
        transcript_path=transcript_path,
        metadata_key="speaker_samples_manifest_path",
    )

    assert resolved == manifest_path.resolve()


def test_run_pipeline_uses_output_root_for_packaged_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Run the child main module from the explicit output root in packaged mode."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("audio:\n  voice_clone:\n    enabled: false\n", encoding="utf-8")
    output_root = tmp_path / "outputs"

    monkeypatch.setattr(setup_and_run, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(setup_and_run, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(setup_and_run, "CONFIG_FILE_PATH", config_path)
    monkeypatch.setattr(setup_and_run, "resolve_root_project_dir", lambda: None)

    recorded: dict[str, object] = {}

    def _fake_run_cmd(*, step: str, command: list[str], cwd: Path | None = None, env=None, stream_output: bool = False):
        del env
        recorded["step"] = step
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["stream_output"] = stream_output
        return None

    monkeypatch.setattr(setup_and_run, "run_cmd", _fake_run_cmd)

    setup_and_run.run_pipeline(uv_executable="uv", overrides=["orchestrator.target_seconds=180"])

    assert recorded["step"] == "pipeline_run"
    assert recorded["cwd"] == output_root.resolve()
    assert recorded["stream_output"] is True
    command = recorded["command"]
    assert isinstance(command, list)
    assert command[:3] == [sys.executable, "-m", setup_and_run.MAIN_MODULE]
    assert "--config-path" in command
    assert "--config-name" in command
    assert "hydra.run.dir=." in command
    assert "hydra.output_subdir=null" in command
    assert "orchestrator.target_seconds=180" in command
