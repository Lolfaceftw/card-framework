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
    assert (
        f"audio.eta.store_path={output_root.joinpath('audio_stage', 'eta_history.sqlite3').resolve().as_posix()}"
        in overrides
    )
    assert f"audio.output_transcript_path={output_root.joinpath('transcript.json').resolve().as_posix()}" in overrides
    assert f"transcript_path={output_root.joinpath('transcript.json').resolve().as_posix()}" in overrides
    assert f"audio.voice_clone.runner_project_dir={runner_dir.resolve().as_posix()}" in overrides
    assert f"audio.voice_clone.cfg_path={checkpoints_dir.joinpath('config.yaml').resolve().as_posix()}" in overrides
    assert f"audio.voice_clone.model_dir={checkpoints_dir.resolve().as_posix()}" in overrides


def test_build_run_overrides_uses_shared_eta_store_for_workspace_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Keep ETA history in one workspace-level store even when work dirs are per-run."""
    runner_dir = tmp_path / "runtime_vendor"
    checkpoints_dir = tmp_path / "runtime_checkpoints"

    monkeypatch.setattr(setup_and_run, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(setup_and_run, "OUTPUT_ROOT", None)
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

    assert (
        f"audio.work_dir={tmp_path.joinpath('artifacts', 'audio_stage', 'runs', 'run123').resolve().as_posix()}"
        in overrides
    )
    assert (
        f"audio.eta.store_path={tmp_path.joinpath('artifacts', 'audio_stage', 'eta_history.sqlite3').resolve().as_posix()}"
        in overrides
    )


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


def test_main_setup_only_reports_timed_loading_status(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    """Emit timed setup status lines for long bootstrap phases."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup_and_run, "normalize_cli_overrides", lambda _: ())
    monkeypatch.setattr(
        setup_and_run,
        "resolve_bootstrap_start_stage",
        lambda **_: "stage-1",
    )

    def _fake_resolve_repo_backed_boolean_override(
        overrides,
        *,
        key: str,
        repo_key_path,
        fallback: bool,
    ) -> bool:
        del overrides, repo_key_path, fallback
        values = {
            "audio.voice_clone.enabled": True,
            "audio.voice_clone.live_drafting.enabled": False,
            "audio.interjector.enabled": False,
            "audio.speaker_samples.enabled": True,
        }
        return values[key]

    monkeypatch.setattr(
        setup_and_run,
        "resolve_repo_backed_boolean_override",
        _fake_resolve_repo_backed_boolean_override,
    )
    monkeypatch.setattr(setup_and_run, "check_prerequisites", lambda **_: None)
    monkeypatch.setattr(
        setup_and_run,
        "ensure_indextts_repo",
        lambda **_: setup_and_run.RepoSyncResult(
            cloned=False,
            updated=False,
            pull_skipped_dirty=False,
            lfs_pulled=False,
        ),
    )
    monkeypatch.setattr(setup_and_run, "is_repo_checkout_runtime", lambda: True)
    monkeypatch.setattr(
        setup_and_run,
        "smart_sync_projects",
        lambda **_: ("card-framework",),
    )
    monkeypatch.setattr(
        setup_and_run,
        "ensure_indextts_model",
        lambda **_: setup_and_run.ModelProvisionResult(
            downloaded=False,
            source="cached",
        ),
    )
    monkeypatch.setattr(setup_and_run, "print_summary", lambda **_: None)

    exit_code = setup_and_run.main(["--setup-only"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Loading vendored IndexTTS runtime..." in captured.err
    assert "Loaded vendored IndexTTS runtime in" in captured.err
    assert "Loading runtime dependency sync..." in captured.err
    assert "Loaded runtime dependency sync in" in captured.err
    assert "Loading IndexTTS model provisioning..." in captured.err
    assert "Loaded IndexTTS model provisioning in" in captured.err
