"""Tests for packaged-safe setup-and-run path handling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import card_framework.cli.setup_and_run as setup_and_run
from card_framework.runtime.pipeline_sessions import PipelineSessionStore
from card_framework.runtime.pipeline_sessions import PipelineResumePlan


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
    assert "session.enabled=true" in overrides
    assert "session.session_id=run123" in overrides
    assert (
        "session.summary_output_path="
        f"{output_root.joinpath('artifacts', 'pipeline_sessions', 'run123', 'summary.xml').resolve().as_posix()}"
        in overrides
    )


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
    assert (
        "session.store_path="
        f"{tmp_path.joinpath('artifacts', 'pipeline_sessions', 'sessions.sqlite3').resolve().as_posix()}"
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

    def _fake_subprocess_run(command, cwd=None, check=False, text=False):
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["check"] = check
        recorded["text"] = text
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(setup_and_run.subprocess, "run", _fake_subprocess_run)

    setup_and_run.run_pipeline(uv_executable="uv", overrides=["orchestrator.target_seconds=180"])

    assert recorded["cwd"] == str(output_root.resolve())
    assert recorded["check"] is False
    assert recorded["text"] is True
    command = recorded["command"]
    assert isinstance(command, list)
    assert command[:3] == [sys.executable, "-m", setup_and_run.MAIN_MODULE]
    assert "--config-path" in command
    assert "--config-name" in command
    assert "hydra.run.dir=." in command
    assert "hydra.output_subdir=null" in command
    assert "orchestrator.target_seconds=180" in command


def test_run_pipeline_raises_keyboard_interrupt_for_exit_code_130(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("audio:\n  voice_clone:\n    enabled: false\n", encoding="utf-8")

    monkeypatch.setattr(setup_and_run, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(setup_and_run, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(setup_and_run, "CONFIG_FILE_PATH", config_path)
    monkeypatch.setattr(setup_and_run, "resolve_root_project_dir", lambda: None)
    monkeypatch.setattr(
        setup_and_run.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=130),
    )

    try:
        setup_and_run.run_pipeline(uv_executable="uv", overrides=[])
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("Expected KeyboardInterrupt for child exit code 130.")


def test_build_resume_overrides_retargets_stage_three_inputs(tmp_path: Path) -> None:
    plan = PipelineResumePlan(
        session_id="session-123",
        status="paused",
        requested_start_stage="stage-1",
        resume_stage="stage-3",
        current_stage="stage-3",
        prepared_overrides=(
            "pipeline.start_stage=stage-1",
            "transcript_path=/tmp/original.transcript.json",
            "session.enabled=true",
            "session.session_id=session-123",
        ),
        artifacts={
            "transcript_path": str(tmp_path / "session.transcript.json"),
            "session_summary_path": str(tmp_path / "summary.xml"),
            "voice_clone_manifest_path": str(tmp_path / "voice_clone" / "manifest.json"),
            "interjector_manifest_path": str(
                tmp_path / "interjector" / "interjector_manifest.json"
            ),
        },
        options={"voice_clone_enabled": True, "interjector_enabled": False},
        stage_validity={"stage-1": True, "stage-2": True, "stage-3": False, "stage-4": False},
        last_error="",
        last_exit_reason="paused",
    )

    overrides = setup_and_run.build_resume_overrides(plan)

    assert "pipeline.start_stage=stage-3" in overrides
    assert f"transcript_path={plan.artifacts['transcript_path'].replace(chr(92), '/')}" in overrides
    assert (
        f"pipeline.final_summary_path={plan.artifacts['session_summary_path'].replace(chr(92), '/')}"
        in overrides
    )


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
    monkeypatch.setattr(setup_and_run, "run_calibration", lambda **_: None)
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


def test_main_resume_cleanup_removes_stale_stage_outputs_before_rerun(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Clear stale stage outputs before resuming from the saved checkpoint boundary."""
    monkeypatch.chdir(tmp_path)
    session_id = "session-resume"
    artifacts = {
        "work_dir": str(tmp_path / "artifacts" / "audio_stage" / "runs" / session_id),
        "transcript_path": str(
            tmp_path / "artifacts" / "transcripts" / f"{session_id}.transcript.json"
        ),
        "workspace_summary_path": str(tmp_path / "summary.xml"),
        "session_summary_path": str(
            tmp_path / "artifacts" / "pipeline_sessions" / session_id / "summary.xml"
        ),
        "voice_clone_manifest_path": str(
            tmp_path
            / "artifacts"
            / "audio_stage"
            / "runs"
            / session_id
            / "voice_clone"
            / "manifest.json"
        ),
        "interjector_manifest_path": str(
            tmp_path
            / "artifacts"
            / "audio_stage"
            / "runs"
            / session_id
            / "interjector"
            / "interjector_manifest.json"
        ),
    }
    transcript_path = Path(artifacts["transcript_path"])
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        (
            '{"segments": [{"speaker": "SPEAKER_00", "start_time": 0, '
            '"end_time": 1000, "text": "hello"}], "metadata": {}}\n'
        ),
        encoding="utf-8",
    )
    session_summary_path = Path(artifacts["session_summary_path"])
    session_summary_path.parent.mkdir(parents=True, exist_ok=True)
    session_summary_path.write_text(
        "<SPEAKER_00>Hello</SPEAKER_00>\n",
        encoding="utf-8",
    )
    voice_clone_dir = Path(artifacts["voice_clone_manifest_path"]).parent
    interjector_dir = Path(artifacts["interjector_manifest_path"]).parent
    voice_clone_dir.mkdir(parents=True, exist_ok=True)
    interjector_dir.mkdir(parents=True, exist_ok=True)
    (voice_clone_dir / "partial.wav").write_bytes(b"RIFF")
    (interjector_dir / "partial.wav").write_bytes(b"RIFF")
    session_store = PipelineSessionStore(
        tmp_path / "artifacts" / "pipeline_sessions" / "sessions.sqlite3"
    )
    session_store.create_or_replace_session(
        session_id=session_id,
        requested_start_stage="stage-1",
        prepared_overrides=["pipeline.start_stage=stage-1"],
        artifacts=artifacts,
        options={
            "voice_clone_enabled": True,
            "interjector_enabled": False,
        },
    )

    monkeypatch.setattr(setup_and_run, "normalize_cli_overrides", lambda _: ())
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
    monkeypatch.setattr(setup_and_run, "run_pipeline", lambda **_: None)
    monkeypatch.setattr(setup_and_run, "print_summary", lambda **_: None)

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
            "audio.speaker_samples.enabled": False,
        }
        return values[key]

    monkeypatch.setattr(
        setup_and_run,
        "resolve_repo_backed_boolean_override",
        _fake_resolve_repo_backed_boolean_override,
    )

    exit_code = setup_and_run.main(["--resume-session", session_id])

    assert exit_code == 0
    assert transcript_path.exists()
    assert session_summary_path.exists()
    assert not voice_clone_dir.exists()
    assert not interjector_dir.exists()


def test_main_preserves_child_pipeline_failure_reason(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Keep the detailed failure saved by the child runtime."""
    monkeypatch.chdir(tmp_path)
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        (
            '{"segments": [{"speaker": "SPEAKER_00", "start_time": 0, '
            '"end_time": 1000, "text": "hello"}], "metadata": {}}\n'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(setup_and_run, "uuid4", lambda: "session-child-failure")
    monkeypatch.setattr(setup_and_run, "normalize_cli_overrides", lambda _: ())
    monkeypatch.setattr(
        setup_and_run,
        "resolve_bootstrap_start_stage",
        lambda **_: "stage-2",
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
    monkeypatch.setattr(setup_and_run, "run_calibration", lambda **_: None)
    monkeypatch.setattr(setup_and_run, "print_summary", lambda **_: None)

    def _fake_resolve_repo_backed_boolean_override(
        overrides,
        *,
        key: str,
        repo_key_path,
        fallback: bool,
    ) -> bool:
        del overrides, repo_key_path, fallback
        values = {
            "audio.voice_clone.enabled": False,
            "audio.voice_clone.live_drafting.enabled": False,
            "audio.interjector.enabled": False,
            "audio.speaker_samples.enabled": False,
        }
        return values[key]

    monkeypatch.setattr(
        setup_and_run,
        "resolve_repo_backed_boolean_override",
        _fake_resolve_repo_backed_boolean_override,
    )

    def _fake_run_pipeline(*, uv_executable: str, overrides) -> None:
        del uv_executable, overrides
        store = PipelineSessionStore(
            setup_and_run.resolve_pipeline_session_store_path()
        )
        store.mark_session_failed(
            session_id="session-child-failure",
            current_stage=None,
            reason="Child runtime failed during stage-2.",
        )
        raise setup_and_run.BootstrapError(
            step="pipeline_run",
            message="Command failed with exit code 1.",
        )

    monkeypatch.setattr(setup_and_run, "run_pipeline", _fake_run_pipeline)

    exit_code = setup_and_run.main([])

    record = PipelineSessionStore(
        setup_and_run.resolve_pipeline_session_store_path()
    ).get_session("session-child-failure")
    assert exit_code == 1
    assert record is not None
    assert record.status == "failed"
    assert record.last_error == "Child runtime failed during stage-2."
