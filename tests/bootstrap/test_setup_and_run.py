from __future__ import annotations

import codecs
import json
from pathlib import Path
import subprocess

import pytest

import setup_and_run as bootstrap


def _write_project_files(project_dir: Path) -> None:
    """Create minimal lock and project files used by smart-sync tests."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (project_dir / "uv.lock").write_text("version = 1\n", encoding="utf-8")


def test_check_prerequisites_raises_when_required_command_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_which(command_name: str) -> str | None:
        if command_name == "ffmpeg":
            return None
        return f"C:/tools/{command_name}"

    monkeypatch.setattr(bootstrap.shutil, "which", _fake_which)
    monkeypatch.setattr(
        bootstrap,
        "run_cmd",
        lambda **kwargs: subprocess.CompletedProcess(
            args=kwargs["command"], returncode=0, stdout="", stderr=""
        ),
    )

    with pytest.raises(bootstrap.BootstrapError, match="ffmpeg"):
        bootstrap.check_prerequisites(git_executable="git")


def test_checkpoints_ready_requires_weight_file(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "config.yaml").write_text("version: test\n", encoding="utf-8")
    (checkpoints_dir / "pinyin.vocab").write_text("a\n", encoding="utf-8")

    ready, reason = bootstrap.checkpoints_ready(checkpoints_dir)
    assert ready is False
    assert "No model weight files" in reason

    (checkpoints_dir / "model.safetensors").write_bytes(b"x" * bootstrap.MIN_WEIGHT_BYTES)
    ready_after, _reason_after = bootstrap.checkpoints_ready(checkpoints_dir)
    assert ready_after is True


def test_smart_sync_projects_skips_when_fingerprints_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    _write_project_files(repo_root)
    _write_project_files(index_tts_dir)
    (repo_root / ".venv").mkdir(parents=True, exist_ok=True)
    (index_tts_dir / ".venv").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)
    monkeypatch.setattr(
        bootstrap,
        "SETUP_STATE_PATH",
        repo_root / "artifacts" / "bootstrap" / "setup_state.json",
    )

    captured: list[list[str]] = []

    def _fake_run_cmd(
        *,
        step: str,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del step, cwd, env
        captured.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bootstrap, "run_cmd", _fake_run_cmd)

    first_synced = bootstrap.smart_sync_projects(uv_executable="uv", force_sync=False)
    assert first_synced == ("root", "index_tts")
    assert len(captured) == 2

    captured.clear()
    second_synced = bootstrap.smart_sync_projects(uv_executable="uv", force_sync=False)
    assert second_synced == ()
    assert captured == []


def test_smart_sync_projects_can_skip_nested_indextts_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    _write_project_files(repo_root)
    (repo_root / ".venv").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)
    monkeypatch.setattr(
        bootstrap,
        "SETUP_STATE_PATH",
        repo_root / "artifacts" / "bootstrap" / "setup_state.json",
    )

    captured: list[list[str]] = []

    def _fake_run_cmd(
        *,
        step: str,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del step, env
        captured.append(command)
        assert cwd == repo_root
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bootstrap, "run_cmd", _fake_run_cmd)

    synced = bootstrap.smart_sync_projects(
        uv_executable="uv",
        force_sync=False,
        include_index_tts=False,
    )

    assert synced == ("root",)
    assert captured == [["uv", "sync", "--locked"]]


def test_ensure_indextts_repo_skips_pull_when_git_tree_dirty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    index_tts_dir.mkdir(parents=True, exist_ok=True)
    (index_tts_dir / ".git").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)

    commands: list[list[str]] = []

    def _fake_run_cmd(
        *,
        step: str,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del step, cwd, env
        commands.append(command)
        if command[1:] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=" M changed.py\n",
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bootstrap, "run_cmd", _fake_run_cmd)

    result = bootstrap.ensure_indextts_repo(git_executable="git", skip_update=False)
    assert result.pull_skipped_dirty is True
    assert not any(command[:3] == ["git", "pull", "--ff-only"] for command in commands)
    assert any(command[:3] == ["git", "lfs", "pull"] for command in commands)


def test_ensure_indextts_model_falls_back_to_modelscope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    checkpoints_dir = index_tts_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "config.yaml").write_text("version: test\n", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)

    commands: list[list[str]] = []

    def _fake_run_cmd(
        *,
        step: str,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        commands.append(command)
        if "hf" in command:
            raise bootstrap.BootstrapError(
                step=step,
                message="hf failed",
                command=tuple(command),
                stderr_tail="simulated",
            )
        if "modelscope" in command:
            (checkpoints_dir / "model.safetensors").write_bytes(
                b"x" * bootstrap.MIN_WEIGHT_BYTES
            )
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bootstrap, "run_cmd", _fake_run_cmd)

    result = bootstrap.ensure_indextts_model(uv_executable="uv", force_download=False)
    assert result.downloaded is True
    assert result.source == "modelscope"
    assert any("hf" in command for command in commands)
    assert any("modelscope" in command for command in commands)


def test_build_run_overrides_include_required_voice_clone_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    index_tts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)

    audio_path = repo_root / "input audio.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"wav")

    overrides = bootstrap.build_run_overrides(audio_path=audio_path, run_id="20260302_120000")
    override_map = dict(entry.split("=", 1) for entry in overrides)

    assert override_map["pipeline.start_stage"] == "stage-1"
    assert override_map["audio.voice_clone.enabled"] == "true"
    assert override_map["audio.voice_clone.execution_backend"] == "subprocess"
    assert override_map["logging.print_to_terminal"] == "true"
    assert override_map["logging.summarizer_critic_print_to_terminal"] == "false"
    assert override_map["audio.speaker_samples.source_audio"] == "vocals"
    assert override_map["audio.speaker_samples.target_duration_seconds"] == "30"
    assert "\\" not in override_map["audio.audio_path"]
    assert override_map["audio.audio_path"].endswith("input audio.wav")


def test_build_run_overrides_can_omit_audio_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    index_tts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)

    overrides = bootstrap.build_run_overrides(run_id="20260302_120000")
    override_map = dict(entry.split("=", 1) for entry in overrides)

    assert "audio.audio_path" not in override_map
    assert override_map["pipeline.start_stage"] == "stage-1"


def test_build_run_overrides_can_disable_voice_clone_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    index_tts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)

    overrides = bootstrap.build_run_overrides(
        run_id="20260302_120000",
        enable_voice_clone=False,
    )
    override_map = dict(entry.split("=", 1) for entry in overrides)

    assert override_map["audio.voice_clone.enabled"] == "false"
    assert "audio.voice_clone.provider" not in override_map
    assert "audio.voice_clone.execution_backend" not in override_map
    assert "audio.voice_clone.runner_project_dir" not in override_map
    assert "audio.voice_clone.cfg_path" not in override_map
    assert "audio.voice_clone.model_dir" not in override_map


def test_build_run_overrides_can_enable_interjector_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    index_tts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)

    overrides = bootstrap.build_run_overrides(
        run_id="20260302_120000",
        start_stage="stage-4",
        enable_voice_clone=False,
        enable_interjector=True,
    )
    override_map = dict(entry.split("=", 1) for entry in overrides)

    assert override_map["pipeline.start_stage"] == "stage-4"
    assert override_map["audio.interjector.enabled"] == "true"
    assert "transcript_path" not in override_map


def test_build_run_overrides_stage_three_omits_stage_one_transcript_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    index_tts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)

    overrides = bootstrap.build_run_overrides(
        run_id="20260302_120000",
        start_stage="stage-3",
    )
    override_map = dict(entry.split("=", 1) for entry in overrides)

    assert override_map["pipeline.start_stage"] == "stage-3"
    assert "audio.output_transcript_path" not in override_map
    assert "transcript_path" not in override_map


def test_resolve_start_stage_accepts_stage_four() -> None:
    assert (
        bootstrap.resolve_start_stage(["pipeline.start_stage=stage-4"]) == "stage-4"
    )


def test_resolve_bootstrap_start_stage_defaults_to_stage_two_for_transcript_override() -> None:
    assert (
        bootstrap.resolve_bootstrap_start_stage(
            shortcut_overrides=(),
            pass_through_overrides=["transcript_path=transcript.json"],
            cli_audio_path=None,
        )
        == "stage-2"
    )


def test_resolve_bootstrap_start_stage_defaults_to_stage_two_when_root_transcript_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "transcript.json").write_text('{"segments": []}', encoding="utf-8")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    assert (
        bootstrap.resolve_bootstrap_start_stage(
            shortcut_overrides=(),
            pass_through_overrides=(),
            cli_audio_path=None,
        )
        == "stage-2"
    )


def test_resolve_bootstrap_start_stage_keeps_stage_one_when_audio_path_cli_is_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "transcript.json").write_text('{"segments": []}', encoding="utf-8")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    assert (
        bootstrap.resolve_bootstrap_start_stage(
            shortcut_overrides=(),
            pass_through_overrides=(),
            cli_audio_path="audio.wav",
        )
        == "stage-1"
    )


def test_build_start_stage_selection_detail_reports_reused_transcript_and_summary_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    transcript_path = repo_root / "transcript.json"
    transcript_path.write_text('{"segments": []}', encoding="utf-8")
    summary_path = repo_root / "summary.xml"
    summary_path.write_text("<SPEAKER_00>hello</SPEAKER_00>", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    detail = bootstrap.build_start_stage_selection_detail(
        start_stage="stage-2",
        voiceclone_from_summary=None,
        pass_through_overrides=(),
        cli_audio_path=None,
    )

    assert "pipeline.start_stage=stage-2" in detail
    assert str(transcript_path.resolve()) in detail
    assert "--voiceclone-from-summary" in detail
    assert summary_path.resolve().as_posix() in detail


def test_build_start_stage_selection_detail_reports_voiceclone_shortcut(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    summary_path = repo_root / "summary.xml"
    summary_path.write_text("<SPEAKER_00>hello</SPEAKER_00>", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    detail = bootstrap.build_start_stage_selection_detail(
        start_stage="stage-3",
        voiceclone_from_summary="summary.xml",
        pass_through_overrides=(),
        cli_audio_path=None,
    )

    assert "pipeline.start_stage=stage-3" in detail
    assert "--voiceclone-from-summary" in detail
    assert str(summary_path.resolve()) in detail


def test_build_calibration_warning_message_for_stage_two() -> None:
    message = bootstrap.build_calibration_warning_message(start_stage="stage-2")

    assert message is not None
    assert "before summarization" in message


def test_resolve_audio_override_path_supports_relative_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    audio_path = repo_root / "audio.wav"
    audio_path.write_bytes(b"wav")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    resolved = bootstrap.resolve_audio_override_path(
        overrides=["audio.audio_path=audio.wav"]
    )

    assert resolved == audio_path.resolve()


def test_normalize_cli_overrides_rejects_malformed_values() -> None:
    with pytest.raises(bootstrap.BootstrapError, match="Expected KEY=VALUE"):
        bootstrap.normalize_cli_overrides(["pipeline.start_stage"])


def test_build_shortcut_overrides_for_voiceclone_from_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    summary_path = repo_root / "outputs" / "summary.xml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("<SPEAKER_00>hello</SPEAKER_00>", encoding="utf-8")
    manifest_path = (
        repo_root
        / "artifacts"
        / "audio_stage"
        / "runs"
        / "20260302_120000"
        / "speaker_samples"
        / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"samples": []}', encoding="utf-8")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    overrides = bootstrap.build_shortcut_overrides(
        voiceclone_from_summary="outputs/summary.xml",
        run_id="20260302_130000",
    )
    override_map = dict(entry.split("=", 1) for entry in overrides)

    assert override_map["pipeline.start_stage"] == "stage-3"
    assert override_map["pipeline.final_summary_path"].endswith("/outputs/summary.xml")
    assert override_map["audio.speaker_samples.enabled"] == "false"
    assert override_map["audio.voice_clone.enabled"] == "true"
    shortcut_transcript = Path(override_map["transcript_path"])
    assert shortcut_transcript.exists()
    payload = json.loads(shortcut_transcript.read_text(encoding="utf-8"))
    assert payload["metadata"]["speaker_samples_manifest_path"] == manifest_path.as_posix()


def test_build_shortcut_overrides_requires_existing_speaker_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    summary_path = repo_root / "outputs" / "summary.xml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("<SPEAKER_00>hello</SPEAKER_00>", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    with pytest.raises(bootstrap.BootstrapError, match="No speaker-sample manifest"):
        bootstrap.build_shortcut_overrides(
            voiceclone_from_summary="outputs/summary.xml",
            run_id="20260302_130001",
        )


def test_requires_audio_path_input_uses_last_override_wins() -> None:
    assert (
        bootstrap.requires_audio_path_input(
            [
                "pipeline.start_stage=stage-1",
                "pipeline.start_stage=stage-2",
            ]
        )
        is False
    )
    assert bootstrap.requires_audio_path_input(["pipeline.start_stage=stage-1"]) is True


def test_ensure_transcript_override_for_stage_three_creates_synthetic_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    manifest_path = (
        repo_root
        / "artifacts"
        / "audio_stage"
        / "runs"
        / "20260302_120000"
        / "speaker_samples"
        / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"samples": []}', encoding="utf-8")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    overrides = [
        "pipeline.start_stage=stage-3",
        "pipeline.final_summary_path=summary.xml",
    ]
    bootstrap.ensure_transcript_override_for_stage(
        overrides=overrides,
        start_stage="stage-3",
        run_id="20260302_130000",
    )
    override_map = dict(entry.split("=", 1) for entry in overrides)
    transcript_path = Path(override_map["transcript_path"])
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))

    assert transcript_path.exists()
    assert payload["metadata"]["speaker_samples_manifest_path"] == manifest_path.as_posix()


def test_ensure_transcript_override_for_stage_two_prefers_root_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    root_transcript = repo_root / "transcript.json"
    root_transcript.write_text('{"segments": []}', encoding="utf-8")
    artifact_transcript = repo_root / "artifacts" / "transcripts" / "latest.transcript.json"
    artifact_transcript.parent.mkdir(parents=True, exist_ok=True)
    artifact_transcript.write_text('{"segments": [{"speaker": "S"}]}', encoding="utf-8")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    overrides = ["pipeline.start_stage=stage-2"]
    bootstrap.ensure_transcript_override_for_stage(
        overrides=overrides,
        start_stage="stage-2",
        run_id="20260302_130000",
    )

    override_map = dict(entry.split("=", 1) for entry in overrides)
    assert Path(override_map["transcript_path"]) == root_transcript.resolve()


def test_ensure_transcript_override_for_stage_four_does_not_add_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    overrides = [
        "pipeline.start_stage=stage-4",
        "pipeline.final_summary_path=summary.xml",
        "pipeline.voice_clone_manifest_path=voice_clone/manifest.json",
    ]
    bootstrap.ensure_transcript_override_for_stage(
        overrides=overrides,
        start_stage="stage-4",
        run_id="20260302_130000",
    )

    override_map = dict(entry.split("=", 1) for entry in overrides)
    assert "transcript_path" not in override_map


def test_resolve_calibration_transcript_path_skips_stage_one_future_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    resolved = bootstrap.resolve_calibration_transcript_path(
        overrides=[
            "pipeline.start_stage=stage-1",
            "transcript_path=artifacts/transcripts/20260302_130000.transcript.json",
        ],
        start_stage="stage-1",
    )

    assert resolved is None


def test_stage_two_requires_audio_fallback_when_transcript_lacks_vocals_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    transcript_path = repo_root / "transcript.json"
    transcript_path.write_text('{"segments": [], "metadata": {}}', encoding="utf-8")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    assert (
        bootstrap.stage_two_requires_audio_fallback(
            transcript_path=transcript_path,
            speaker_samples_enabled=True,
        )
        is True
    )


def test_stage_two_requires_audio_fallback_skips_when_vocals_metadata_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    vocals_path = repo_root / "vocals.wav"
    vocals_path.write_bytes(b"vocals")
    transcript_path = repo_root / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [],
                "metadata": {"vocals_audio_path": "vocals.wav"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    assert (
        bootstrap.stage_two_requires_audio_fallback(
            transcript_path=transcript_path,
            speaker_samples_enabled=True,
        )
        is False
    )


def test_stage_two_requires_audio_fallback_skips_when_manifest_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    manifest_path = repo_root / "speaker_samples" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"samples": []}', encoding="utf-8")
    transcript_path = repo_root / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [],
                "metadata": {"speaker_samples_manifest_path": "speaker_samples/manifest.json"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    assert (
        bootstrap.stage_two_requires_audio_fallback(
            transcript_path=transcript_path,
            speaker_samples_enabled=True,
        )
        is False
    )


def test_stage_two_requires_audio_fallback_accepts_utf8_bom_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    transcript_path = repo_root / "transcript.json"
    transcript_path.write_bytes(
        codecs.BOM_UTF8 + b'{"segments": [], "metadata": {}}'
    )

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    assert (
        bootstrap.stage_two_requires_audio_fallback(
            transcript_path=transcript_path,
            speaker_samples_enabled=True,
        )
        is True
    )


def test_resolve_audio_input_supports_relative_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    input_audio = repo_root / "audio.wav"
    input_audio.write_bytes(b"wav")

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    resolved = bootstrap.resolve_audio_input("audio.wav")
    assert resolved == input_audio.resolve()


def test_run_pipeline_streams_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def _fake_run_cmd(**kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(dict(kwargs))
        command = kwargs["command"]
        assert isinstance(command, list)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bootstrap, "run_cmd", _fake_run_cmd)
    monkeypatch.setattr(bootstrap, "REPO_ROOT", Path("C:/repo"))

    bootstrap.run_pipeline(
        uv_executable="uv",
        overrides=["pipeline.start_stage=stage-1"],
    )

    assert len(captured) == 1
    call = captured[0]
    assert call["step"] == "pipeline_run"
    assert call["cwd"] == Path("C:/repo")
    assert call["stream_output"] is True
    assert call["command"] == ["uv", "run", "main.py", "pipeline.start_stage=stage-1"]


def test_main_skips_calibration_when_live_draft_audio_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Skip the calibration step when live stage-2 audio drafting is enabled."""
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    repo_root.mkdir(parents=True, exist_ok=True)
    index_tts_dir.mkdir(parents=True, exist_ok=True)

    repo_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []
    calibration_calls: list[dict[str, object]] = []
    pipeline_calls: list[dict[str, object]] = []

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)
    monkeypatch.setattr(bootstrap, "check_prerequisites", lambda **kwargs: None)
    monkeypatch.setattr(
        bootstrap,
        "ensure_indextts_repo",
        lambda **kwargs: repo_calls.append(dict(kwargs))
        or bootstrap.RepoSyncResult(
            cloned=False,
            updated=False,
            pull_skipped_dirty=False,
            lfs_pulled=False,
        ),
    )
    monkeypatch.setattr(bootstrap, "smart_sync_projects", lambda **kwargs: ())
    monkeypatch.setattr(
        bootstrap,
        "ensure_indextts_model",
        lambda **kwargs: model_calls.append(dict(kwargs))
        or bootstrap.ModelProvisionResult(downloaded=False, source="cached"),
    )
    monkeypatch.setattr(bootstrap, "utc_now_compact", lambda: "20260307_000000")
    monkeypatch.setattr(
        bootstrap,
        "build_start_stage_selection_detail",
        lambda **kwargs: "pipeline.start_stage=stage-2",
    )
    monkeypatch.setattr(
        bootstrap,
        "ensure_transcript_override_for_stage",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        bootstrap,
        "resolve_transcript_override_path",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        bootstrap,
        "resolve_audio_override_path",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        bootstrap,
        "stage_two_requires_audio_fallback",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        bootstrap,
        "run_calibration",
        lambda **kwargs: calibration_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        bootstrap,
        "run_pipeline",
        lambda **kwargs: pipeline_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(bootstrap, "print_summary", lambda **kwargs: None)

    result = bootstrap.main(["--override", "pipeline.start_stage=stage-2"])

    assert result == 0
    assert len(repo_calls) == 1
    assert len(model_calls) == 1
    assert calibration_calls == []
    assert len(pipeline_calls) == 1


def test_main_keeps_calibration_runtime_when_live_draft_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Provision IndexTTS and run calibration for the legacy stage-2 estimate path."""
    repo_root = tmp_path / "repo"
    index_tts_dir = repo_root / "third_party" / "index_tts"
    repo_root.mkdir(parents=True, exist_ok=True)
    index_tts_dir.mkdir(parents=True, exist_ok=True)

    repo_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []
    calibration_calls: list[dict[str, object]] = []
    pipeline_calls: list[dict[str, object]] = []

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)
    monkeypatch.setattr(bootstrap, "INDEX_TTS_DIR", index_tts_dir)
    monkeypatch.setattr(bootstrap, "check_prerequisites", lambda **kwargs: None)
    monkeypatch.setattr(
        bootstrap,
        "ensure_indextts_repo",
        lambda **kwargs: repo_calls.append(dict(kwargs))
        or bootstrap.RepoSyncResult(
            cloned=False,
            updated=False,
            pull_skipped_dirty=False,
            lfs_pulled=False,
        ),
    )
    monkeypatch.setattr(bootstrap, "smart_sync_projects", lambda **kwargs: ())
    monkeypatch.setattr(
        bootstrap,
        "ensure_indextts_model",
        lambda **kwargs: model_calls.append(dict(kwargs))
        or bootstrap.ModelProvisionResult(downloaded=False, source="cached"),
    )
    monkeypatch.setattr(bootstrap, "utc_now_compact", lambda: "20260307_000000")
    monkeypatch.setattr(
        bootstrap,
        "build_start_stage_selection_detail",
        lambda **kwargs: "pipeline.start_stage=stage-2",
    )
    monkeypatch.setattr(
        bootstrap,
        "ensure_transcript_override_for_stage",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        bootstrap,
        "resolve_transcript_override_path",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        bootstrap,
        "resolve_audio_override_path",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        bootstrap,
        "stage_two_requires_audio_fallback",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        bootstrap,
        "run_calibration",
        lambda **kwargs: calibration_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        bootstrap,
        "run_pipeline",
        lambda **kwargs: pipeline_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(bootstrap, "print_summary", lambda **kwargs: None)

    result = bootstrap.main(
        [
            "--override",
            "pipeline.start_stage=stage-2",
            "--override",
            "audio.voice_clone.enabled=false",
            "--override",
            "audio.voice_clone.live_drafting.enabled=false",
        ]
    )

    assert result == 0
    assert len(repo_calls) == 1
    assert len(model_calls) == 1
    assert len(calibration_calls) == 1
    assert len(pipeline_calls) == 1
