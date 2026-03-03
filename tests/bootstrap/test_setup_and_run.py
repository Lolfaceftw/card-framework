from __future__ import annotations

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
