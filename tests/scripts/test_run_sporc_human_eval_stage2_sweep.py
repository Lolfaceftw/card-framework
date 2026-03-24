"""Coverage for the SPoRC human-eval stage-2 sweep driver."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from card_framework.shared.paths import REPO_ROOT


def _load_module():
    script_path = (REPO_ROOT / "scripts" / "run_sporc_human_eval_stage2_sweep.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "run_sporc_human_eval_stage2_sweep",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_to_container_path_uses_posix_workspace_root() -> None:
    """Container-bound paths should stay Linux-style even on Windows hosts."""
    module = _load_module()
    host_path = (REPO_ROOT / "artifacts" / "tmp" / "transcript.json").resolve()

    container_path = module._to_container_path(host_path)

    assert container_path == "/workspace/card-framework/artifacts/tmp/transcript.json"
    assert "\\" not in container_path
    assert not container_path.startswith("C:")


def test_machine_episode_label_avoids_spaces_for_hydra_paths() -> None:
    """Internal run directories should avoid Hydra-hostile spaces and punctuation."""
    module = _load_module()

    label = module._machine_episode_label(
        1,
        "The Lion, the Lamb & the Meadow (Chap. 12-13)",
    )

    assert label == "01_the_lion_the_lamb_the_meadow_chap_12_13"
    assert " " not in label
    assert "(" not in label
    assert ")" not in label


def test_sanitize_console_text_replaces_unencodable_characters() -> None:
    """Console tee output should survive cp1252-only terminals."""
    module = _load_module()
    original_stdout = module.sys.stdout
    module.sys.stdout = SimpleNamespace(encoding="cp1252")
    try:
        sanitized = module._sanitize_console_text("Quote \u2014 arrow \U0001F449")
    finally:
        module.sys.stdout = original_stdout

    assert sanitized == 'Quote - arrow ?'


def test_build_run_command_appends_quota_control_overrides() -> None:
    """Optional loop-budget overrides should be forwarded to setup_and_run."""
    module = _load_module()

    command = module._build_run_command(
        output_root=(REPO_ROOT / "artifacts" / "tmp" / "quota-test").resolve(),
        transcript_path=(REPO_ROOT / "artifacts" / "tmp" / "transcript.json").resolve(),
        target_seconds=300,
        llm_profile="deepseek_chat",
        llm_model="deepseek-chat",
        orchestrator_max_iterations=2,
        summarizer_max_tool_turns=8,
        critic_max_tool_turns=3,
        disable_interjector_tool_call_planning=True,
    )

    joined = " ".join(command)
    assert "selection.llm=deepseek_chat" in joined
    assert "selection.stage_llm.summarizer=deepseek_chat" in joined
    assert "profiles.llm.deepseek_chat.model=deepseek-chat" in joined
    assert "orchestrator.max_iterations=2" in joined
    assert "agents.summarizer.max_tool_turns=8" in joined
    assert "agents.critic.max_tool_turns=3" in joined
    assert "audio.interjector.tool_call_planning_enabled=false" in joined


def test_source_separate_episode_voice_samples_rewrites_manifest_to_cleaned_wavs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    voice_samples_dir = (tmp_path / "voice_samples").resolve()
    voice_samples_dir.mkdir(parents=True, exist_ok=True)
    (voice_samples_dir / "SPEAKER_00.wav").write_bytes(b"RIFF00")
    (voice_samples_dir / "SPEAKER_01.wav").write_bytes(b"RIFF11")
    episode = module.ParsedEpisode(
        order=1,
        title="Episode Alpha",
        sample_id="sample-001",
        audio_slice_path=(tmp_path / "episode.mp3").resolve(),
        transcript_blocks=(),
        speakers=("SPEAKER_00", "SPEAKER_01"),
        voice_samples_dir=voice_samples_dir,
        original_voice_manifest={
            "samples": [
                {"speaker": "SPEAKER_00", "path": "/workspace/card-framework/SPEAKER_00.wav"},
                {"speaker": "SPEAKER_01", "path": "/workspace/card-framework/SPEAKER_01.wav"},
            ]
        },
        bootstrap_mapping={},
    )

    class _FakeSeparator:
        def separate_vocals(
            self,
            input_audio_path: Path,
            output_dir: Path,
            *,
            device: str,
            progress_callback=None,
        ) -> Path:
            del device, progress_callback
            vocals_path = (output_dir / "htdemucs" / input_audio_path.stem / "vocals.wav").resolve()
            vocals_path.parent.mkdir(parents=True, exist_ok=True)
            vocals_path.write_bytes(input_audio_path.read_bytes() + b"_clean")
            return vocals_path

    monkeypatch.setattr(module, "load_audio_config", lambda project_root: {"separation": {}})
    monkeypatch.setattr(module, "build_source_separator", lambda audio_cfg: _FakeSeparator())
    monkeypatch.setattr(module, "resolve_device", lambda raw: raw)

    cleaned = module._source_separate_episode_voice_samples(
        episodes=(episode,),
        cleaned_root=(tmp_path / "cleaned").resolve(),
        device="cpu",
    )

    assert len(cleaned) == 1
    cleaned_episode = cleaned[0]
    manifest_path = cleaned_episode.voice_samples_dir / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["cleaned_voice_samples"] is True
    assert manifest_payload["source_sample_cleaning_mode"] == (
        "demucs_vocals_from_existing_voice_samples"
    )
    assert [sample["path"] for sample in manifest_payload["samples"]] == [
        "SPEAKER_00.wav",
        "SPEAKER_01.wav",
    ]
    assert (cleaned_episode.voice_samples_dir / "SPEAKER_00.wav").read_bytes().endswith(
        b"_clean"
    )
    assert (cleaned_episode.voice_samples_dir / "SPEAKER_01.wav").read_bytes().endswith(
        b"_clean"
    )


def test_run_docker_command_monitored_writes_args_json_and_invokes_wrapper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    recorded: dict[str, object] = {}

    def fake_run(command, *, cwd, check=False):
        del check
        recorded["command"] = command
        recorded["cwd"] = cwd
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    status_path = (tmp_path / "run_status.json").resolve()
    return_code = module._run_docker_command_monitored(
        command=["docker", "compose", "--profile", "gpu", "run"],
        stdout_path=(tmp_path / "stdout.log").resolve(),
        stderr_path=(tmp_path / "stderr.log").resolve(),
        monitor_path=(tmp_path / "resource_monitor.jsonl").resolve(),
        status_path=status_path,
        sample_interval_seconds=2.5,
    )

    args_json_path = status_path.with_name(f"{status_path.stem}.args.json")
    assert return_code == 7
    assert json.loads(args_json_path.read_text(encoding="utf-8")) == [
        "compose",
        "--profile",
        "gpu",
        "run",
    ]
    assert recorded["cwd"] == str(REPO_ROOT)
    assert "run_docker_monitored.py" in " ".join(recorded["command"])


def test_attempt_live_draft_recovery_handles_max_iteration_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    preset_root = (tmp_path / "run").resolve()
    voice_clone_dir = (preset_root / "audio_stage" / "voice_clone").resolve()
    voice_clone_dir.mkdir(parents=True, exist_ok=True)
    state_path = (voice_clone_dir / "live_draft_test.state.json").resolve()
    state_path.write_text("{}", encoding="utf-8")
    stdout_path = (tmp_path / "stdout.log").resolve()
    stdout_path.write_text("ERROR Max iterations (3) reached.\n", encoding="utf-8")
    invoked: dict[str, object] = {}

    def fake_run_with_tee(*, command, log_path):
        invoked["stage3_command"] = command
        invoked["stage3_log_path"] = log_path
        return 0

    def fake_run_docker_command_monitored(
        *,
        command,
        stdout_path,
        stderr_path,
        monitor_path,
        status_path,
        sample_interval_seconds=5.0,
    ):
        del sample_interval_seconds
        invoked["stage4_command"] = command
        invoked["stage4_stdout_path"] = stdout_path
        invoked["stage4_stderr_path"] = stderr_path
        invoked["stage4_monitor_path"] = monitor_path
        invoked["stage4_status_path"] = status_path
        return 0

    monkeypatch.setattr(module, "_run_with_tee", fake_run_with_tee)
    monkeypatch.setattr(
        module,
        "_run_docker_command_monitored",
        fake_run_docker_command_monitored,
    )
    monkeypatch.setattr(module, "_to_container_path", lambda path: f"/workspace/{Path(path).name}")

    result = module._attempt_live_draft_recovery(
        preset_root=preset_root,
        stdout_path=stdout_path,
        logs_root=(tmp_path / "logs").resolve(),
        log_stem="sample_30s",
        llm_profile="deepseek_chat",
        llm_model="deepseek-chat",
    )

    assert result is not None
    assert result["recovered_from_max_iterations"] == "true"
    assert result["recovered_from_reason"] == "max_iterations"
    assert result["live_draft_state_path"] == str(state_path)
    assert "recover_sporc_live_draft_stage3.py" in " ".join(invoked["stage3_command"])
    assert "pipeline.start_stage=stage-4" in " ".join(invoked["stage4_command"])


def test_attempt_live_draft_recovery_handles_context_window_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    preset_root = (tmp_path / "run").resolve()
    voice_clone_dir = (preset_root / "audio_stage" / "voice_clone").resolve()
    voice_clone_dir.mkdir(parents=True, exist_ok=True)
    state_path = (voice_clone_dir / "live_draft_test.state.json").resolve()
    state_path.write_text("{}", encoding="utf-8")
    stdout_path = (tmp_path / "stdout.log").resolve()
    stdout_path.write_text(
        "maximum context length is 131072 tokens\n"
        "Please reduce the length of the messages or completion.\n",
        encoding="utf-8",
    )
    invoked: dict[str, object] = {}

    def fake_run_with_tee(*, command, log_path):
        invoked["stage3_command"] = command
        invoked["stage3_log_path"] = log_path
        return 0

    def fake_run_docker_command_monitored(
        *,
        command,
        stdout_path,
        stderr_path,
        monitor_path,
        status_path,
        sample_interval_seconds=5.0,
    ):
        del sample_interval_seconds
        invoked["stage4_command"] = command
        invoked["stage4_stdout_path"] = stdout_path
        invoked["stage4_stderr_path"] = stderr_path
        invoked["stage4_monitor_path"] = monitor_path
        invoked["stage4_status_path"] = status_path
        return 0

    monkeypatch.setattr(module, "_run_with_tee", fake_run_with_tee)
    monkeypatch.setattr(
        module,
        "_run_docker_command_monitored",
        fake_run_docker_command_monitored,
    )
    monkeypatch.setattr(module, "_to_container_path", lambda path: f"/workspace/{Path(path).name}")

    result = module._attempt_live_draft_recovery(
        preset_root=preset_root,
        stdout_path=stdout_path,
        logs_root=(tmp_path / "logs").resolve(),
        log_stem="sample_5m",
        llm_profile="deepseek_chat",
        llm_model="deepseek-chat",
    )

    assert result is not None
    assert result["recovered_from_max_iterations"] == "false"
    assert result["recovered_from_reason"] == "context_window"
    assert result["live_draft_state_path"] == str(state_path)
    assert "recover_sporc_live_draft_stage3.py" in " ".join(invoked["stage3_command"])
    assert "pipeline.start_stage=stage-4" in " ".join(invoked["stage4_command"])
