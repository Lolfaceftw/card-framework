"""Tests for the public `card_framework.infer` API."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import card_framework.api as infer_api
from card_framework.api import InferenceResult, infer
from card_framework.runtime.bootstrap import RuntimeBootstrapError
from card_framework.shared.runtime_layout import RuntimeLayout


@pytest.fixture(autouse=True)
def _force_windows_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep infer API tests on the supported packaged-runtime platform by default."""
    monkeypatch.setattr("card_framework.api.platform.system", lambda: "Windows")


@pytest.fixture(autouse=True)
def _stub_resolve_uv_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep infer API tests independent from host uv PATH configuration."""
    monkeypatch.setattr(
        "card_framework.api.resolve_uv_executable",
        lambda *, uv_executable: uv_executable,
    )


@pytest.fixture(autouse=True)
def _stub_ctc_forced_aligner_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep infer API tests from attempting a real aligner bootstrap install."""
    monkeypatch.setattr(
        "card_framework.api.ensure_ctc_forced_aligner_runtime",
        lambda **_: None,
    )


def test_infer_rejects_missing_audio(tmp_path: Path) -> None:
    """Fail fast when the source audio path does not exist."""
    with pytest.raises(FileNotFoundError):
        infer(tmp_path / "missing.wav", tmp_path / "outputs", 300, device="cpu")


def test_infer_rejects_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject packaged whole-pipeline runs outside the supported OS contract."""
    monkeypatch.setattr("card_framework.api.platform.system", lambda: "Linux")

    with pytest.raises(RuntimeError, match="Windows only"):
        infer(tmp_path / "missing.wav", tmp_path / "outputs", 300, device="cpu")


def test_infer_rejects_non_positive_target_duration(tmp_path: Path) -> None:
    """Reject invalid duration targets before pipeline execution starts."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")

    with pytest.raises(ValueError, match="target_duration_seconds"):
        infer(audio_path, tmp_path / "outputs", 0, device="cpu")


def test_infer_rejects_invalid_device(tmp_path: Path) -> None:
    """Reject unsupported packaged-runtime device values."""
    with pytest.raises(ValueError, match="device"):
        infer(
            tmp_path / "missing.wav",
            tmp_path / "outputs",
            300,
            device="metal",  # type: ignore[arg-type]
        )


def test_infer_rejects_cuda_without_supported_126_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject CUDA requests unless PyTorch reports a CUDA 12.6 runtime."""
    _set_fake_torch_runtime_inspections(
        monkeypatch,
        _torch_runtime_state(cuda_version="12.8", cuda_available=True),
    )
    monkeypatch.setattr(
        "card_framework.api._detect_host_cuda_runtime",
        lambda: infer_api._HostCudaDetection(version=None, source=None),
    )

    with pytest.raises(RuntimeError, match="only CUDA 12.6"):
        infer(tmp_path / "missing.wav", tmp_path / "outputs", 300, device="cuda")


def test_infer_auto_repairs_cpu_torch_when_host_reports_cuda_126(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Replace the default CPU PyTorch wheels when the host reports CUDA 12.6."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "audio:",
                "  voice_clone:",
                "    enabled: false",
                "    live_drafting:",
                "      enabled: false",
                "  interjector:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api.ensure_runtime_requirements", lambda **_: None)
    monkeypatch.setattr(
        "card_framework.api.resolve_runtime_layout",
        lambda: _runtime_layout(tmp_path),
    )
    monkeypatch.setattr("card_framework.api.ensure_index_tts_runtime", lambda **_: None)
    _set_fake_torch_runtime_inspections(
        monkeypatch,
        _torch_runtime_state(cuda_version=None, cuda_available=False),
        _torch_runtime_state(cuda_version="12.6", cuda_available=True),
    )
    monkeypatch.setattr(
        "card_framework.api._detect_host_cuda_runtime",
        lambda: infer_api._HostCudaDetection(version="12.6", source="nvidia-smi"),
    )

    install_calls: list[str] = []
    monkeypatch.setattr(
        "card_framework.api._install_supported_torch_cuda_runtime",
        lambda: install_calls.append("install"),
    )
    monkeypatch.setattr(
        "card_framework.api.subprocess.run",
        lambda command, cwd=None, check=False, capture_output=True, text=True, encoding=None, errors=None, env=None: (
            _write_success_outputs(
                _command_output_root(
                    command,
                    Path(cwd) if cwd is not None else tmp_path,
                )
            )
            or subprocess.CompletedProcess(command, 0, "", "")
        ),
    )

    with pytest.warns(RuntimeWarning, match="automatically replaced"):
        infer(audio_path, tmp_path / "outputs", 180, device="cuda")

    assert install_calls == ["install"]


def test_install_supported_torch_cuda_runtime_prefers_uv_in_uv_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use `uv pip` instead of raw pip when the caller is in a uv-managed project."""
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "card_framework.api.resolve_uv_executable",
        lambda *, uv_executable: "uv",
    )
    monkeypatch.setattr("card_framework.api._clear_loaded_torch_modules", lambda: None)

    recorded_commands: list[tuple[list[str], str | None]] = []

    def _fake_run(
        command: list[str],
        cwd: Path | str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, encoding, errors, env
        recorded_commands.append((command, str(cwd) if cwd is not None else None))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("card_framework.api.subprocess.run", _fake_run)

    infer_api._install_supported_torch_cuda_runtime()

    assert recorded_commands[0][0] == [
        "uv",
        "pip",
        "uninstall",
        "--python",
        sys.executable,
        "torch",
        "torchaudio",
    ]
    assert recorded_commands[1][0] == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall",
        "--torch-backend",
        "cu126",
        "torch",
        "torchaudio",
    ]
    assert recorded_commands[0][1] == str(tmp_path)
    assert recorded_commands[1][1] == str(tmp_path)


def test_infer_runs_pipeline_and_returns_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Execute the public API through its subprocess boundary and return artifacts."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "audio:",
                "  voice_clone:",
                "    enabled: true",
                "    output_dir_name: voice_clone",
                "    manifest_filename: manifest.json",
                "    merged_output_filename: voice_cloned.wav",
                "    live_drafting:",
                "      enabled: true",
                "  interjector:",
                "    enabled: true",
                "    output_dir_name: interjector",
                "    manifest_filename: interjector_manifest.json",
                "    merged_output_filename: voice_cloned_interjected.wav",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    layout = _runtime_layout(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api.ensure_runtime_requirements", lambda **_: None)
    bootstrap_calls: list[tuple[RuntimeLayout, str, str]] = []

    def _fake_bootstrap(
        *,
        layout: RuntimeLayout,
        uv_executable: str,
        python_executable: str,
        **_: object,
    ) -> None:
        bootstrap_calls.append((layout, uv_executable, python_executable))

    monkeypatch.setattr("card_framework.api.ensure_index_tts_runtime", _fake_bootstrap)
    monkeypatch.setattr("card_framework.api.resolve_runtime_layout", lambda: layout)

    recorded_command: list[str] = []
    recorded_cwd: str | None = None
    recorded_encoding: str | None = None
    recorded_errors: str | None = None

    def _fake_run(
        command: list[str],
        cwd: Path | str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal recorded_cwd, recorded_encoding, recorded_errors
        del check, capture_output, text, env
        recorded_command[:] = command
        recorded_cwd = str(cwd) if cwd is not None else None
        recorded_encoding = encoding
        recorded_errors = errors
        output_root = _command_output_root(
            command,
            Path(cwd) if cwd is not None else tmp_path,
        )
        _write_success_outputs(output_root)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("card_framework.api.subprocess.run", _fake_run)

    result = infer(audio_path, tmp_path / "outputs", 300, device="cpu")

    assert isinstance(result, InferenceResult)
    assert result.output_dir == (tmp_path / "outputs").resolve()
    assert result.transcript_path == (tmp_path / "outputs" / "transcript.json").resolve()
    assert result.summary_xml_path == (tmp_path / "outputs" / "summary.xml").resolve()
    assert result.voice_clone_manifest_path == (
        tmp_path / "outputs" / "audio_stage" / "voice_clone" / "manifest.json"
    ).resolve()
    assert result.voice_clone_audio_path == (
        tmp_path / "outputs" / "audio_stage" / "voice_clone" / "voice_cloned.wav"
    ).resolve()
    assert result.interjector_manifest_path == (
        tmp_path
        / "outputs"
        / "audio_stage"
        / "interjector"
        / "interjector_manifest.json"
    ).resolve()
    assert result.final_audio_path == (
        tmp_path
        / "outputs"
        / "audio_stage"
        / "interjector"
        / "voice_cloned_interjected.wav"
    ).resolve()
    assert bootstrap_calls == [(layout, "uv", sys.executable)]
    assert recorded_cwd == str(tmp_path)
    assert recorded_encoding == "utf-8"
    assert recorded_errors == "replace"
    assert recorded_command[:3] == [sys.executable, "-m", "card_framework.cli.setup_and_run"]
    assert "--config-file" in recorded_command
    recorded_config_path = Path(recorded_command[recorded_command.index("--config-file") + 1])
    assert recorded_config_path.suffix in {".yaml", ".yml"}
    assert "--workspace-root" in recorded_command
    assert str(tmp_path) in recorded_command
    assert "--output-root" in recorded_command
    assert str((tmp_path / "outputs").resolve()) in recorded_command
    assert "--audio-path" in recorded_command
    assert str(audio_path.resolve()) in recorded_command
    assert "pipeline.start_stage=stage-1" in recorded_command
    assert "orchestrator.target_seconds=300" in recorded_command


def test_infer_bootstraps_ctc_forced_aligner_for_default_stage1_alignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bootstrap the pinned aligner when packaged stage-1 settings require it."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "audio:",
                "  voice_clone:",
                "    enabled: false",
                "    live_drafting:",
                "      enabled: false",
                "  interjector:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api.ensure_runtime_requirements", lambda **_: None)
    monkeypatch.setattr("card_framework.api.resolve_runtime_layout", lambda: _runtime_layout(tmp_path))
    monkeypatch.setattr("card_framework.api.ensure_index_tts_runtime", lambda **_: None)

    bootstrap_calls: list[str] = []
    monkeypatch.setattr(
        "card_framework.api.ensure_ctc_forced_aligner_runtime",
        lambda *, python_executable: bootstrap_calls.append(python_executable),
    )
    monkeypatch.setattr(
        "card_framework.api.subprocess.run",
        lambda command, cwd=None, check=False, capture_output=True, text=True, encoding=None, errors=None, env=None: (
            _write_success_outputs(
                _command_output_root(
                    command,
                    Path(cwd) if cwd is not None else tmp_path,
                )
            )
            or subprocess.CompletedProcess(command, 0, "", "")
        ),
    )

    infer(audio_path, tmp_path / "outputs", 180, device="cpu")

    assert bootstrap_calls == [sys.executable]


def test_infer_disables_packaged_forced_alignment_when_bootstrap_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fall back to approximate timings when the pinned aligner cannot be installed."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "audio:",
                "  voice_clone:",
                "    enabled: false",
                "    live_drafting:",
                "      enabled: false",
                "  interjector:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api.ensure_runtime_requirements", lambda **_: None)
    monkeypatch.setattr("card_framework.api.resolve_runtime_layout", lambda: _runtime_layout(tmp_path))
    monkeypatch.setattr("card_framework.api.ensure_index_tts_runtime", lambda **_: None)
    monkeypatch.setattr(
        "card_framework.api.ensure_ctc_forced_aligner_runtime",
        lambda **_: (_ for _ in ()).throw(
            RuntimeBootstrapError(
                step="ctc_forced_aligner_install",
                message="missing build tools",
            )
        ),
    )

    invoked_config: dict[str, Any] = {}

    def _fake_run(
        command: list[str],
        cwd: Path | str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, encoding, errors, env
        invoked_config.update(_load_invoked_config(command))
        output_root = _command_output_root(
            command,
            Path(cwd) if cwd is not None else tmp_path,
        )
        (output_root / "transcript.json").write_text('{"segments": []}\n', encoding="utf-8")
        (output_root / "summary.xml").write_text("<summary></summary>\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("card_framework.api.subprocess.run", _fake_run)

    with pytest.warns(RuntimeWarning, match="approximate timing fallbacks"):
        infer(audio_path, tmp_path / "outputs", 180, device="cpu")

    forced_alignment_cfg = _mapping(
        _mapping(_mapping(invoked_config["audio"])["asr"])["forced_alignment"]
    )
    assert forced_alignment_cfg["enabled"] is False
    assert forced_alignment_cfg["required"] is False


def test_infer_skips_ctc_forced_aligner_bootstrap_when_alignment_and_interjector_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not bootstrap the aligner when packaged config disables both call sites."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "audio:",
                "  asr:",
                "    forced_alignment:",
                "      enabled: false",
                "  voice_clone:",
                "    enabled: false",
                "    live_drafting:",
                "      enabled: false",
                "  interjector:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api.ensure_runtime_requirements", lambda **_: None)
    monkeypatch.setattr("card_framework.api.resolve_runtime_layout", lambda: _runtime_layout(tmp_path))
    monkeypatch.setattr("card_framework.api.ensure_index_tts_runtime", lambda **_: None)

    bootstrap_calls: list[str] = []
    monkeypatch.setattr(
        "card_framework.api.ensure_ctc_forced_aligner_runtime",
        lambda *, python_executable: bootstrap_calls.append(python_executable),
    )
    monkeypatch.setattr(
        "card_framework.api.subprocess.run",
        lambda command, cwd=None, check=False, capture_output=True, text=True, encoding=None, errors=None, env=None: (
            _write_success_outputs(
                _command_output_root(
                    command,
                    Path(cwd) if cwd is not None else tmp_path,
                )
            )
            or subprocess.CompletedProcess(command, 0, "", "")
        ),
    )

    infer(audio_path, tmp_path / "outputs", 180, device="cpu")

    assert bootstrap_calls == []


def test_infer_omits_voice_clone_overrides_when_voice_clone_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not inject stage-3 runner paths into the CLI when synthesis is disabled."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "audio:",
                "  voice_clone:",
                "    enabled: false",
                "    live_drafting:",
                "      enabled: false",
                "  interjector:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    layout = _runtime_layout(tmp_path)
    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api.ensure_runtime_requirements", lambda **_: None)
    monkeypatch.setattr("card_framework.api.resolve_runtime_layout", lambda: layout)
    monkeypatch.setattr("card_framework.api.ensure_index_tts_runtime", lambda **_: None)

    recorded_command: list[str] = []

    def _fake_run(
        command: list[str],
        cwd: Path | str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, encoding, errors, env
        recorded_command[:] = command
        output_root = _command_output_root(
            command,
            Path(cwd) if cwd is not None else tmp_path,
        )
        (output_root / "transcript.json").write_text('{"segments": []}\n', encoding="utf-8")
        (output_root / "summary.xml").write_text("<summary></summary>\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("card_framework.api.subprocess.run", _fake_run)

    result = infer(audio_path, tmp_path / "outputs", 180, device="cpu")

    assert "--output-root" in recorded_command
    assert "orchestrator.target_seconds=180" in recorded_command
    assert result.voice_clone_manifest_path is None
    assert result.voice_clone_audio_path is None
    assert result.interjector_manifest_path is None
    assert result.final_audio_path is None


def test_infer_vllm_url_override_forces_vllm_first_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use the per-call vLLM URL override to replace config-selected LLM providers."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  _target_: card_framework.providers.deepseek_provider.DeepSeekProvider",
                "  api_key: existing-deepseek",
                "stage_llm:",
                "  critic:",
                "    _target_: card_framework.providers.deepseek_provider.DeepSeekProvider",
                "    api_key: critic-deepseek",
                "audio:",
                "  voice_clone:",
                "    enabled: false",
                "    live_drafting:",
                "      enabled: false",
                "  interjector:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api.ensure_runtime_requirements", lambda **_: None)
    monkeypatch.setattr("card_framework.api.resolve_runtime_layout", lambda: _runtime_layout(tmp_path))
    monkeypatch.setattr("card_framework.api.ensure_index_tts_runtime", lambda **_: None)
    _install_fake_cuda_126_runtime(monkeypatch)

    recorded_command: list[str] = []
    invoked_config: dict[str, Any] = {}

    def _fake_run(
        command: list[str],
        cwd: Path | str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, encoding, errors, env
        recorded_command[:] = command
        invoked_config.update(_load_invoked_config(command))
        output_root = _command_output_root(
            command,
            Path(cwd) if cwd is not None else tmp_path,
        )
        (output_root / "transcript.json").write_text('{"segments": []}\n', encoding="utf-8")
        (output_root / "summary.xml").write_text("<summary></summary>\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("card_framework.api.subprocess.run", _fake_run)

    infer(
        audio_path,
        tmp_path / "outputs",
        240,
        device="cuda",
        vllm_url="http://vllm.example:8000/v1",
        vllm_api_key="vllm-secret",
    )

    assert invoked_config["llm"] == {
        "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
        "base_url": "http://vllm.example:8000/v1",
        "api_key": "vllm-secret",
    }
    assert invoked_config["stage_llm"] == {
        "summarizer": {},
        "critic": {},
        "interjector": {},
    }
    assert _mapping(invoked_config["audio"])["device"] == "cuda"
    assert _mapping(_mapping(invoked_config["audio"])["voice_clone"])["device"] == "cuda"
    assert not any("vllm-secret" in item for item in recorded_command)


def test_infer_prompts_for_missing_provider_api_key_without_cli_secret_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prompt securely for missing API keys and keep them out of the subprocess argv."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  _target_: card_framework.providers.deepseek_provider.DeepSeekProvider",
                "audio:",
                "  voice_clone:",
                "    enabled: false",
                "    live_drafting:",
                "      enabled: false",
                "  interjector:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api.ensure_runtime_requirements", lambda **_: None)
    monkeypatch.setattr("card_framework.api.resolve_runtime_layout", lambda: _runtime_layout(tmp_path))
    monkeypatch.setattr("card_framework.api.ensure_index_tts_runtime", lambda **_: None)
    monkeypatch.setattr("card_framework.api._can_prompt_interactively", lambda: True)
    monkeypatch.setattr(
        "card_framework.api.getpass.getpass",
        lambda prompt: "deepseek-secret",
    )

    recorded_command: list[str] = []
    invoked_config: dict[str, Any] = {}

    def _fake_run(
        command: list[str],
        cwd: Path | str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, encoding, errors, env
        recorded_command[:] = command
        invoked_config.update(_load_invoked_config(command))
        output_root = _command_output_root(
            command,
            Path(cwd) if cwd is not None else tmp_path,
        )
        (output_root / "transcript.json").write_text('{"segments": []}\n', encoding="utf-8")
        (output_root / "summary.xml").write_text("<summary></summary>\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("card_framework.api.subprocess.run", _fake_run)

    infer(audio_path, tmp_path / "outputs", 210, device="cpu")

    assert _mapping(invoked_config["llm"])["api_key"] == "deepseek-secret"
    assert not any("deepseek-secret" in item for item in recorded_command)


def test_infer_prompts_for_pyannote_access_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prompt for the Hugging Face token required by the pyannote diarization path."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  _target_: card_framework.providers.vllm_provider.VLLMProvider",
                "  base_url: http://localhost:8000/v1",
                "  api_key: EMPTY",
                "audio:",
                "  diarization:",
                "    provider: pyannote_community1",
                "    pyannote:",
                "      auth_token: ''",
                "      auth_token_env: HUGGINGFACE_TOKEN",
                "  voice_clone:",
                "    enabled: false",
                "    live_drafting:",
                "      enabled: false",
                "  interjector:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api.ensure_runtime_requirements", lambda **_: None)
    monkeypatch.setattr("card_framework.api.resolve_runtime_layout", lambda: _runtime_layout(tmp_path))
    monkeypatch.setattr("card_framework.api.ensure_index_tts_runtime", lambda **_: None)
    monkeypatch.setattr("card_framework.api._can_prompt_interactively", lambda: True)
    monkeypatch.setattr(
        "card_framework.api.getpass.getpass",
        lambda prompt: "hf_access_token",
    )

    invoked_config: dict[str, Any] = {}

    def _fake_run(
        command: list[str],
        cwd: Path | str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, encoding, errors, env
        invoked_config.update(_load_invoked_config(command))
        output_root = _command_output_root(
            command,
            Path(cwd) if cwd is not None else tmp_path,
        )
        (output_root / "transcript.json").write_text('{"segments": []}\n', encoding="utf-8")
        (output_root / "summary.xml").write_text("<summary></summary>\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("card_framework.api.subprocess.run", _fake_run)

    infer(audio_path, tmp_path / "outputs", 210, device="cpu")

    pyannote_cfg = _mapping(_mapping(_mapping(invoked_config["audio"])["diarization"])["pyannote"])
    assert pyannote_cfg["auth_token"] == "hf_access_token"


def test_infer_rejects_missing_credentials_without_interactive_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail with actionable guidance when a required credential cannot be resolved."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  _target_: card_framework.providers.deepseek_provider.DeepSeekProvider",
                "audio:",
                "  voice_clone:",
                "    enabled: false",
                "    live_drafting:",
                "      enabled: false",
                "  interjector:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api._can_prompt_interactively", lambda: False)

    with pytest.raises(RuntimeError) as exc_info:
        infer(audio_path, tmp_path / "outputs", 300, device="cpu")

    message = str(exc_info.value)
    assert "DEEPSEEK_API_KEY" in message
    assert "`llm.api_key`" in message


def test_infer_surfaces_runtime_bootstrap_error_without_type_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preserve the original bootstrap exception when temp config cleanup is active."""
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  _target_: card_framework.providers.deepseek_provider.DeepSeekProvider",
                "audio:",
                "  voice_clone:",
                "    enabled: false",
                "    live_drafting:",
                "      enabled: false",
                "  interjector:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("CARD_FRAMEWORK_CONFIG", str(config_path))
    monkeypatch.setattr("card_framework.api.ensure_runtime_requirements", lambda **_: None)
    monkeypatch.setattr("card_framework.api.resolve_runtime_layout", lambda: _runtime_layout(tmp_path))
    monkeypatch.setattr("card_framework.api._can_prompt_interactively", lambda: True)
    monkeypatch.setattr(
        "card_framework.api.getpass.getpass",
        lambda prompt: "deepseek-secret",
    )

    def _raise_bootstrap_error(**_: object) -> None:
        raise RuntimeBootstrapError(step="bootstrap", message="simulated failure")

    monkeypatch.setattr("card_framework.api.ensure_index_tts_runtime", _raise_bootstrap_error)

    with pytest.raises(RuntimeBootstrapError, match="simulated failure"):
        infer(audio_path, tmp_path / "outputs", 210, device="cpu")


def _runtime_layout(tmp_path: Path) -> RuntimeLayout:
    """Build one deterministic runtime layout for API tests."""
    return RuntimeLayout(
        runtime_home=(tmp_path / "runtime_home").resolve(),
        vendor_source_dir=(tmp_path / "vendor_source").resolve(),
        vendor_runtime_dir=(tmp_path / "runtime_home" / "vendor" / "index_tts").resolve(),
        checkpoints_dir=(tmp_path / "runtime_home" / "checkpoints" / "index_tts").resolve(),
        bootstrap_state_path=(tmp_path / "runtime_home" / "bootstrap" / "state.json").resolve(),
    )


def _install_fake_cuda_126_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal fake torch module that satisfies the CUDA 12.6 contract."""
    _set_fake_torch_runtime_inspections(
        monkeypatch,
        _torch_runtime_state(cuda_version="12.6", cuda_available=True),
    )


def _set_fake_torch_runtime_inspections(
    monkeypatch: pytest.MonkeyPatch,
    *states: infer_api._TorchCudaRuntimeState,
) -> None:
    """Patch the packaged CUDA inspection helper with deterministic states."""
    state_iterator = iter(states)
    monkeypatch.setattr(
        "card_framework.api._inspect_torch_cuda_runtime",
        lambda: next(state_iterator),
    )


def _torch_runtime_state(
    *,
    cuda_version: str | None,
    cuda_available: bool,
    import_error: str | None = None,
) -> infer_api._TorchCudaRuntimeState:
    """Build one fake PyTorch runtime inspection payload."""
    return infer_api._TorchCudaRuntimeState(
        cuda_version=cuda_version,
        cuda_available=cuda_available,
        import_error=import_error,
    )


def _load_invoked_config(command: list[str]) -> dict[str, Any]:
    """Load the effective config file passed to the CLI subprocess."""
    from omegaconf import OmegaConf

    if "--config-file" in command:
        config_path = Path(command[command.index("--config-file") + 1])
    else:
        config_dir = Path(command[command.index("--config-path") + 1])
        config_name = command[command.index("--config-name") + 1]
        config_path = config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            config_path = config_dir / f"{config_name}.yml"
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _command_output_root(command: list[str], fallback_root: Path) -> Path:
    """Resolve the infer wrapper output root from one recorded command."""
    if "--output-root" not in command:
        return fallback_root.resolve()
    return Path(command[command.index("--output-root") + 1]).resolve()


def _write_success_outputs(output_root: Path) -> None:
    """Create the minimal success artifact set for infer result construction."""
    (output_root / "transcript.json").write_text('{"segments": []}\n', encoding="utf-8")
    (output_root / "summary.xml").write_text("<summary></summary>\n", encoding="utf-8")
    voice_clone_dir = output_root / "audio_stage" / "voice_clone"
    voice_clone_dir.mkdir(parents=True, exist_ok=True)
    (voice_clone_dir / "manifest.json").write_text('{"artifacts": []}\n', encoding="utf-8")
    (voice_clone_dir / "voice_cloned.wav").write_bytes(b"RIFF")
    interjector_dir = output_root / "audio_stage" / "interjector"
    interjector_dir.mkdir(parents=True, exist_ok=True)
    (interjector_dir / "interjector_manifest.json").write_text(
        '{"artifacts": []}\n',
        encoding="utf-8",
    )
    (interjector_dir / "voice_cloned_interjected.wav").write_bytes(b"RIFF")


def _mapping(value: object) -> Mapping[str, Any]:
    """Narrow one loaded config node to a string-keyed mapping for assertions."""
    assert isinstance(value, Mapping)
    return value
