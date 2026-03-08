"""Tests for the summary matrix helper script."""

from __future__ import annotations

import io
import importlib.util
from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _ROOT / "scripts" / "run_summary_matrix.py"
_SPEC = importlib.util.spec_from_file_location(
    "run_summary_matrix_for_tests",
    _SCRIPT_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load scripts/run_summary_matrix.py for tests.")
run_summary_matrix = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = run_summary_matrix
_SPEC.loader.exec_module(run_summary_matrix)


def test_build_summary_filename_uses_expected_prefix() -> None:
    """Name per-pair summary files with summarizer and critic slugs first."""
    summarizer = run_summary_matrix.DEFAULT_VLLM_MODELS[0]
    critic = run_summary_matrix.ModelProfile(
        slug="deepseek_chat",
        model_name="deepseek-chat",
        provider_target="providers.deepseek_provider.DeepSeekProvider",
    )

    filename = run_summary_matrix.build_summary_filename(summarizer, critic)

    assert filename == "qwen3_5_27b_deepseek_chat-summary.xml"


def test_build_model_profiles_adds_deepseek_when_requested() -> None:
    """Append the optional DeepSeek model when a key is available."""
    model_profiles = run_summary_matrix.build_model_profiles(
        include_deepseek=True,
        deepseek_model="deepseek-reasoner",
    )

    assert [profile.slug for profile in model_profiles] == [
        "qwen3_5_27b",
        "qwen3_5_9b",
        "qwen3_5_4b",
        "deepseek_reasoner",
    ]


def test_build_model_pairs_include_self_pairs() -> None:
    """Build ordered summarizer/critic pairs including self-pairs."""
    model_profiles = run_summary_matrix.build_model_profiles(
        include_deepseek=True,
        deepseek_model="deepseek-reasoner",
    )

    pairs = run_summary_matrix.build_model_pairs(model_profiles)

    assert len(pairs) == 16
    assert (model_profiles[0], model_profiles[0]) in pairs
    assert (model_profiles[0], model_profiles[1]) in pairs
    assert (model_profiles[1], model_profiles[0]) in pairs


def test_resolve_input_source_rejects_transcript_without_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Require a usable speaker-sample manifest for non-interactive stage-2 runs."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    transcript_path = repo_root / "transcript.json"
    transcript_path.write_text('{"segments": [], "metadata": {}}', encoding="utf-8")

    monkeypatch.setattr(run_summary_matrix, "REPO_ROOT", repo_root)

    args = run_summary_matrix.parse_args(
        [
            "--vllm-host",
            "example-host",
            "--transcript-path",
            "transcript.json",
        ]
    )

    with pytest.raises(
        run_summary_matrix.SummaryMatrixError,
        match="speaker_samples_manifest_path",
    ):
        run_summary_matrix.resolve_input_source(args)


def test_build_setup_command_uses_summary_only_overrides_and_env_interpolation(
    tmp_path: Path,
) -> None:
    """Build a vLLM summarizer plus DeepSeek critic command safely."""
    args = run_summary_matrix.parse_args(
        [
            "--vllm-host",
            "202.92.159.240",
            "--transcript-path",
            "transcript.json",
            "--override",
            "orchestrator.max_iterations=2",
        ]
    )
    resolved_input = run_summary_matrix.ResolvedInput(
        mode="transcript",
        path=(tmp_path / "transcript.json").resolve(),
    )
    summarizer = run_summary_matrix.DEFAULT_VLLM_MODELS[0]
    critic = run_summary_matrix.build_model_profiles(
        include_deepseek=True,
        deepseek_model="deepseek-chat",
    )[-1]

    command = run_summary_matrix.build_setup_command(
        args=args,
        resolved_input=resolved_input,
        summarizer=summarizer,
        critic=critic,
        output_dir=tmp_path,
    )

    joined = " ".join(command)
    assert "audio.interjector.enabled=false" in joined
    assert "audio.voice_clone.enabled=false" not in joined
    assert "audio.voice_clone.live_drafting.enabled=false" not in joined
    assert "pipeline.start_stage=stage-2" in joined
    assert "orchestrator.max_iterations=2" in joined
    assert "llm.base_url=http://${oc.env:SUMMARY_MATRIX_VLLM_HOST}:8000/v1" in joined
    assert "+stage_llm.critic._target_=providers.deepseek_provider.DeepSeekProvider" in joined
    assert "+stage_llm.critic.api_key=${oc.env:SUMMARY_MATRIX_DEEPSEEK_API_KEY}" in joined
    assert "202.92.159.240:8000" not in joined


def test_build_setup_command_supports_deepseek_summarizer_and_vllm_critic(
    tmp_path: Path,
) -> None:
    """Build a DeepSeek summarizer plus vLLM critic command safely."""
    args = run_summary_matrix.parse_args(
        [
            "--vllm-host",
            "202.92.159.240",
            "--transcript-path",
            "transcript.json",
        ]
    )
    resolved_input = run_summary_matrix.ResolvedInput(
        mode="transcript",
        path=(tmp_path / "transcript.json").resolve(),
    )
    model_profiles = run_summary_matrix.build_model_profiles(
        include_deepseek=True,
        deepseek_model="deepseek-reasoner",
    )
    summarizer = model_profiles[-1]
    critic = model_profiles[0]

    command = run_summary_matrix.build_setup_command(
        args=args,
        resolved_input=resolved_input,
        summarizer=summarizer,
        critic=critic,
        output_dir=tmp_path,
    )

    joined = " ".join(command)
    assert "llm.api_key=${oc.env:SUMMARY_MATRIX_DEEPSEEK_API_KEY}" in joined
    assert "llm.model=deepseek-reasoner" in joined
    assert "llm.base_url=${oc.env:SUMMARY_MATRIX_DEEPSEEK_BASE_URL}" in joined
    assert "+stage_llm.critic._target_=providers.vllm_provider.VLLMProvider" in joined
    assert "+stage_llm.critic.base_url=http://${oc.env:SUMMARY_MATRIX_VLLM_HOST}:8000/v1" in joined


def test_run_pair_streams_output_copies_summary_and_writes_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist the copied summary XML, live stream, and child log."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(run_summary_matrix, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        run_summary_matrix,
        "SUMMARY_SOURCE_PATH",
        repo_root / "summary.xml",
    )

    class _FakePopen:
        def __init__(
            self,
            command: list[str],
            *,
            cwd: Path,
            env: dict[str, str],
            stdout: int,
            stderr: int,
            text: bool,
            encoding: str,
            errors: str,
            bufsize: int,
        ) -> None:
            del command, env, stdout, stderr, text, encoding, errors, bufsize
            assert cwd == repo_root
            (repo_root / "summary.xml").write_text(
                "<SPEAKER_00>candidate summary</SPEAKER_00>\n",
                encoding="utf-8",
            )
            self.stdout = io.StringIO("pipeline ok\n")

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(run_summary_matrix.subprocess, "Popen", _FakePopen)
    streamed_stdout = io.StringIO()
    monkeypatch.setattr(run_summary_matrix.sys, "stdout", streamed_stdout)

    args = run_summary_matrix.parse_args(
        [
            "--vllm-host",
            "example-host",
            "--audio-path",
            str((tmp_path / "audio.wav").resolve()),
        ]
    )
    resolved_input = run_summary_matrix.ResolvedInput(
        mode="audio",
        path=(tmp_path / "audio.wav").resolve(),
    )
    summarizer = run_summary_matrix.DEFAULT_VLLM_MODELS[1]
    critic = run_summary_matrix.build_model_profiles(
        include_deepseek=False,
        deepseek_model="deepseek-chat",
    )[2]

    result = run_summary_matrix.run_pair(
        args=args,
        resolved_input=resolved_input,
        summarizer=summarizer,
        critic=critic,
        output_dir=output_dir,
        env={"SUMMARY_MATRIX_VLLM_HOST": "example-host"},
    )

    assert result.status == "ok"
    summary_path = Path(result.summary_path)
    log_path = Path(result.log_path)
    assert summary_path.name == "qwen3_5_9b_qwen3_5_4b-summary.xml"
    assert summary_path.read_text(encoding="utf-8").startswith("<SPEAKER_00>")
    assert log_path.read_text(encoding="utf-8") == "pipeline ok\n"
    assert streamed_stdout.getvalue() == "pipeline ok\n"
