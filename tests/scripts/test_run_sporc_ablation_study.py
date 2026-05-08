"""Coverage for the SPoRC ablation-study wrapper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf

from card_framework.shared.paths import REPO_ROOT


def _load_module():
    script_path = (REPO_ROOT / "scripts" / "run_sporc_ablation_study.py").resolve()
    spec = importlib.util.spec_from_file_location("run_sporc_ablation_study", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_base_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  duration_tolerance_ratio: 0.05",
                "  max_iterations: 60",
                "  loop_memory:",
                "    artifact_path: artifacts/audio_stage/summarizer_loop_memory.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _args(module, tmp_path: Path, config_path: Path):
    return module.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "ablation"),
            "--dataset-manifest",
            str(tmp_path / "manifest.json"),
            "--config",
            str(config_path),
            "--vllm-base-url",
            "http://127.0.0.1:8000/v1",
            "--vllm-model",
            "Qwen/Qwen3.6-27B",
            "--deepseek-api-key-env",
            "DEEPSEEK_API_KEY",
        ]
    )


def _cell(module):
    return module.AblationCell(
        cell_id="s_thinking__c_thinking__baseline",
        ablation="baseline",
        summarizer_thinking=True,
        critic_thinking=True,
        summarizer_provider=module.VLLM_THINKING_PROVIDER_ID,
        critic_provider=module.VLLM_THINKING_PROVIDER_ID,
        judge_provider=module.DEEPSEEK_JUDGE_PROVIDER_ID,
        memory_enabled=True,
        max_iterations=1,
        config_path="config.yaml",
        output_root="runs",
        checkpoint_path="checkpoint.json",
        log_path="execute.log",
        command=("python", "-V"),
    )


def test_build_ablation_cells_materializes_default_hierarchy(tmp_path: Path) -> None:
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    _write_base_config(config_path)
    args = _args(module, tmp_path, config_path)
    output_root = tmp_path / "ablation"

    provider_profiles_path = module._write_provider_profiles(args, output_root)
    cells = module._build_ablation_cells(
        args,
        output_root=output_root,
        provider_profiles_path=provider_profiles_path,
    )

    assert len(cells) == 6
    no_critic_cells = [cell for cell in cells if cell.ablation == "no_critic"]
    assert no_critic_cells
    assert all(cell.max_iterations == 0 for cell in no_critic_cells)
    assert all(cell.memory_enabled for cell in no_critic_cells)
    no_memory_cell = next(cell for cell in cells if cell.ablation == "no_memory")
    no_memory_cfg = OmegaConf.load(no_memory_cell.config_path)
    assert no_memory_cfg.orchestrator.loop_memory.artifact_path == ""
    no_critic_cell = next(cell for cell in cells if cell.ablation == "no_critic")
    no_critic_cfg = OmegaConf.load(no_critic_cell.config_path)
    assert "summarizer_loop_memory.json" in str(
        no_critic_cfg.orchestrator.loop_memory.artifact_path
    )
    baseline_cell = next(cell for cell in cells if cell.ablation == "baseline")
    baseline_cfg = OmegaConf.load(baseline_cell.config_path)
    assert "loop_memory" in baseline_cfg.orchestrator
    assert "summarizer_loop_memory.json" in str(
        baseline_cfg.orchestrator.loop_memory.artifact_path
    )


def test_write_provider_profiles_sets_nontruncating_controls(tmp_path: Path) -> None:
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    _write_base_config(config_path)
    args = _args(module, tmp_path, config_path)

    provider_profiles_path = module._write_provider_profiles(args, tmp_path)
    payload = OmegaConf.to_container(
        OmegaConf.load(provider_profiles_path),
        resolve=False,
    )

    profiles = {entry["id"]: entry["llm"] for entry in payload["providers"]}
    thinking = profiles[module.VLLM_THINKING_PROVIDER_ID]
    nonthinking = profiles[module.VLLM_NONTHINKING_PROVIDER_ID]
    judge = profiles[module.DEEPSEEK_JUDGE_PROVIDER_ID]
    assert "default_max_tokens" not in thinking
    assert "default_max_tokens" not in nonthinking
    assert thinking["sporc_enable_thinking"] is True
    assert nonthinking["sporc_enable_thinking"] is False
    assert judge["model"] == "deepseek-v4-pro"
    assert judge["default_max_tokens"] == 384_000
    assert judge["thinking_mode"] == "disabled"
    assert judge["stream"] is False
    assert judge["response_format"] == {"type": "json_object"}


def test_write_provider_profiles_accepts_explicit_vllm_token_budget(
    tmp_path: Path,
) -> None:
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    _write_base_config(config_path)
    args = _args(module, tmp_path, config_path)
    args.vllm_max_tokens = 123

    provider_profiles_path = module._write_provider_profiles(args, tmp_path)
    payload = OmegaConf.to_container(
        OmegaConf.load(provider_profiles_path),
        resolve=False,
    )

    profiles = {entry["id"]: entry["llm"] for entry in payload["providers"]}
    assert profiles[module.VLLM_THINKING_PROVIDER_ID]["default_max_tokens"] == 123
    assert profiles[module.VLLM_NONTHINKING_PROVIDER_ID]["default_max_tokens"] == 123


def test_parse_progress_event_accepts_only_machine_progress_lines() -> None:
    module = _load_module()
    payload = {
        module.PROGRESS_EVENT_KEY: "sample_done",
        "sample_index": 1,
        "duration_seconds": 10.0,
    }

    assert module._parse_progress_event(json.dumps(payload)) == payload
    assert module._parse_progress_event("ordinary child log line") is None
    assert module._parse_progress_event(json.dumps({"event": "sample_done"})) is None


def test_progress_tracker_eta_uses_learning_rate() -> None:
    module = _load_module()
    cell = _cell(module)
    tracker = module.AblationProgressTracker(
        total_cells=1,
        total_samples=2,
        eta_learning_rate=0.5,
        progress_interval_seconds=1.0,
        enabled=False,
    )

    tracker.start_cell(cell=cell, cell_index=1)
    tracker.handle_event(
        cell=cell,
        payload={module.PROGRESS_EVENT_KEY: "run_start", "sample_count": 2},
    )
    tracker.handle_event(
        cell=cell,
        payload={
            module.PROGRESS_EVENT_KEY: "sample_done",
            "sample_index": 1,
            "sample_count": 2,
            "sample_id": "sample-a",
            "counts_toward_target": True,
            "duration_seconds": 10.0,
        },
    )

    first_snapshot = tracker.snapshot()
    assert first_snapshot.completed_samples == 1
    assert first_snapshot.ema_sample_seconds == 10.0
    assert first_snapshot.eta_seconds == 10.0

    tracker.handle_event(
        cell=cell,
        payload={
            module.PROGRESS_EVENT_KEY: "sample_done",
            "sample_index": 2,
            "sample_count": 2,
            "sample_id": "sample-b",
            "counts_toward_target": True,
            "duration_seconds": 20.0,
        },
    )

    second_snapshot = tracker.snapshot()
    assert second_snapshot.completed_samples == 2
    assert second_snapshot.ema_sample_seconds == 15.0
    assert second_snapshot.eta_seconds == 0.0


def test_progress_tracker_counts_precompleted_extension_samples() -> None:
    module = _load_module()
    cell = _cell(module)
    tracker = module.AblationProgressTracker(
        total_cells=1,
        total_samples=100,
        eta_learning_rate=0.5,
        progress_interval_seconds=1.0,
        enabled=False,
    )

    tracker.start_cell(cell=cell, cell_index=1, precompleted_sample_count=20)
    assert tracker.snapshot().completed_samples == 20

    tracker.handle_event(
        cell=cell,
        payload={module.PROGRESS_EVENT_KEY: "run_start", "sample_count": 100},
    )
    tracker.handle_event(
        cell=cell,
        payload={module.PROGRESS_EVENT_KEY: "run_complete", "status": "completed"},
    )

    snapshot = tracker.snapshot()
    assert snapshot.completed_samples == 100
    assert snapshot.completed_cells == 1


def test_aggregate_reports_uses_factualness_as_primary_endpoint(tmp_path: Path) -> None:
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    _write_base_config(config_path)
    args = _args(module, tmp_path, config_path)
    output_root = tmp_path / "ablation"
    provider_profiles_path = module._write_provider_profiles(args, output_root)
    cell = module._build_ablation_cells(
        args,
        output_root=output_root,
        provider_profiles_path=provider_profiles_path,
    )[0]
    report_dir = Path(cell.output_root) / "run-1"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "duration_tolerance_ratio": 0.05,
                "results_path": str(report_dir / "results.md"),
                "sample_artifacts_root": str(report_dir / "samples"),
                "summary": {
                    "target_sample_count": 2,
                    "sample_count": 2,
                    "executed_count": 2,
                    "success_count": 2,
                    "failure_count": 0,
                    "replacement_count": 0,
                },
                "samples": [
                    {
                        "status": "completed",
                        "counts_toward_target": True,
                        "target_seconds": 300,
                        "candidate_seconds": 302,
                        "iterations_run": 2,
                        "tool_invocations": 8,
                        "metrics": {
                            "factualness": 0.8,
                            "naturalness": 0.5,
                            "speaker_grammar_similarity": 0.4,
                            "composite": 0.5667,
                        },
                    },
                    {
                        "status": "completed",
                        "counts_toward_target": True,
                        "target_seconds": 300,
                        "candidate_seconds": 330,
                        "iterations_run": 3,
                        "tool_invocations": 10,
                        "metrics": {
                            "factualness": 0.6,
                            "naturalness": 0.7,
                            "speaker_grammar_similarity": 0.6,
                            "composite": 0.6333,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    aggregate = module._aggregate_reports(output_root=output_root, cells=(cell,))

    assert aggregate["primary_metric"] == "factualness"
    summary = aggregate["cells"][0]
    assert summary["mean_factualness"] == 0.7
    assert summary["within_tolerance_rate"] == 0.5
    assert summary["mean_absolute_duration_delta_seconds"] == 16.0
    assert summary["mean_composite"] == 0.6
    report_path = report_dir / "report.json"
    assert module._cell_needs_extension(report_path, requested_sample_count=3)
    assert not module._cell_needs_extension(report_path, requested_sample_count=2)
    assert module._command_with_resume_from_checkpoint(
        ("python", "-m", "runner"),
    ) == ("python", "-m", "runner", "--resume-from-checkpoint")

    paper_section = module._render_paper_section(aggregate)
    assert "\\label{fig:sporc-ablation-factualness}" in paper_section
    assert "Mean factualness by ablation variant" in paper_section
