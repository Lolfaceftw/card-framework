"""Coverage for the SPoRC max-iteration sweep wrapper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf

from card_framework.shared.paths import REPO_ROOT


def _load_module():
    script_path = (REPO_ROOT / "scripts" / "run_sporc_max_iter_sweep.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "run_sporc_max_iter_sweep",
        script_path,
    )
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
            str(tmp_path / "iter_sweep"),
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


def _write_report(report_dir: Path, *, max_iterations: int, factualness: float) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "duration_tolerance_ratio": 0.05,
                "results_path": str(report_dir / "results.md"),
                "sample_artifacts_root": str(report_dir / "samples"),
                "summary": {
                    "target_sample_count": 1,
                    "sample_count": 1,
                    "executed_count": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "replacement_count": 0,
                },
                "samples": [
                    {
                        "status": "completed",
                        "counts_toward_target": True,
                        "target_seconds": 300,
                        "candidate_seconds": 300 + max_iterations,
                        "iterations_run": max_iterations,
                        "tool_invocations": 10 + max_iterations,
                        "metrics": {
                            "factualness": factualness,
                            "naturalness": 0.7,
                            "speaker_grammar_similarity": 0.6,
                            "composite": (factualness + 0.7 + 0.6) / 3,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_parser_defaults_to_one_sample_per_iteration_for_speed() -> None:
    module = _load_module()
    args = module.build_parser().parse_args([])

    assert args.max_samples == 1
    assert args.max_iterations_list == (1, 2, 3, 4, 5)


def test_build_cells_materializes_iteration_axis(tmp_path: Path) -> None:
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    _write_base_config(config_path)
    args = _args(module, tmp_path, config_path)
    output_root = tmp_path / "iter_sweep"

    provider_profiles_path = module._write_provider_profiles(args, output_root)
    cells = module._build_cells(
        args,
        output_root=output_root,
        provider_profiles_path=provider_profiles_path,
    )

    assert [cell.max_iterations for cell in cells] == [1, 2, 3, 4, 5]
    assert all(cell.memory_enabled for cell in cells)
    assert all("--max-samples" in cell.command for cell in cells)
    assert all(cell.command[cell.command.index("--max-samples") + 1] == "1" for cell in cells)
    for cell in cells:
        command = list(cell.command)
        assert command[command.index("--max-iterations") + 1] == str(
            cell.max_iterations
        )
        cfg = OmegaConf.load(cell.config_path)
        assert f"max_iter_{cell.max_iterations}" in str(
            cfg.orchestrator.loop_memory.artifact_path
        )


def test_reports_write_csv_and_svg_for_iteration_curve(tmp_path: Path) -> None:
    module = _load_module()
    config_path = tmp_path / "config.yaml"
    _write_base_config(config_path)
    args = _args(module, tmp_path, config_path)
    args.max_iterations_list = module._parse_max_iterations("1,2")
    output_root = tmp_path / "iter_sweep"
    provider_profiles_path = module._write_provider_profiles(args, output_root)
    cells = module._build_cells(
        args,
        output_root=output_root,
        provider_profiles_path=provider_profiles_path,
    )
    _write_report(Path(cells[0].output_root) / "run-a", max_iterations=1, factualness=0.5)
    _write_report(Path(cells[1].output_root) / "run-b", max_iterations=2, factualness=0.75)

    aggregate = module._aggregate_reports(output_root=output_root, cells=cells)
    json_path, markdown_path, csv_path, svg_path = module._write_reports(
        output_root=output_root,
        aggregate=aggregate,
    )

    assert aggregate["primary_metric"] == "factualness"
    assert [cell["max_iterations"] for cell in aggregate["cells"]] == [1, 2]
    assert aggregate["cells"][1]["mean_factualness"] == 0.75
    assert json_path.exists()
    assert "max_iterations, n".replace(" ", "") in csv_path.read_text(
        encoding="utf-8"
    ).replace(" ", "")
    assert "Highest factualness" in markdown_path.read_text(encoding="utf-8")
    svg = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "judge performance by max_iterations" in svg
    assert "polyline" in svg
