"""Run a SPoRC max-iteration sweep and graph judge performance."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence

from omegaconf import OmegaConf

from card_framework.benchmark.summarizer_critic.reporting_utils import (
    format_confidence_interval,
    mean_confidence_interval_95,
    render_markdown_table,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_PREPARED_FULL_ROOT,
    DURATION_PRESET_SECONDS,
)
from card_framework.shared.paths import DEFAULT_CONFIG_PATH, REPO_ROOT


DEFAULT_DATASET_MANIFEST = (
    DEFAULT_PREPARED_FULL_ROOT / "benchmark_longform_15k_max3_manifest.json"
).resolve()
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts" / "summarizer_critic_benchmark" / "sporc_max_iter_sweeps"
).resolve()
VLLM_THINKING_PROVIDER_ID = "vllm_qwen36_27b_thinking"
VLLM_NONTHINKING_PROVIDER_ID = "vllm_qwen36_27b_nonthinking"
DEEPSEEK_JUDGE_PROVIDER_ID = "deepseek_v4_pro_nonthinking_judge"
PRIMARY_METRIC = "factualness"
METRIC_KEYS = (
    "factualness",
    "naturalness",
    "speaker_grammar_similarity",
    "composite",
)


@dataclass(frozen=True)
class IterationSweepCell:
    """Describe one max-iteration setting in the sweep."""

    cell_id: str
    max_iterations: int
    summarizer_provider: str
    critic_provider: str
    judge_provider: str
    memory_enabled: bool
    config_path: str
    output_root: str
    checkpoint_path: str
    log_path: str
    command: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_env(names: Iterable[str], default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _optional_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    value_int = int(value)
    return value_int if value_int > 0 else None


def _mean(values: Iterable[float]) -> float:
    sample = [float(value) for value in values]
    if not sample:
        return 0.0
    return sum(sample) / len(sample)


def _parse_max_iterations(raw_value: str) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    for part in raw_value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid max-iterations value: {text!r}"
            ) from exc
        if value < 0:
            raise argparse.ArgumentTypeError(
                "Max-iterations values must be non-negative."
            )
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one max-iterations value.")
    return tuple(values)


def _provider_id_for_mode(thinking_mode: str) -> str:
    return (
        VLLM_THINKING_PROVIDER_ID
        if thinking_mode == "thinking"
        else VLLM_NONTHINKING_PROVIDER_ID
    )


def _write_provider_profiles(args: argparse.Namespace, output_root: Path) -> Path:
    """Write run-local provider profiles for the contestant and judge."""
    provider_profiles_path = output_root / "provider_profiles.max_iter.yaml"
    provider_profiles_path.parent.mkdir(parents=True, exist_ok=True)
    vllm_base = {
        "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
        "base_url": args.vllm_base_url,
        "api_key": args.vllm_api_key,
        "model": args.vllm_model,
        "request_timeout_seconds": args.vllm_request_timeout_seconds,
    }
    vllm_max_tokens = _optional_positive_int(args.vllm_max_tokens)
    if vllm_max_tokens is not None:
        vllm_base["default_max_tokens"] = vllm_max_tokens
    thinking_llm = {
        **vllm_base,
        "sporc_enable_thinking": True,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    nonthinking_llm = {
        **vllm_base,
        "sporc_enable_thinking": False,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    profiles = {
        "providers": [
            {
                "id": VLLM_THINKING_PROVIDER_ID,
                "description": "Qwen/Qwen3.6-27B through vLLM with thinking enabled.",
                "required_env": [],
                "llm": thinking_llm,
            },
            {
                "id": VLLM_NONTHINKING_PROVIDER_ID,
                "description": "Qwen/Qwen3.6-27B through vLLM with thinking disabled.",
                "required_env": [],
                "llm": nonthinking_llm,
            },
            {
                "id": DEEPSEEK_JUDGE_PROVIDER_ID,
                "description": "DeepSeek V4 Pro non-thinking SPoRC judge.",
                "required_env": [args.deepseek_api_key_env],
                "llm": {
                    "_target_": (
                        "card_framework.providers.deepseek_provider.DeepSeekProvider"
                    ),
                    "api_key": f"$ENV{{{args.deepseek_api_key_env}}}",
                    "model": args.deepseek_model,
                    "base_url": args.deepseek_base_url,
                    "request_timeout_seconds": args.deepseek_request_timeout_seconds,
                    "default_max_tokens": args.deepseek_max_tokens,
                    "thinking_mode": "disabled",
                    "stream": False,
                    "temperature": args.deepseek_temperature,
                    "response_format": {"type": "json_object"},
                },
            },
        ]
    }
    OmegaConf.save(config=OmegaConf.create(profiles), f=provider_profiles_path)
    return provider_profiles_path.resolve()


def _write_cell_config(
    *,
    base_config_path: Path,
    cell_root: Path,
    memory_enabled: bool,
) -> Path:
    """Write a cell-local config with isolated loop memory."""
    cfg = OmegaConf.load(base_config_path)
    if "orchestrator" not in cfg:
        cfg.orchestrator = {}
    if "loop_memory" not in cfg.orchestrator:
        cfg.orchestrator.loop_memory = {}
    if memory_enabled:
        memory_path = cell_root / "loop_memory" / "summarizer_loop_memory.json"
        cfg.orchestrator.loop_memory.artifact_path = str(memory_path.resolve())
    else:
        cfg.orchestrator.loop_memory.artifact_path = ""
    config_path = cell_root / "config.max_iter.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=config_path)
    return config_path.resolve()


def _build_execute_command(
    *,
    args: argparse.Namespace,
    provider_profiles_path: Path,
    cell_root: Path,
    cell_config_path: Path,
    provider_id: str,
    max_iterations: int,
) -> tuple[str, ...]:
    command: list[str] = [
        sys.executable,
        "-m",
        "card_framework.benchmark.summarizer_critic.sporc",
        "execute",
        "--dataset-manifest",
        str(Path(args.dataset_manifest).resolve()),
        "--output-root",
        str((cell_root / "runs").resolve()),
        "--provider-profiles",
        str(provider_profiles_path),
        "--config",
        str(cell_config_path),
        "--summarizer-provider",
        provider_id,
        "--critic-provider",
        provider_id,
        "--judge-provider",
        DEEPSEEK_JUDGE_PROVIDER_ID,
        "--embedding-mode",
        "disabled",
        "--duration-preset",
        args.duration_preset,
        "--max-samples",
        str(args.max_samples),
        "--sample-seed",
        str(args.sample_seed),
        "--transcript-only",
        "--max-iterations",
        str(max_iterations),
        "--checkpoint-path",
        str((cell_root / "checkpoint.json").resolve()),
        "--summarizer-timeout-seconds",
        str(args.summarizer_timeout_seconds),
        "--critic-timeout-seconds",
        str(args.critic_timeout_seconds),
        "--retrieval-timeout-seconds",
        str(args.retrieval_timeout_seconds),
        "--index-timeout-seconds",
        str(args.index_timeout_seconds),
        "--disable-stream-output",
    ]
    return tuple(command)


def _build_cells(
    args: argparse.Namespace,
    *,
    output_root: Path,
    provider_profiles_path: Path,
) -> tuple[IterationSweepCell, ...]:
    provider_id = _provider_id_for_mode(args.thinking_mode)
    memory_enabled = not bool(args.disable_memory)
    base_config_path = Path(args.config).resolve()
    cells: list[IterationSweepCell] = []
    for max_iterations in args.max_iterations_list:
        cell_id = f"max_iter_{max_iterations}"
        cell_root = output_root / "cells" / cell_id
        cell_config_path = _write_cell_config(
            base_config_path=base_config_path,
            cell_root=cell_root,
            memory_enabled=memory_enabled,
        )
        command = _build_execute_command(
            args=args,
            provider_profiles_path=provider_profiles_path,
            cell_root=cell_root,
            cell_config_path=cell_config_path,
            provider_id=provider_id,
            max_iterations=max_iterations,
        )
        cells.append(
            IterationSweepCell(
                cell_id=cell_id,
                max_iterations=max_iterations,
                summarizer_provider=provider_id,
                critic_provider=provider_id,
                judge_provider=DEEPSEEK_JUDGE_PROVIDER_ID,
                memory_enabled=memory_enabled,
                config_path=str(cell_config_path),
                output_root=str((cell_root / "runs").resolve()),
                checkpoint_path=str((cell_root / "checkpoint.json").resolve()),
                log_path=str((cell_root / "execute.log").resolve()),
                command=command,
            )
        )
    return tuple(cells)


def _quote_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def _write_plan(
    *,
    args: argparse.Namespace,
    output_root: Path,
    provider_profiles_path: Path,
    cells: tuple[IterationSweepCell, ...],
) -> Path:
    plan_path = output_root / "max_iter_sweep_plan.json"
    payload = {
        "generated_at_utc": _utc_now(),
        "score_hierarchy": {
            "primary": PRIMARY_METRIC,
            "exploratory": list(METRIC_KEYS[1:]),
        },
        "protocol": {
            "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
            "sample_seed": args.sample_seed,
            "max_samples_per_cell": args.max_samples,
            "duration_preset": args.duration_preset,
            "target_seconds": DURATION_PRESET_SECONDS[args.duration_preset],
            "thinking_mode": args.thinking_mode,
            "memory_enabled": not bool(args.disable_memory),
            "max_iterations": list(args.max_iterations_list),
            "transcript_only": True,
            "embedding_mode": "disabled",
            "vllm_model": args.vllm_model,
            "vllm_base_url": args.vllm_base_url,
            "vllm_default_max_tokens": _optional_positive_int(args.vllm_max_tokens),
            "deepseek_model": args.deepseek_model,
            "deepseek_thinking_mode": "disabled",
            "deepseek_default_max_tokens": args.deepseek_max_tokens,
        },
        "provider_profiles_path": str(provider_profiles_path),
        "cells": [
            {
                **asdict(cell),
                "command_display": _quote_command(cell.command),
            }
            for cell in cells
        ],
    }
    _write_json(plan_path, payload)
    return plan_path.resolve()


def _load_cells_from_plan(plan_path: Path) -> tuple[IterationSweepCell, ...]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list):
        raise ValueError(f"Missing cells in sweep plan: {plan_path}")
    cells: list[IterationSweepCell] = []
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, dict):
            continue
        raw_command = raw_cell.get("command", ())
        command = tuple(str(part) for part in raw_command)
        cells.append(
            IterationSweepCell(
                cell_id=str(raw_cell["cell_id"]),
                max_iterations=int(raw_cell["max_iterations"]),
                summarizer_provider=str(raw_cell["summarizer_provider"]),
                critic_provider=str(raw_cell["critic_provider"]),
                judge_provider=str(raw_cell["judge_provider"]),
                memory_enabled=bool(raw_cell["memory_enabled"]),
                config_path=str(raw_cell["config_path"]),
                output_root=str(raw_cell["output_root"]),
                checkpoint_path=str(raw_cell["checkpoint_path"]),
                log_path=str(raw_cell["log_path"]),
                command=command,
            )
        )
    return tuple(cells)


def _latest_cell_report(cell: IterationSweepCell) -> Path | None:
    output_root = Path(cell.output_root)
    if not output_root.exists():
        return None
    reports = sorted(
        output_root.glob("*/report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0].resolve() if reports else None


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_metric(sample: dict[str, Any], metric_name: str) -> float:
    metrics = sample.get("metrics")
    if not isinstance(metrics, dict):
        return 0.0
    return float(metrics.get(metric_name, 0.0) or 0.0)


def _retained_samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    samples = report.get("samples") if isinstance(report.get("samples"), list) else []
    return [
        sample
        for sample in samples
        if bool(sample.get("counts_toward_target", True))
        and str(sample.get("status", "")) != "skipped"
    ]


def _summarize_cell(
    *,
    cell: IterationSweepCell,
    report_path: Path,
) -> dict[str, Any]:
    report = _load_report(report_path)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    retained_samples = _retained_samples(report)
    tolerance_ratio = float(report.get("duration_tolerance_ratio", 0.0) or 0.0)
    duration_delta_values = [
        abs(
            float(sample.get("candidate_seconds", 0.0) or 0.0)
            - float(sample.get("target_seconds", 0.0) or 0.0)
        )
        for sample in retained_samples
    ]
    within_tolerance = [
        sample
        for sample in retained_samples
        if abs(
            float(sample.get("candidate_seconds", 0.0) or 0.0)
            - float(sample.get("target_seconds", 0.0) or 0.0)
        )
        <= float(sample.get("target_seconds", 0.0) or 0.0) * tolerance_ratio
    ]
    metric_values = {
        metric_name: [_sample_metric(sample, metric_name) for sample in retained_samples]
        for metric_name in METRIC_KEYS
    }
    return {
        "cell_id": cell.cell_id,
        "max_iterations": cell.max_iterations,
        "memory_enabled": cell.memory_enabled,
        "status": report.get("status", ""),
        "report_path": str(report_path),
        "results_path": report.get("results_path", ""),
        "sample_artifacts_root": report.get("sample_artifacts_root", ""),
        "n": int(summary.get("executed_count", len(retained_samples)) or 0),
        "retained_n": int(summary.get("sample_count", len(retained_samples)) or 0),
        "target_n": int(summary.get("target_sample_count", 0) or 0),
        "success_count": int(summary.get("success_count", 0) or 0),
        "failure_count": int(summary.get("failure_count", 0) or 0),
        "replacement_count": int(summary.get("replacement_count", 0) or 0),
        "within_tolerance_rate": (
            len(within_tolerance) / len(retained_samples) if retained_samples else 0.0
        ),
        "mean_absolute_duration_delta_seconds": _mean(duration_delta_values),
        "mean_iterations_run": _mean(
            float(sample.get("iterations_run", 0.0) or 0.0)
            for sample in retained_samples
        ),
        "mean_tool_invocations": _mean(
            float(sample.get("tool_invocations", 0.0) or 0.0)
            for sample in retained_samples
        ),
        "mean_factualness": _mean(metric_values["factualness"]),
        "mean_factualness_ci95": mean_confidence_interval_95(
            metric_values["factualness"],
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        "mean_naturalness": _mean(metric_values["naturalness"]),
        "mean_speaker_grammar_similarity": _mean(
            metric_values["speaker_grammar_similarity"]
        ),
        "mean_composite": _mean(metric_values["composite"]),
    }


def _aggregate_reports(
    *,
    output_root: Path,
    cells: tuple[IterationSweepCell, ...],
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    missing_cells: list[str] = []
    for cell in cells:
        report_path = _latest_cell_report(cell)
        if report_path is None:
            missing_cells.append(cell.cell_id)
            continue
        summaries.append(_summarize_cell(cell=cell, report_path=report_path))
    summaries.sort(key=lambda item: int(item["max_iterations"]))
    return {
        "generated_at_utc": _utc_now(),
        "primary_metric": PRIMARY_METRIC,
        "metrics": list(METRIC_KEYS),
        "output_root": str(output_root),
        "missing_cells": missing_cells,
        "cells": summaries,
    }


def _format_ci(bounds: tuple[float, float] | None) -> str:
    return format_confidence_interval(bounds)


def _render_markdown_report(aggregate: dict[str, Any]) -> str:
    rows: list[list[object]] = []
    for cell in aggregate["cells"]:
        rows.append(
            [
                cell["max_iterations"],
                cell["n"],
                f"{cell['mean_factualness']:.3f}",
                _format_ci(cell["mean_factualness_ci95"]),
                f"{cell['mean_naturalness']:.3f}",
                f"{cell['mean_speaker_grammar_similarity']:.3f}",
                f"{cell['mean_composite']:.3f}",
                f"{cell['within_tolerance_rate']:.3f}",
                f"{cell['mean_absolute_duration_delta_seconds']:.1f}",
                f"{cell['mean_iterations_run']:.1f}",
                f"{cell['mean_tool_invocations']:.1f}",
                f"{cell['success_count']}/{cell['n']}",
                cell["replacement_count"],
            ]
        )
    ranked = sorted(
        aggregate["cells"],
        key=lambda item: float(item.get("mean_factualness", 0.0) or 0.0),
        reverse=True,
    )
    best_line = "No completed max-iteration cells were found."
    if ranked:
        best = ranked[0]
        best_line = (
            f"Highest factualness in this sweep: max_iterations="
            f"{best['max_iterations']} at {best['mean_factualness']:.3f}."
        )
    table = render_markdown_table(
        [
            "max_iter",
            "n",
            "factual",
            "factual_95_ci",
            "natural",
            "speaker_grammar",
            "composite",
            "within_tol",
            "abs_delta_s",
            "iters_run",
            "tools",
            "success",
            "replaced",
        ],
        rows,
    )
    return "\n".join(
        [
            "# SPoRC Max-Iteration Sweep",
            "",
            (
                "Primary automatic endpoint: DeepSeek V4 Pro non-thinking "
                "factualness."
            ),
            "",
            best_line,
            "",
            "Graph: `max_iter_sweep.svg`",
            "",
            table,
            "",
        ]
    )


def _write_csv(output_root: Path, aggregate: dict[str, Any]) -> Path:
    csv_path = output_root / "max_iter_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "max_iterations",
        "n",
        "mean_factualness",
        "factualness_ci95_lower",
        "factualness_ci95_upper",
        "mean_naturalness",
        "mean_speaker_grammar_similarity",
        "mean_composite",
        "within_tolerance_rate",
        "mean_absolute_duration_delta_seconds",
        "mean_iterations_run",
        "mean_tool_invocations",
        "success_count",
        "failure_count",
        "replacement_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for cell in aggregate["cells"]:
            ci95 = cell.get("mean_factualness_ci95")
            lower = ci95[0] if isinstance(ci95, list | tuple) and len(ci95) == 2 else ""
            upper = ci95[1] if isinstance(ci95, list | tuple) and len(ci95) == 2 else ""
            writer.writerow(
                {
                    "max_iterations": cell["max_iterations"],
                    "n": cell["n"],
                    "mean_factualness": cell["mean_factualness"],
                    "factualness_ci95_lower": lower,
                    "factualness_ci95_upper": upper,
                    "mean_naturalness": cell["mean_naturalness"],
                    "mean_speaker_grammar_similarity": (
                        cell["mean_speaker_grammar_similarity"]
                    ),
                    "mean_composite": cell["mean_composite"],
                    "within_tolerance_rate": cell["within_tolerance_rate"],
                    "mean_absolute_duration_delta_seconds": (
                        cell["mean_absolute_duration_delta_seconds"]
                    ),
                    "mean_iterations_run": cell["mean_iterations_run"],
                    "mean_tool_invocations": cell["mean_tool_invocations"],
                    "success_count": cell["success_count"],
                    "failure_count": cell["failure_count"],
                    "replacement_count": cell["replacement_count"],
                }
            )
    return csv_path.resolve()


def _svg_point(
    *,
    x_index: int,
    values_count: int,
    value: float,
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
) -> tuple[float, float]:
    if values_count <= 1:
        x_coord = plot_left + plot_width / 2.0
    else:
        x_coord = plot_left + (plot_width * x_index / (values_count - 1))
    y_coord = plot_top + (1.0 - max(0.0, min(1.0, value))) * plot_height
    return x_coord, y_coord


def _render_svg(aggregate: dict[str, Any]) -> str:
    cells = list(aggregate["cells"])
    width = 900
    height = 500
    plot_left = 72.0
    plot_top = 38.0
    plot_width = 720.0
    plot_height = 360.0
    styles = {
        "factualness": "#1f77b4",
        "naturalness": "#2ca02c",
        "speaker_grammar_similarity": "#d62728",
        "composite": "#6b4c9a",
    }
    labels = {
        "factualness": "factualness",
        "naturalness": "naturalness",
        "speaker_grammar_similarity": "speaker grammar",
        "composite": "composite",
    }
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{plot_left}" y="22" font-family="Arial" '
            'font-size="16" font-weight="700">'
            "judge performance by max_iterations</text>"
        ),
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_coord = plot_top + (1.0 - tick) * plot_height
        lines.extend(
            [
                (
                    f'<line x1="{plot_left}" y1="{y_coord:.1f}" '
                    f'x2="{plot_left + plot_width}" y2="{y_coord:.1f}" '
                    'stroke="#d8d8d8" stroke-width="1"/>'
                ),
                (
                    f'<text x="{plot_left - 12}" y="{y_coord + 4:.1f}" '
                    'font-family="Arial" font-size="11" text-anchor="end">'
                    f"{tick:.2f}</text>"
                ),
            ]
        )
    lines.extend(
        [
            (
                f'<line x1="{plot_left}" y1="{plot_top}" '
                f'x2="{plot_left}" y2="{plot_top + plot_height}" '
                'stroke="#222" stroke-width="1.2"/>'
            ),
            (
                f'<line x1="{plot_left}" y1="{plot_top + plot_height}" '
                f'x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" '
                'stroke="#222" stroke-width="1.2"/>'
            ),
        ]
    )
    for index, cell in enumerate(cells):
        x_coord, _ = _svg_point(
            x_index=index,
            values_count=len(cells),
            value=0.0,
            plot_left=plot_left,
            plot_top=plot_top,
            plot_width=plot_width,
            plot_height=plot_height,
        )
        lines.extend(
            [
                (
                    f'<line x1="{x_coord:.1f}" y1="{plot_top + plot_height}" '
                    f'x2="{x_coord:.1f}" y2="{plot_top + plot_height + 5}" '
                    'stroke="#222" stroke-width="1"/>'
                ),
                (
                    f'<text x="{x_coord:.1f}" y="{plot_top + plot_height + 22}" '
                    'font-family="Arial" font-size="12" text-anchor="middle">'
                    f'{cell["max_iterations"]}</text>'
                ),
            ]
        )
    for metric_name in METRIC_KEYS:
        points: list[tuple[float, float]] = []
        for index, cell in enumerate(cells):
            key = f"mean_{metric_name}"
            points.append(
                _svg_point(
                    x_index=index,
                    values_count=len(cells),
                    value=float(cell.get(key, 0.0) or 0.0),
                    plot_left=plot_left,
                    plot_top=plot_top,
                    plot_width=plot_width,
                    plot_height=plot_height,
                )
            )
        color = styles[metric_name]
        if len(points) >= 2:
            polyline_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            lines.append(
                f'<polyline points="{polyline_points}" fill="none" '
                f'stroke="{color}" stroke-width="2.4"/>'
            )
        for x_coord, y_coord in points:
            lines.append(
                f'<circle cx="{x_coord:.1f}" cy="{y_coord:.1f}" r="4" '
                f'fill="{color}"/>'
            )
    legend_x = plot_left + plot_width + 24
    legend_y = plot_top + 18
    for index, metric_name in enumerate(METRIC_KEYS):
        y_coord = legend_y + index * 24
        color = styles[metric_name]
        lines.extend(
            [
                (
                    f'<line x1="{legend_x}" y1="{y_coord}" '
                    f'x2="{legend_x + 20}" y2="{y_coord}" '
                    f'stroke="{color}" stroke-width="2.4"/>'
                ),
                (
                    f'<text x="{legend_x + 28}" y="{y_coord + 4}" '
                    'font-family="Arial" font-size="12">'
                    f"{escape(labels[metric_name])}</text>"
                ),
            ]
        )
    lines.extend(
        [
            (
                f'<text x="{plot_left + plot_width / 2:.1f}" '
                f'y="{height - 28}" font-family="Arial" font-size="13" '
                'text-anchor="middle">max_iterations</text>'
            ),
            (
                f'<text x="18" y="{plot_top + plot_height / 2:.1f}" '
                'font-family="Arial" font-size="13" text-anchor="middle" '
                'transform="rotate(-90 18 '
                f'{plot_top + plot_height / 2:.1f})">judge score</text>'
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_reports(
    *,
    output_root: Path,
    aggregate: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    json_path = output_root / "max_iter_sweep_report.json"
    markdown_path = output_root / "max_iter_sweep_report.md"
    svg_path = output_root / "max_iter_sweep.svg"
    _write_json(json_path, aggregate)
    markdown_path.write_text(_render_markdown_report(aggregate), encoding="utf-8")
    csv_path = _write_csv(output_root, aggregate)
    svg_path.write_text(_render_svg(aggregate), encoding="utf-8")
    return json_path.resolve(), markdown_path.resolve(), csv_path, svg_path.resolve()


def _cell_report_completed(report_path: Path | None, *, requested_samples: int) -> bool:
    if report_path is None:
        return False
    try:
        payload = _load_report(report_path)
    except (OSError, json.JSONDecodeError):
        return False
    if str(payload.get("status", "")) != "completed":
        return False
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return False
    sample_count = int(summary.get("sample_count", 0) or 0)
    return sample_count >= max(0, int(requested_samples))


def _command_with_resume(command: tuple[str, ...]) -> tuple[str, ...]:
    if "--resume-from-checkpoint" in command:
        return command
    return (*command, "--resume-from-checkpoint")


def _run_cell(cell: IterationSweepCell, *, command: tuple[str, ...]) -> int:
    log_path = Path(cell.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"\n[{_utc_now()}] command: {_quote_command(command)}\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is not None:
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                print(line, end="")
        return process.wait()


def _resolve_output_root(raw_output_root: str) -> Path:
    if raw_output_root:
        return Path(raw_output_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (DEFAULT_OUTPUT_ROOT / stamp).resolve()


def build_parser() -> argparse.ArgumentParser:
    """Build the max-iteration sweep CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run SPoRC over a max-iterations sweep and graph judge scores."
    )
    parser.add_argument("--mode", choices=("plan", "run", "report"), default="plan")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dataset-manifest", default=str(DEFAULT_DATASET_MANIFEST))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--duration-preset", choices=tuple(DURATION_PRESET_SECONDS), default="5m")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--max-iterations-list",
        type=_parse_max_iterations,
        default=_parse_max_iterations("1,2,3,4,5"),
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("thinking", "nonthinking"),
        default="thinking",
    )
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument(
        "--vllm-base-url",
        default=_first_env(
            (
                "CARD_FRAMEWORK_VLLM_BASE_URL",
                "CARD_FRAMEWORK_VLLM_URL",
                "VLLM_BASE_URL",
                "VLLM_URL",
            ),
            "http://127.0.0.1:8000/v1",
        ),
    )
    parser.add_argument(
        "--vllm-api-key",
        default=_first_env(("CARD_FRAMEWORK_VLLM_API_KEY", "VLLM_API_KEY"), "EMPTY"),
    )
    parser.add_argument(
        "--vllm-model",
        default=_first_env(
            ("CARD_FRAMEWORK_VLLM_MODEL", "VLLM_MODEL"),
            "Qwen/Qwen3.6-27B",
        ),
    )
    parser.add_argument(
        "--vllm-max-tokens",
        type=int,
        default=0,
        help="Set a vLLM max_tokens request budget. Use 0 to omit max_tokens.",
    )
    parser.add_argument("--vllm-request-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--deepseek-base-url",
        default=_first_env(("DEEPSEEK_BASE_URL",), "https://api.deepseek.com"),
    )
    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--deepseek-max-tokens", type=int, default=384_000)
    parser.add_argument("--deepseek-temperature", type=float, default=0.0)
    parser.add_argument("--deepseek-request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--summarizer-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--critic-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--retrieval-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--index-timeout-seconds", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the max-iteration sweep workflow."""
    args = build_parser().parse_args(argv)
    output_root = _resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    existing_plan_path = output_root / "max_iter_sweep_plan.json"
    if args.mode == "report" and existing_plan_path.exists():
        plan_path = existing_plan_path.resolve()
        cells = _load_cells_from_plan(plan_path)
        print(f"Loaded plan: {plan_path}")
    else:
        provider_profiles_path = _write_provider_profiles(args, output_root)
        cells = _build_cells(
            args,
            output_root=output_root,
            provider_profiles_path=provider_profiles_path,
        )
        plan_path = _write_plan(
            args=args,
            output_root=output_root,
            provider_profiles_path=provider_profiles_path,
            cells=cells,
        )
        print(f"Plan written: {plan_path}")

    if args.mode == "run":
        requested_samples = max(0, int(args.max_samples))
        for cell in cells:
            latest_report = _latest_cell_report(cell)
            if _cell_report_completed(
                latest_report,
                requested_samples=requested_samples,
            ):
                print(f"Skipping completed cell: {cell.cell_id}")
                continue
            command = cell.command
            if latest_report is not None and Path(cell.checkpoint_path).exists():
                command = _command_with_resume(command)
                print(f"Resuming cell: {cell.cell_id}")
            else:
                print(f"Running cell: {cell.cell_id}")
            exit_code = _run_cell(cell, command=command)
            if exit_code != 0:
                print(f"Cell failed with exit code {exit_code}: {cell.cell_id}")
                return exit_code

    if args.mode in {"run", "report"}:
        aggregate = _aggregate_reports(output_root=output_root, cells=cells)
        json_path, markdown_path, csv_path, svg_path = _write_reports(
            output_root=output_root,
            aggregate=aggregate,
        )
        print(f"Max-iteration JSON report: {json_path}")
        print(f"Max-iteration Markdown report: {markdown_path}")
        print(f"Max-iteration CSV: {csv_path}")
        print(f"Max-iteration SVG graph: {svg_path}")
        if aggregate.get("missing_cells"):
            print(f"Missing cells: {', '.join(aggregate['missing_cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
