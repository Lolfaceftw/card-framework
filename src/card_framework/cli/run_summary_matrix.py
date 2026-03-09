"""Run the default summary workflow across ordered summarizer/critic pairs.

This helper shells out to ``card_framework.cli.setup_and_run`` for each ordered
model pair so it reuses the repository's normal bootstrap, stage selection,
merged live-draft stage-2/stage-3 flow, and summarizer/critic orchestration
path. The helper disables only stage-4 interjector output, then copies the
resulting ``summary.xml`` into a pair-specific artifact file whose name starts
with ``<summarizer>_<critic>-summary.xml``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from card_framework.shared.paths import REPO_ROOT

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "summary_matrix"
SUMMARY_SOURCE_PATH = REPO_ROOT / "summary.xml"
VLLM_HOST_ENV = "SUMMARY_MATRIX_VLLM_HOST"
DEEPSEEK_API_KEY_ENV = "SUMMARY_MATRIX_DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL_ENV = "SUMMARY_MATRIX_DEEPSEEK_BASE_URL"
SETUP_AND_RUN_MODULE = "card_framework.cli.setup_and_run"
COMMON_SUMMARY_ONLY_OVERRIDES: tuple[str, ...] = (
    "audio.interjector.enabled=false",
)
_CONSOLE_SAFE_TRANSLATIONS = str.maketrans(
    {
        "\u2192": "->",
        "\u2014": "-",
        "\u2013": "-",
        "\u2026": "...",
    }
)


def _format_hydra_override(*, key: str, value: str, append: bool = False) -> str:
    """Return one Hydra CLI override, optionally using append syntax."""
    prefix = "+" if append else ""
    return f"{prefix}{key}={value}"


@dataclass(frozen=True, slots=True)
class SummaryMatrixError(RuntimeError):
    """Represent one configuration or execution error in the matrix runner."""

    message: str

    def __str__(self) -> str:
        """Return the user-facing error text."""
        return self.message


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Describe one model/provider configuration for matrix runs."""

    slug: str
    model_name: str
    provider_target: str
    port: int | None = None

    def build_provider_overrides(self, *, base_key: str) -> list[str]:
        """Return Hydra overrides for one provider slot on one run."""
        append_missing_keys = base_key.startswith("stage_llm.")
        overrides = [
            _format_hydra_override(
                key=f"{base_key}._target_",
                value=self.provider_target,
                append=append_missing_keys,
            )
        ]
        if self.provider_target == "card_framework.providers.vllm_provider.VLLMProvider":
            if self.port is None:
                raise SummaryMatrixError(
                    "vLLM model profiles require a configured port."
                )
            overrides.extend(
                [
                    _format_hydra_override(
                        key=f"{base_key}.base_url",
                        value=build_vllm_base_url_interpolation(self.port),
                        append=append_missing_keys,
                    ),
                    _format_hydra_override(
                        key=f"{base_key}.api_key",
                        value="EMPTY",
                        append=append_missing_keys,
                    ),
                ]
            )
            return overrides

        overrides.extend(
            [
                _format_hydra_override(
                    key=f"{base_key}.api_key",
                    value=f"${{oc.env:{DEEPSEEK_API_KEY_ENV}}}",
                    append=append_missing_keys,
                ),
                _format_hydra_override(
                    key=f"{base_key}.model",
                    value=self.model_name,
                    append=append_missing_keys,
                ),
                _format_hydra_override(
                    key=f"{base_key}.base_url",
                    value=f"${{oc.env:{DEEPSEEK_BASE_URL_ENV}}}",
                    append=append_missing_keys,
                ),
            ]
        )
        return overrides


@dataclass(frozen=True, slots=True)
class ResolvedInput:
    """Capture the non-interactive input source chosen for the matrix run."""

    mode: str
    path: Path


@dataclass(frozen=True, slots=True)
class PairResult:
    """Store one matrix run outcome for manifest reporting."""

    summarizer_slug: str
    summarizer_model: str
    critic_slug: str
    critic_model: str
    status: str
    summary_path: str
    log_path: str
    command: list[str]
    input_mode: str
    input_path: str
    returncode: int
    error: str = ""


DEFAULT_VLLM_MODELS: tuple[ModelProfile, ...] = (
    ModelProfile(
        slug="qwen3_5_27b",
        model_name="Qwen/Qwen3.5-27B",
        provider_target="card_framework.providers.vllm_provider.VLLMProvider",
        port=8000,
    ),
    ModelProfile(
        slug="qwen3_5_9b",
        model_name="Qwen/Qwen3.5-9B",
        provider_target="card_framework.providers.vllm_provider.VLLMProvider",
        port=8001,
    ),
    ModelProfile(
        slug="qwen3_5_4b",
        model_name="Qwen/Qwen3.5-4B",
        provider_target="card_framework.providers.vllm_provider.VLLMProvider",
        port=8002,
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the summary matrix runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run card_framework.cli.setup_and_run across the built-in "
            "summarizer/critic model matrix and save one summary XML per "
            "ordered pair."
        )
    )
    parser.add_argument(
        "--vllm-host",
        required=True,
        help="Host or IP serving the built-in Qwen vLLM endpoints.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--audio-path",
        help=(
            "Audio file for stage-1 runs. Use --transcript-path instead if you "
            "want to reuse an existing transcript for faster stage-2 runs."
        ),
    )
    input_group.add_argument(
        "--transcript-path",
        help=(
            "Transcript JSON for stage-2 runs. The transcript must already point "
            "at a usable speaker-sample manifest for non-interactive execution."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for per-pair summaries, logs, and the manifest. "
            "Defaults to artifacts/summary_matrix/<timestamp>."
        ),
    )
    parser.add_argument(
        "--deepseek-api-key",
        default="",
        help=(
            "Optional DeepSeek API key. When provided, the matrix adds a DeepSeek "
            "model to the run pool without exposing the key in child "
            "process arguments."
        ),
    )
    parser.add_argument(
        "--deepseek-model",
        default="deepseek-chat",
        help="DeepSeek model to add when --deepseek-api-key is provided.",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default="https://api.deepseek.com",
        help="Base URL for the optional DeepSeek provider.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra Hydra override forwarded to every setup-and-run invocation. "
            "Repeat this flag for multiple values."
        ),
    )
    parser.add_argument(
        "--uv-executable",
        default="uv",
        help="uv executable to use for child setup-and-run calls.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed model pair.",
    )
    return parser.parse_args(argv)


def utc_now_compact() -> str:
    """Return a UTC timestamp suitable for artifact folder names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_path_input(raw_value: str, *, field_name: str) -> Path:
    """Resolve one file or directory path relative to the repository root."""
    normalized = raw_value.strip().strip('"').strip("'")
    if not normalized:
        raise SummaryMatrixError(f"{field_name} must not be empty.")
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate


def resolve_existing_file(raw_value: str, *, field_name: str) -> Path:
    """Resolve one existing file path."""
    candidate = resolve_path_input(raw_value, field_name=field_name)
    if not candidate.exists():
        raise SummaryMatrixError(f"{field_name} does not exist: {candidate}")
    if not candidate.is_file():
        raise SummaryMatrixError(f"{field_name} is not a file: {candidate}")
    return candidate


def resolve_output_dir(raw_value: str | None) -> Path:
    """Resolve and create the output directory for matrix artifacts."""
    if raw_value is None or not raw_value.strip():
        output_dir = (DEFAULT_OUTPUT_ROOT / utc_now_compact()).resolve()
    else:
        output_dir = resolve_path_input(raw_value, field_name="Output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _discover_existing_transcript_path() -> Path | None:
    """Return the preferred reusable transcript JSON path when one exists."""
    root_transcript = (REPO_ROOT / "transcript.json").resolve()
    if root_transcript.is_file():
        return root_transcript

    root_candidates = sorted(
        (
            path.resolve()
            for path in REPO_ROOT.glob("*.transcript.json")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if root_candidates:
        return root_candidates[0]

    artifacts_dir = REPO_ROOT / "artifacts" / "transcripts"
    artifact_candidates = sorted(
        (
            path.resolve()
            for path in artifacts_dir.glob("*.transcript.json")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if artifact_candidates:
        return artifact_candidates[0]
    return None


def _load_transcript_metadata(transcript_path: Path) -> dict[str, Any]:
    """Load transcript metadata using BOM-tolerant decoding."""
    try:
        payload = json.loads(transcript_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryMatrixError(
            f"Failed to read transcript metadata: {transcript_path}"
        ) from exc
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return metadata


def transcript_has_usable_speaker_manifest(transcript_path: Path) -> bool:
    """Return whether transcript metadata points at an existing sample manifest."""
    metadata = _load_transcript_metadata(transcript_path)
    raw_manifest = str(metadata.get("speaker_samples_manifest_path", "")).strip()
    if not raw_manifest:
        return False
    manifest_path = Path(raw_manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()
    return manifest_path.is_file()


def resolve_input_source(args: argparse.Namespace) -> ResolvedInput:
    """Resolve a non-interactive input source for the matrix runner."""
    if args.audio_path:
        return ResolvedInput(
            mode="audio",
            path=resolve_existing_file(args.audio_path, field_name="Audio path"),
        )

    if args.transcript_path:
        transcript_path = resolve_existing_file(
            args.transcript_path,
            field_name="Transcript path",
        )
        if not transcript_has_usable_speaker_manifest(transcript_path):
            raise SummaryMatrixError(
                "Transcript path does not include a usable "
                "metadata.speaker_samples_manifest_path. Pass --audio-path to run "
                "stage-1 or use a stage-2 transcript that already has speaker samples."
            )
        return ResolvedInput(mode="transcript", path=transcript_path)

    discovered = _discover_existing_transcript_path()
    if discovered is None:
        raise SummaryMatrixError(
            "No reusable transcript was found. Pass --audio-path for stage-1 or "
            "--transcript-path for stage-2."
        )
    if not transcript_has_usable_speaker_manifest(discovered):
        raise SummaryMatrixError(
            "The auto-discovered transcript does not include a usable "
            "speaker-sample manifest. Pass --audio-path to let the default wrapper "
            "generate fresh stage-1 artifacts, or point --transcript-path at a "
            "stage-2 transcript that already has speaker samples."
        )
    return ResolvedInput(mode="transcript", path=discovered)


def build_vllm_base_url_interpolation(port: int) -> str:
    """Build an env-backed vLLM base URL override for Hydra."""
    return f"http://${{oc.env:{VLLM_HOST_ENV}}}:{port}/v1"


def build_model_profiles(
    *,
    include_deepseek: bool,
    deepseek_model: str,
) -> tuple[ModelProfile, ...]:
    """Return the model pool used to build ordered model pairs."""
    models = list(DEFAULT_VLLM_MODELS)
    if include_deepseek:
        models.append(
            ModelProfile(
                slug=slugify_model_name(deepseek_model),
                model_name=deepseek_model,
                provider_target="card_framework.providers.deepseek_provider.DeepSeekProvider",
            )
        )
    return tuple(models)


def build_model_pairs(
    model_profiles: tuple[ModelProfile, ...],
) -> tuple[tuple[ModelProfile, ModelProfile], ...]:
    """Return ordered summarizer/critic model pairs including self-pairs."""
    return tuple(product(model_profiles, repeat=2))


def slugify_model_name(model_name: str) -> str:
    """Convert one model name into a filesystem-safe lowercase slug."""
    parts: list[str] = []
    current: list[str] = []
    for char in model_name.strip().lower():
        if char.isalnum():
            current.append(char)
            continue
        if current:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return "_".join(parts) if parts else "model"


def build_pair_stem(summarizer: ModelProfile, critic: ModelProfile) -> str:
    """Return the common filename stem for one summarizer/critic pair."""
    return f"{summarizer.slug}_{critic.slug}"


def build_summary_filename(
    summarizer: ModelProfile,
    critic: ModelProfile,
) -> str:
    """Return the required output summary filename for one pair."""
    return f"{build_pair_stem(summarizer, critic)}-summary.xml"


def _path_for_override(path: Path) -> str:
    """Return an absolute path string safe for Hydra CLI overrides."""
    return path.resolve().as_posix()


def _contains_override_key(overrides: list[str], key: str) -> bool:
    """Return whether an override list already defines one key."""
    expected_prefix = f"{key}="
    return any(override.startswith(expected_prefix) for override in overrides)


def _print_console_line(message: str) -> None:
    """Print one operator-facing line safely for the active stdout encoding."""
    _write_console_chunk(f"{message}\n")


def _write_console_chunk(text: str) -> None:
    """Write one operator-facing chunk safely for the active stdout encoding."""
    sanitized = text.translate(_CONSOLE_SAFE_TRANSLATIONS)
    encoding = getattr(sys.stdout, "encoding", None)
    if encoding:
        sanitized = sanitized.encode(encoding, errors="replace").decode(encoding)
    sys.stdout.write(sanitized)
    sys.stdout.flush()


def _run_child_command_streaming(
    *,
    command: list[str],
    env: dict[str, str],
) -> tuple[int, str]:
    """Run one child command while streaming its output and capturing a log copy."""
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
    )
    stdout = process.stdout
    if stdout is None:
        raise SummaryMatrixError("Failed to capture child process output stream.")

    captured_chunks: list[str] = []
    try:
        for chunk in iter(lambda: stdout.read(1), ""):
            captured_chunks.append(chunk)
            _write_console_chunk(chunk)
    finally:
        stdout.close()
    return process.wait(), "".join(captured_chunks)


def _ensure_console_line_break(text: str) -> None:
    """Terminate the current console line when streamed child output ended mid-line."""
    if text and not text.endswith(("\n", "\r")):
        _write_console_chunk("\n")


def build_setup_command(
    *,
    args: argparse.Namespace,
    resolved_input: ResolvedInput,
    summarizer: ModelProfile,
    critic: ModelProfile,
    output_dir: Path,
) -> list[str]:
    """Build one setup-and-run subprocess command for a model pair."""
    pair_stem = build_pair_stem(summarizer, critic)
    loop_memory_path = output_dir / f"{pair_stem}-loop-memory.json"

    forwarded_overrides = list(args.override)
    if resolved_input.mode == "transcript":
        if not _contains_override_key(forwarded_overrides, "pipeline.start_stage"):
            forwarded_overrides.append("pipeline.start_stage=stage-2")
        if not _contains_override_key(forwarded_overrides, "transcript_path"):
            forwarded_overrides.append(
                f"transcript_path={_path_for_override(resolved_input.path)}"
            )

    overrides: list[str] = [
        *forwarded_overrides,
        *COMMON_SUMMARY_ONLY_OVERRIDES,
        f"orchestrator.loop_memory.artifact_path={_path_for_override(loop_memory_path)}",
        *summarizer.build_provider_overrides(base_key="llm"),
        *critic.build_provider_overrides(base_key="stage_llm.critic"),
    ]

    command = [
        args.uv_executable,
        "run",
        "python",
        "-m",
        SETUP_AND_RUN_MODULE,
    ]
    if resolved_input.mode == "audio":
        command.extend(["--audio-path", str(resolved_input.path)])
    for override in overrides:
        command.extend(["--override", override])
    return command


def build_child_env(args: argparse.Namespace, *, deepseek_api_key: str) -> dict[str, str]:
    """Build the environment used by child setup-and-run invocations."""
    env = dict(os.environ)
    env[VLLM_HOST_ENV] = args.vllm_host.strip()
    env[DEEPSEEK_BASE_URL_ENV] = args.deepseek_base_url.strip()
    if deepseek_api_key:
        env[DEEPSEEK_API_KEY_ENV] = deepseek_api_key
    else:
        env.pop(DEEPSEEK_API_KEY_ENV, None)
    return env


def _decode_subprocess_output(data: bytes) -> str:
    """Decode subprocess output safely for logs and failure summaries."""
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")


def _combine_process_output(completed: subprocess.CompletedProcess[bytes]) -> str:
    """Return a single log payload from stdout and stderr."""
    stdout_text = _decode_subprocess_output(completed.stdout)
    stderr_text = _decode_subprocess_output(completed.stderr)
    if stdout_text and stderr_text:
        return f"{stdout_text}\n\n[stderr]\n{stderr_text}"
    return stdout_text or stderr_text


def _tail_lines(text: str, *, line_count: int = 25) -> str:
    """Return the last few lines of a log payload."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= line_count:
        return "\n".join(lines)
    return "\n".join(lines[-line_count:])


def run_pair(
    *,
    args: argparse.Namespace,
    resolved_input: ResolvedInput,
    summarizer: ModelProfile,
    critic: ModelProfile,
    output_dir: Path,
    env: dict[str, str],
) -> PairResult:
    """Execute one summarizer/critic pair and persist its artifacts."""
    pair_stem = build_pair_stem(summarizer, critic)
    log_path = (output_dir / f"{pair_stem}.log").resolve()
    summary_path = (output_dir / build_summary_filename(summarizer, critic)).resolve()
    command = build_setup_command(
        args=args,
        resolved_input=resolved_input,
        summarizer=summarizer,
        critic=critic,
        output_dir=output_dir,
    )

    returncode, combined_output = _run_child_command_streaming(
        command=command,
        env=env,
    )
    log_path.write_text(combined_output, encoding="utf-8")
    _ensure_console_line_break(combined_output)

    if returncode != 0:
        return PairResult(
            summarizer_slug=summarizer.slug,
            summarizer_model=summarizer.model_name,
            critic_slug=critic.slug,
            critic_model=critic.model_name,
            status="failed",
            summary_path="",
            log_path=str(log_path),
            command=command,
            input_mode=resolved_input.mode,
            input_path=str(resolved_input.path),
            returncode=returncode,
            error=_tail_lines(combined_output),
        )

    if not SUMMARY_SOURCE_PATH.is_file():
        return PairResult(
            summarizer_slug=summarizer.slug,
            summarizer_model=summarizer.model_name,
            critic_slug=critic.slug,
            critic_model=critic.model_name,
            status="failed",
            summary_path="",
            log_path=str(log_path),
            command=command,
            input_mode=resolved_input.mode,
            input_path=str(resolved_input.path),
            returncode=returncode,
            error=(
                "card_framework.cli.setup_and_run completed but summary.xml was "
                "not created."
            ),
        )

    shutil.copy2(SUMMARY_SOURCE_PATH, summary_path)
    return PairResult(
        summarizer_slug=summarizer.slug,
        summarizer_model=summarizer.model_name,
        critic_slug=critic.slug,
        critic_model=critic.model_name,
        status="ok",
        summary_path=str(summary_path),
        log_path=str(log_path),
        command=command,
        input_mode=resolved_input.mode,
        input_path=str(resolved_input.path),
        returncode=returncode,
    )


def write_manifest(
    *,
    output_dir: Path,
    resolved_input: ResolvedInput,
    results: list[PairResult],
) -> Path:
    """Write one JSON manifest describing the matrix run."""
    manifest_path = (output_dir / "matrix_results.json").resolve()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "input_mode": resolved_input.mode,
        "input_path": str(resolved_input.path),
        "results": [asdict(result) for result in results],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def print_run_header(
    *,
    output_dir: Path,
    resolved_input: ResolvedInput,
    model_count: int,
    pair_count: int,
) -> None:
    """Print one concise matrix-run header."""
    _print_console_line(f"Output directory: {output_dir}")
    _print_console_line(f"Input mode: {resolved_input.mode}")
    _print_console_line(f"Input path: {resolved_input.path}")
    _print_console_line(f"Model pool: {model_count}")
    _print_console_line(f"Pairs: {pair_count}")


def print_result(
    *,
    index: int,
    total: int,
    result: PairResult,
) -> None:
    """Print one concise result line for a matrix pair."""
    prefix = f"[{index}/{total}] {result.summarizer_slug} x {result.critic_slug}"
    if result.status == "ok":
        _print_console_line(f"{prefix}: wrote {result.summary_path}")
        return
    _print_console_line(f"{prefix}: FAILED (exit={result.returncode})")
    if result.error:
        _print_console_line(result.error)


def print_pair_start(
    *,
    index: int,
    total: int,
    summarizer: ModelProfile,
    critic: ModelProfile,
) -> None:
    """Print one concise start line before streaming a pair's child output."""
    _print_console_line(
        f"[{index}/{total}] {summarizer.slug} x {critic.slug}: running..."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the summary matrix helper."""
    args = parse_args(argv)
    try:
        resolved_input = resolve_input_source(args)
        output_dir = resolve_output_dir(args.output_dir)
        deepseek_api_key = args.deepseek_api_key.strip() or os.getenv(
            "DEEPSEEK_API_KEY",
            "",
        ).strip()
        model_profiles = build_model_profiles(
            include_deepseek=bool(deepseek_api_key),
            deepseek_model=args.deepseek_model.strip() or "deepseek-chat",
        )
        model_pairs = build_model_pairs(model_profiles)
        env = build_child_env(args, deepseek_api_key=deepseek_api_key)
        total = len(model_pairs)

        if not deepseek_api_key:
            _print_console_line(
                "DeepSeek model skipped because no --deepseek-api-key was provided."
            )

        print_run_header(
            output_dir=output_dir,
            resolved_input=resolved_input,
            model_count=len(model_profiles),
            pair_count=total,
        )

        results: list[PairResult] = []
        current_index = 0
        for summarizer, critic in model_pairs:
            current_index += 1
            print_pair_start(
                index=current_index,
                total=total,
                summarizer=summarizer,
                critic=critic,
            )
            result = run_pair(
                args=args,
                resolved_input=resolved_input,
                summarizer=summarizer,
                critic=critic,
                output_dir=output_dir,
                env=env,
            )
            results.append(result)
            print_result(index=current_index, total=total, result=result)
            if result.status != "ok" and args.fail_fast:
                manifest_path = write_manifest(
                    output_dir=output_dir,
                    resolved_input=resolved_input,
                    results=results,
                )
                _print_console_line(f"Manifest written to {manifest_path}")
                return 1

        manifest_path = write_manifest(
            output_dir=output_dir,
            resolved_input=resolved_input,
            results=results,
        )
        succeeded = sum(1 for result in results if result.status == "ok")
        failed = len(results) - succeeded
        _print_console_line(f"Manifest written to {manifest_path}")
        _print_console_line(
            f"Completed {len(results)} pairs: {succeeded} succeeded, {failed} failed."
        )
        return 0 if failed == 0 else 1
    except SummaryMatrixError as exc:
        _print_console_line(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
