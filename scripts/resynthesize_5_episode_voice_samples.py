"""Resynthesize arbitrary text with the packaged five-episode voice samples."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from omegaconf import OmegaConf

from card_framework.audio_pipeline.factory import build_voice_clone_provider
from card_framework.audio_pipeline.voice_clone_contracts import VoiceCloneProvider
from card_framework.shared.config_loading import load_runtime_config
from card_framework.shared.events import event_bus
from card_framework.shared.paths import (
    REPO_DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    SPORC_HUMAN_EVAL_ASSETS_DIR,
)

DEFAULT_INVENTORY_PATH = (
    SPORC_HUMAN_EVAL_ASSETS_DIR
    / "descript_voice_recording_kit"
    / "voice_sample_inventory.csv"
).resolve()
DEFAULT_VOICE_SAMPLES_ROOT = (SPORC_HUMAN_EVAL_ASSETS_DIR / "voice_samples").resolve()
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "five_episode_voice_resynthesis"
).resolve()
MANIFEST_FILENAME_TEMPLATE = "manifest_{text_hash}.json"


@dataclass(slots=True, frozen=True)
class VoiceSampleSpec:
    """Describe one five-episode speaker reference sample."""

    episode_order: int
    episode_title: str
    speaker: str
    sample_path: Path


@dataclass(slots=True, frozen=True)
class RenderedSample:
    """Describe one resynthesized sample output."""

    sample: VoiceSampleSpec
    output_audio_path: Path
    status: str


@dataclass(slots=True, frozen=True)
class ResynthesisResult:
    """Describe one complete text-resynthesis run."""

    output_dir: Path
    manifest_path: Path
    text_sha256: str
    rendered_samples: tuple[RenderedSample, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for five-episode voice resynthesis."""
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize input text once for every voice sample listed in the "
            "five-episode human-eval voice manifests or optional CSV inventory."
        )
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        "--text",
        help="Text to synthesize. If omitted, stdin or an interactive prompt is used.",
    )
    text_group.add_argument(
        "--text-file",
        type=Path,
        help="UTF-8 text file whose contents should be synthesized.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated WAV files and manifest. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--voice-samples-root",
        type=Path,
        default=DEFAULT_VOICE_SAMPLES_ROOT,
        help=(
            "Root containing the five episode voice_sample folders. "
            f"Default: {DEFAULT_VOICE_SAMPLES_ROOT}"
        ),
    )
    parser.add_argument(
        "--inventory-path",
        type=Path,
        help=(
            "Optional CSV inventory override. When omitted, episode manifest.json "
            "files under --voice-samples-root are used."
        ),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_DEFAULT_CONFIG_PATH,
        help=f"Runtime config YAML used for voice-clone settings. Default: {REPO_DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="DOTLIST",
        help=(
            "Additional OmegaConf dotlist override, such as "
            "audio.voice_clone.device=cpu. Repeat for multiple overrides."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Convenience override for audio.voice_clone.device.",
    )
    parser.add_argument(
        "--execution-backend",
        choices=("subprocess", "inprocess"),
        help="Convenience override for audio.voice_clone.execution_backend.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate WAV files even when matching text-hash outputs already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entrypoint."""
    args = parse_args(argv)
    provider: VoiceCloneProvider | None = None
    event_callbacks = _subscribe_cli_events()
    try:
        text = resolve_input_text(args)
        inventory_path = (
            resolve_workspace_path(args.inventory_path)
            if args.inventory_path is not None
            else None
        )
        voice_samples_root = resolve_workspace_path(args.voice_samples_root)
        config_path = resolve_workspace_path(args.config_path)
        output_dir = resolve_workspace_path(args.output_dir)
        samples = load_voice_samples(
            voice_samples_root=voice_samples_root,
            inventory_path=inventory_path,
        )
        audio_cfg = load_audio_config_for_voice_clone(
            config_path=config_path,
            overrides=tuple(args.override),
            device=args.device,
            execution_backend=args.execution_backend,
        )
        provider = build_voice_clone_provider(audio_cfg, project_root=REPO_ROOT)
        result = resynthesize_voice_samples(
            provider=provider,
            samples=samples,
            text=text,
            output_dir=output_dir,
            force=bool(args.force),
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if provider is not None:
            provider.close()
        _unsubscribe_cli_events(event_callbacks)

    rendered = sum(1 for item in result.rendered_samples if item.status == "rendered")
    skipped = sum(
        1 for item in result.rendered_samples if item.status == "skipped_existing"
    )
    print(f"Output directory: {result.output_dir}")
    print(f"Manifest: {result.manifest_path}")
    print(
        "Completed "
        f"{len(result.rendered_samples)} sample(s): {rendered} rendered, {skipped} skipped."
    )
    return 0


def resolve_workspace_path(path: Path) -> Path:
    """Resolve a CLI path relative to the repository workspace."""
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _subscribe_cli_events() -> tuple[tuple[str, Any], ...]:
    """Stream repo event-bus status messages for this standalone helper."""

    def _print_event(message: object = "", **_: object) -> None:
        normalized = str(message).strip()
        if normalized:
            print(normalized, flush=True)

    subscriptions: tuple[tuple[str, Any], ...] = (
        ("system_message", _print_event),
        ("status_message", _print_event),
    )
    for event_type, callback in subscriptions:
        event_bus.subscribe(event_type, callback)
    return subscriptions


def _unsubscribe_cli_events(subscriptions: tuple[tuple[str, Any], ...]) -> None:
    """Remove CLI event-bus subscriptions registered for this run."""
    for event_type, callback in subscriptions:
        event_bus.unsubscribe(event_type, callback)


def resolve_input_text(args: argparse.Namespace) -> str:
    """Resolve synthesis text from CLI, stdin, file, or prompt."""
    if args.text_file is not None:
        text = resolve_workspace_path(args.text_file).read_text(encoding="utf-8")
    elif args.text is not None:
        text = str(args.text)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        text = input("Enter text to synthesize: ")

    normalized = text.strip()
    if not normalized:
        raise ValueError("Synthesis text must be non-empty.")
    return normalized


def load_voice_sample_inventory(inventory_path: Path) -> tuple[VoiceSampleSpec, ...]:
    """Load the five-episode voice sample inventory CSV."""
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Voice sample inventory does not exist: {inventory_path}"
        )

    samples: list[VoiceSampleSpec] = []
    with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "episode_order",
            "episode_title",
            "speaker",
            "source_voice_sample",
        }
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "Voice sample inventory is missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                episode_order = int(str(row["episode_order"]).strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid episode_order on inventory row {row_number}."
                ) from exc
            episode_title = str(row["episode_title"]).strip()
            speaker = str(row["speaker"]).strip()
            sample_value = str(row["source_voice_sample"]).strip()
            if not episode_title or not speaker or not sample_value:
                raise ValueError(
                    f"Inventory row {row_number} must include episode title, speaker, and sample path."
                )
            sample_path = _resolve_inventory_sample_path(
                inventory_path=inventory_path,
                path_value=sample_value,
            )
            if not sample_path.exists():
                raise FileNotFoundError(
                    f"Voice sample path from inventory row {row_number} does not exist: "
                    f"{sample_path}"
                )
            samples.append(
                VoiceSampleSpec(
                    episode_order=episode_order,
                    episode_title=episode_title,
                    speaker=speaker,
                    sample_path=sample_path,
                )
            )

    if not samples:
        raise ValueError(f"Voice sample inventory has no sample rows: {inventory_path}")
    return tuple(samples)


def load_voice_samples(
    *,
    voice_samples_root: Path,
    inventory_path: Path | None = None,
) -> tuple[VoiceSampleSpec, ...]:
    """Load voice sample specs from an optional CSV or episode manifests."""
    if inventory_path is not None:
        return load_voice_sample_inventory(inventory_path)
    return load_voice_sample_manifests(voice_samples_root)


def load_voice_sample_manifests(
    voice_samples_root: Path,
) -> tuple[VoiceSampleSpec, ...]:
    """Load five-episode voice sample specs from episode manifest JSON files."""
    if not voice_samples_root.exists():
        raise FileNotFoundError(
            f"Voice samples root does not exist: {voice_samples_root}"
        )
    manifest_paths = sorted(voice_samples_root.glob("*/manifest.json"))
    if not manifest_paths:
        raise ValueError(
            f"No episode voice sample manifests found under: {voice_samples_root}"
        )

    samples: list[VoiceSampleSpec] = []
    for manifest_path in manifest_paths:
        episode_order, episode_title = _episode_metadata_from_dir_name(
            manifest_path.parent.name
        )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid voice sample manifest JSON: {manifest_path}"
            ) from exc
        raw_samples = payload.get("samples", [])
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError(
                f"Voice sample manifest has no sample rows: {manifest_path}"
            )
        for row_number, raw_sample in enumerate(raw_samples, start=1):
            if not isinstance(raw_sample, dict):
                raise ValueError(
                    f"Sample row {row_number} in {manifest_path} must be a mapping."
                )
            speaker = str(raw_sample.get("speaker", "")).strip()
            path_value = str(raw_sample.get("path", "")).strip()
            if not speaker or not path_value:
                raise ValueError(
                    f"Sample row {row_number} in {manifest_path} must include speaker and path."
                )
            sample_path = _resolve_manifest_sample_path(
                manifest_path=manifest_path,
                path_value=path_value,
                speaker=speaker,
            )
            if not sample_path.exists():
                raise FileNotFoundError(
                    f"Voice sample path from {manifest_path} row {row_number} "
                    f"does not exist: {sample_path}"
                )
            samples.append(
                VoiceSampleSpec(
                    episode_order=episode_order,
                    episode_title=episode_title,
                    speaker=speaker,
                    sample_path=sample_path,
                )
            )
    if not samples:
        raise ValueError(f"No usable voice samples found under: {voice_samples_root}")
    return tuple(samples)


def load_audio_config_for_voice_clone(
    *,
    config_path: Path,
    overrides: Sequence[str],
    device: str | None,
    execution_backend: str | None,
) -> dict[str, Any]:
    """Load runtime audio config and apply voice-clone CLI overrides."""
    config = load_runtime_config(config_path)
    OmegaConf.update(config, "audio.voice_clone.enabled", True, merge=True)
    OmegaConf.update(config, "audio.voice_clone.verbose", True, merge=True)
    OmegaConf.update(
        config,
        "audio.voice_clone.stream_subprocess_output",
        True,
        merge=True,
    )
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    if device is not None:
        OmegaConf.update(config, "audio.voice_clone.device", device, merge=True)
    if execution_backend is not None:
        OmegaConf.update(
            config,
            "audio.voice_clone.execution_backend",
            execution_backend,
            merge=True,
        )

    audio_section = OmegaConf.select(config, "audio")
    if audio_section is None:
        audio_cfg: dict[str, Any] = {}
    else:
        raw_audio_cfg = OmegaConf.to_container(audio_section, resolve=True)
        if not isinstance(raw_audio_cfg, dict):
            raise ValueError("Config field `audio` must be a mapping.")
        audio_cfg = raw_audio_cfg

    voice_clone_cfg = audio_cfg.get("voice_clone", {})
    if not isinstance(voice_clone_cfg, dict):
        raise ValueError("Config field `audio.voice_clone` must be a mapping.")
    voice_clone_cfg = dict(voice_clone_cfg)
    audio_cfg["voice_clone"] = voice_clone_cfg
    return audio_cfg


def resynthesize_voice_samples(
    *,
    provider: VoiceCloneProvider,
    samples: Sequence[VoiceSampleSpec],
    text: str,
    output_dir: Path,
    force: bool = False,
) -> ResynthesisResult:
    """Render the supplied text through each voice sample."""
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Synthesis text must be non-empty.")
    if not samples:
        raise ValueError("At least one voice sample is required.")

    output_dir.mkdir(parents=True, exist_ok=True)
    text_sha256 = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    text_hash = text_sha256[:12]

    rendered_samples: list[RenderedSample] = []
    for index, sample in enumerate(samples, start=1):
        output_audio_path = build_output_audio_path(
            output_dir=output_dir,
            sample=sample,
            text_hash=text_hash,
        )
        if output_audio_path.exists() and not force:
            print(
                f"[{index}/{len(samples)}] {sample.episode_order:02d} "
                f"{sample.speaker} skipped existing -> {output_audio_path}"
            )
            status = "skipped_existing"
        else:
            print(
                f"[{index}/{len(samples)}] {sample.episode_order:02d} "
                f"{sample.speaker} -> {output_audio_path}"
            )
            provider.synthesize(
                reference_audio_path=sample.sample_path,
                text=normalized_text,
                output_audio_path=output_audio_path,
            )
            status = "rendered"
        rendered_samples.append(
            RenderedSample(
                sample=sample,
                output_audio_path=output_audio_path,
                status=status,
            )
        )

    manifest_path = output_dir / MANIFEST_FILENAME_TEMPLATE.format(text_hash=text_hash)
    write_manifest(
        manifest_path=manifest_path,
        text=normalized_text,
        text_sha256=text_sha256,
        output_dir=output_dir,
        rendered_samples=tuple(rendered_samples),
    )
    return ResynthesisResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        text_sha256=text_sha256,
        rendered_samples=tuple(rendered_samples),
    )


def build_output_audio_path(
    *,
    output_dir: Path,
    sample: VoiceSampleSpec,
    text_hash: str,
) -> Path:
    """Build a deterministic output path for one episode speaker."""
    episode_dir_name = (
        f"{sample.episode_order:02d}_{slugify(sample.episode_title, max_length=72)}"
    )
    speaker_name = slugify(sample.speaker, max_length=48) or "speaker"
    return output_dir / episode_dir_name / f"{speaker_name}_{text_hash}.wav"


def write_manifest(
    *,
    manifest_path: Path,
    text: str,
    text_sha256: str,
    output_dir: Path,
    rendered_samples: Sequence[RenderedSample],
) -> None:
    """Write the resynthesis manifest."""
    payload = {
        "generated_at_utc": utc_now_iso(),
        "input_text": text,
        "text_sha256": text_sha256,
        "output_dir": str(output_dir),
        "sample_count": len(rendered_samples),
        "samples": [
            {
                "episode_order": rendered.sample.episode_order,
                "episode_title": rendered.sample.episode_title,
                "speaker": rendered.sample.speaker,
                "reference_audio_path": str(rendered.sample.sample_path),
                "output_audio_path": str(rendered.output_audio_path),
                "status": rendered.status,
            }
            for rendered in rendered_samples
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(manifest_path)


def slugify(value: str, *, max_length: int) -> str:
    """Convert human text into a compact filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        return ""
    return slug[:max_length].rstrip("_")


def utc_now_iso() -> str:
    """Return a compact UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_inventory_sample_path(*, inventory_path: Path, path_value: str) -> Path:
    """Resolve one voice sample path relative to the inventory CSV."""
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (inventory_path.parent / candidate).resolve()


def _resolve_manifest_sample_path(
    *,
    manifest_path: Path,
    path_value: str,
    speaker: str,
) -> Path:
    """Resolve a manifest-listed sample path, tolerating stale container paths."""
    logical_speaker_path = manifest_path.parent / f"{speaker}.wav"
    if logical_speaker_path.exists():
        return logical_speaker_path.resolve()

    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    if candidate.exists():
        return candidate.resolve()
    fallback = manifest_path.parent / Path(path_value).name
    return fallback.resolve()


def _episode_metadata_from_dir_name(dir_name: str) -> tuple[int, str]:
    """Parse episode order and title from the packaged voice-sample directory."""
    match = re.match(r"^\s*(\d+)\s*-\s*(.+?)\s*$", dir_name)
    if match is None:
        raise ValueError(
            "Episode voice sample directory must start with '<order> - <title>': "
            f"{dir_name}"
        )
    return int(match.group(1)), match.group(2).strip()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
