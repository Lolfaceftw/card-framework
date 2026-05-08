"""Coverage for the five-episode voice sample resynthesis helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf

from card_framework.shared.paths import REPO_ROOT


def _load_module():
    script_path = (
        REPO_ROOT / "scripts" / "resynthesize_5_episode_voice_samples.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "resynthesize_5_episode_voice_samples",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_inventory(package_root: Path) -> Path:
    kit_dir = package_root / "descript_voice_recording_kit"
    voice_dir = package_root / "voice_samples" / "01 - Episode Alpha"
    kit_dir.mkdir(parents=True, exist_ok=True)
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "SPEAKER_00.wav").write_bytes(b"RIFF_host")
    (voice_dir / "SPEAKER_01.wav").write_bytes(b"RIFF_guest")
    inventory_path = kit_dir / "voice_sample_inventory.csv"
    inventory_path.write_text(
        "episode_order,episode_title,speaker,source_voice_sample,recording_statement\n"
        '1,"Episode Alpha",SPEAKER_00,"../voice_samples/01 - Episode Alpha/SPEAKER_00.wav",statement.txt\n'
        '1,"Episode Alpha",SPEAKER_01,"../voice_samples/01 - Episode Alpha/SPEAKER_01.wav",statement.txt\n',
        encoding="utf-8",
    )
    return inventory_path


def _write_manifest_voice_samples(package_root: Path) -> Path:
    voice_dir = package_root / "voice_samples" / "01 - Episode Alpha"
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "SPEAKER_00.wav").write_bytes(b"RIFF_host")
    (voice_dir / "SPEAKER_01.wav").write_bytes(b"RIFF_guest")
    (voice_dir / "manifest.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "speaker": "SPEAKER_00",
                        "path": "/workspace/card-framework/stale/SPEAKER_00.wav",
                        "duration_ms": 12000,
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "path": "/workspace/card-framework/stale/SPEAKER_01.wav",
                        "duration_ms": 14000,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return package_root / "voice_samples"


class _FakeVoiceCloneProvider:
    """Record synthesize calls and write deterministic test outputs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None = None,
        emotion_vector=None,
        progress_callback=None,
    ) -> Path:
        del emo_text, emotion_vector, progress_callback
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        output_audio_path.write_bytes(
            reference_audio_path.read_bytes() + b"|" + text.encode("utf-8")
        )
        self.calls.append(
            {
                "reference_audio_path": reference_audio_path,
                "text": text,
                "output_audio_path": output_audio_path,
            }
        )
        return output_audio_path

    def close(self) -> None:
        self.closed = True


def test_load_voice_sample_inventory_resolves_paths_from_csv_dir(
    tmp_path: Path,
) -> None:
    module = _load_module()
    inventory_path = _write_inventory(tmp_path / "human_eval_assets_10m")

    samples = module.load_voice_sample_inventory(inventory_path.resolve())

    assert [sample.speaker for sample in samples] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(sample.sample_path.exists() for sample in samples)
    assert (
        samples[0].sample_path
        == (
            tmp_path
            / "human_eval_assets_10m"
            / "voice_samples"
            / "01 - Episode Alpha"
            / "SPEAKER_00.wav"
        ).resolve()
    )


def test_load_voice_sample_manifests_falls_back_from_stale_container_paths(
    tmp_path: Path,
) -> None:
    module = _load_module()
    voice_samples_root = _write_manifest_voice_samples(
        tmp_path / "human_eval_assets_10m"
    )

    samples = module.load_voice_sample_manifests(voice_samples_root.resolve())

    assert [sample.speaker for sample in samples] == ["SPEAKER_00", "SPEAKER_01"]
    assert samples[0].episode_order == 1
    assert samples[0].episode_title == "Episode Alpha"
    assert (
        samples[0].sample_path
        == (voice_samples_root / "01 - Episode Alpha" / "SPEAKER_00.wav").resolve()
    )


def test_resynthesize_voice_samples_writes_outputs_and_manifest(tmp_path: Path) -> None:
    module = _load_module()
    inventory_path = _write_inventory(tmp_path / "human_eval_assets_10m")
    samples = module.load_voice_sample_inventory(inventory_path.resolve())
    provider = _FakeVoiceCloneProvider()

    result = module.resynthesize_voice_samples(
        provider=provider,
        samples=samples,
        text="Hello from the resynthesis test.",
        output_dir=(tmp_path / "outputs").resolve(),
    )

    assert len(provider.calls) == 2
    assert len(result.rendered_samples) == 2
    assert all(item.status == "rendered" for item in result.rendered_samples)
    assert all(item.output_audio_path.exists() for item in result.rendered_samples)
    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["sample_count"] == 2
    assert manifest_payload["input_text"] == "Hello from the resynthesis test."
    assert manifest_payload["samples"][0]["status"] == "rendered"

    second_result = module.resynthesize_voice_samples(
        provider=provider,
        samples=samples,
        text="Hello from the resynthesis test.",
        output_dir=(tmp_path / "outputs").resolve(),
    )

    assert len(provider.calls) == 2
    assert [item.status for item in second_result.rendered_samples] == [
        "skipped_existing",
        "skipped_existing",
    ]


def test_main_builds_provider_from_config_and_closes_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    inventory_path = _write_inventory(tmp_path / "human_eval_assets_10m").resolve()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "audio:\n  voice_clone:\n    provider: indextts\n", encoding="utf-8"
    )
    provider = _FakeVoiceCloneProvider()
    captured_audio_cfg: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "load_runtime_config",
        lambda path: OmegaConf.create(
            {"audio": {"voice_clone": {"provider": "indextts"}}}
        ),
    )

    def _fake_build_voice_clone_provider(audio_cfg, *, project_root):
        del project_root
        captured_audio_cfg.update(audio_cfg)
        return provider

    monkeypatch.setattr(
        module,
        "build_voice_clone_provider",
        _fake_build_voice_clone_provider,
    )

    exit_code = module.main(
        [
            "--text",
            "CLI supplied text",
            "--inventory-path",
            str(inventory_path),
            "--config-path",
            str(config_path),
            "--output-dir",
            str((tmp_path / "cli_outputs").resolve()),
            "--device",
            "cpu",
            "--execution-backend",
            "subprocess",
        ]
    )

    assert exit_code == 0
    assert len(provider.calls) == 2
    assert provider.closed is True
    voice_clone_cfg = captured_audio_cfg["voice_clone"]
    assert voice_clone_cfg["enabled"] is True
    assert voice_clone_cfg["verbose"] is True
    assert voice_clone_cfg["stream_subprocess_output"] is True
    assert voice_clone_cfg["device"] == "cpu"
    assert voice_clone_cfg["execution_backend"] == "subprocess"
