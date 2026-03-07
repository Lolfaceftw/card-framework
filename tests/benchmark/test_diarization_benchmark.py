from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import diarization


class _StubDiarizer:
    def diarize(self, *, audio_path: Path, output_dir: Path, device: str):
        del audio_path, output_dir, device
        from audio_pipeline.contracts import DiarizationTurn

        return [
            DiarizationTurn(speaker="SPEAKER_00", start_time_ms=0, end_time_ms=1000),
            DiarizationTurn(
                speaker="SPEAKER_01",
                start_time_ms=1000,
                end_time_ms=2000,
            ),
        ]


def test_load_manifest_supports_alias_fields(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav")
    rttm_path = tmp_path / "sample.rttm"
    rttm_path.write_text(
        "SPEAKER sample 1 0.000 1.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n",
        encoding="utf-8",
    )
    uem_path = tmp_path / "sample.uem"
    uem_path.write_text("sample 1 0.000 1.000\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample",
                        "dataset": "callhome",
                        "audio_filepath": str(audio_path),
                        "rttm_filepath": str(rttm_path),
                        "uem_filepath": str(uem_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    samples = diarization._load_manifest(manifest_path, repo_root=tmp_path)

    assert len(samples) == 1
    assert samples[0].sample_id == "sample"
    assert samples[0].uem_path == str(uem_path)


def test_execute_command_writes_report_and_scores(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "audio:",
                "  device: cpu",
                "  diarization:",
                "    provider: nemo",
                "    pyannote: {}",
                "    sortformer_offline: {}",
                "    sortformer_streaming: {}",
            ]
        ),
        encoding="utf-8",
    )
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav")
    rttm_path = tmp_path / "sample.rttm"
    rttm_path.write_text(
        "\n".join(
            [
                "SPEAKER sample 1 0.000 1.000 <NA> <NA> SPEAKER_00 <NA> <NA>",
                "SPEAKER sample 1 1.000 1.000 <NA> <NA> SPEAKER_01 <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    uem_path = tmp_path / "sample.uem"
    uem_path.write_text("sample 1 0.000 2.000\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample",
                        "dataset": "callhome",
                        "subset": "part2",
                        "audio_filepath": str(audio_path),
                        "rttm_filepath": str(rttm_path),
                        "uem_filepath": str(uem_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(diarization, "build_speaker_diarizer", lambda audio_cfg: _StubDiarizer())
    monkeypatch.setattr(diarization, "_score_turns", lambda **kwargs: (0.0, 0.0))
    monkeypatch.setattr(diarization, "git_info", lambda repo_root: ("commit", "branch"))
    monkeypatch.setattr(diarization, "probe_audio_duration_ms", lambda path: 2000)
    monkeypatch.setattr(diarization.time, "strftime", lambda fmt, value: "20260101T000000Z")

    result = diarization.execute_command(
        argparse.Namespace(
            manifest=str(manifest_path),
            config=str(config_path),
            output_dir=str(tmp_path / "out"),
            providers="nemo,pyannote_community1",
            device="cpu",
            max_samples=0,
            collar=0.0,
            skip_overlap=False,
        )
    )

    assert result == 0
    report_path = tmp_path / "out" / "20260101T000000Z" / "diarization_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["providers"] == ["nemo", "pyannote_community1"]
    assert len(payload["results"]) == 2
    assert payload["aggregates"][0]["mean_der"] == 0.0
    assert payload["aggregates"][0]["mean_jer"] == 0.0


def test_prepare_manifest_command_uses_ami_prep(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_prepare_ami_manifest(**kwargs):
        captured.update(kwargs)
        return {"samples": [{"sample_id": "ES2004a"}]}

    monkeypatch.setattr(diarization, "prepare_ami_manifest", _fake_prepare_ami_manifest)

    result = diarization.prepare_manifest_command(
        argparse.Namespace(
            output=str(tmp_path / "benchmark" / "manifests" / "diarization_ami_test.json"),
            data_root=str(tmp_path / "artifacts" / "diarization_datasets" / "ami"),
            subset="test",
            num_samples=3,
            force_download=True,
        )
    )

    assert result == 0
    assert captured["subset"] == "test"
    assert captured["num_samples"] == 3
    assert captured["force_download"] is True


def test_main_preserves_legacy_execute_style_invocation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_command(args: argparse.Namespace) -> int:
        captured["command"] = args.command
        captured["manifest"] = args.manifest
        return 0

    monkeypatch.setattr(diarization, "execute_command", _fake_execute_command)

    result = diarization.main(["--manifest", "custom_manifest.json"])

    assert result == 0
    assert captured["command"] == "execute"
    assert captured["manifest"] == "custom_manifest.json"
