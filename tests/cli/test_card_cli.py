"""Tests for the `card-framework` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from card_framework.api import InferenceResult
from card_framework.cli import card
from card_framework.runtime.preflight import Check, NotReadyError


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "episode.wav"
    path.write_bytes(b"RIFF")
    return path


def _result(output_dir: Path) -> InferenceResult:
    final = output_dir / "audio_stage" / "interjector" / "voice_cloned_interjected.wav"
    return InferenceResult(
        output_dir=output_dir,
        transcript_path=output_dir / "transcript.json",
        summary_xml_path=output_dir / "summary.xml",
        voice_clone_manifest_path=None,
        voice_clone_audio_path=None,
        interjector_manifest_path=None,
        final_audio_path=final,
    )


def test_shorten_passes_options_to_infer_and_prints_the_final_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    audio: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print only the final audio path on stdout so it can be piped."""
    calls: list[dict[str, object]] = []

    def _fake_infer(audio_wav, output_dir, target, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"audio": audio_wav, "out": output_dir, "target": target, **kwargs})
        return _result(Path(output_dir))

    monkeypatch.setattr("card_framework.api.infer", _fake_infer)
    monkeypatch.chdir(tmp_path)

    code = card.main(["shorten", str(audio), "--to", "1m30s", "--llm", "deepseek_chat"])

    assert code == card.EXIT_OK
    assert calls == [
        {
            "audio": audio,
            "out": tmp_path / "episode-1m30s",
            "target": 90,
            "device": "auto",
            "vllm_url": None,
            "llm": "deepseek_chat",
            "verbose": False,
        }
    ]
    out = capsys.readouterr().out.strip()
    assert out.endswith("voice_cloned_interjected.wav")


def test_shorten_json_prints_every_result_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    audio: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emit machine-readable paths for scripts."""
    monkeypatch.setattr(
        "card_framework.api.infer",
        lambda audio_wav, output_dir, target, **_: _result(Path(output_dir)),
    )

    code = card.main(
        ["shorten", str(audio), "--to", "30s", "-o", str(tmp_path / "out"), "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == card.EXIT_OK
    assert payload["summary_xml_path"] == str(tmp_path / "out" / "summary.xml")
    assert payload["voice_clone_audio_path"] is None


def test_shorten_rejects_a_missing_recording_before_loading_the_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fail fast with the program name and the bad path."""
    code = card.main(["shorten", str(tmp_path / "nope.wav"), "--to", "5m"])

    assert code == card.EXIT_USAGE
    assert "card-framework:" in capsys.readouterr().err


def test_shorten_rejects_an_unreadable_length(
    audio: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Treat a bad --to value as a usage error with a hint."""
    with pytest.raises(SystemExit) as exit_info:
        card.main(["shorten", str(audio), "--to", "five minutes"])

    assert exit_info.value.code == card.EXIT_USAGE
    assert "Try 30s, 5m, or 1m30s" in capsys.readouterr().err


def test_shorten_reports_not_ready_with_fixes(
    monkeypatch: pytest.MonkeyPatch,
    audio: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 3 and show what to fix when a prerequisite is missing."""

    def _not_ready(*_: object, **__: object) -> InferenceResult:
        raise NotReadyError(
            [
                Check(
                    "LLM",
                    "fail",
                    "nothing is answering at http://localhost:8000/v1",
                    "Start vLLM.",
                )
            ]
        )

    monkeypatch.setattr("card_framework.api.infer", _not_ready)

    code = card.main(["shorten", str(audio), "--to", "5m"])

    err = capsys.readouterr().err
    assert code == card.EXIT_NOT_READY
    assert "nothing is answering at http://localhost:8000/v1" in err
    assert "Start vLLM." in err


def test_shorten_reports_pipeline_failure(
    monkeypatch: pytest.MonkeyPatch,
    audio: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 1 when the pipeline itself fails."""

    def _failed(*_: object, **__: object) -> InferenceResult:
        raise RuntimeError("CARD pipeline execution failed. The pipeline log is at card.log")

    monkeypatch.setattr("card_framework.api.infer", _failed)

    assert card.main(["shorten", str(audio), "--to", "5m"]) == card.EXIT_FAILED
    assert "card.log" in capsys.readouterr().err


def test_shorten_llm_and_llm_url_are_mutually_exclusive(audio: Path) -> None:
    """Refuse two LLM choices at once."""
    with pytest.raises(SystemExit) as exit_info:
        card.main(["shorten", str(audio), "--to", "5m", "--llm", "x", "--llm-url", "http://y"])

    assert exit_info.value.code == card.EXIT_USAGE


@pytest.mark.parametrize(
    ("statuses", "expected_code", "last_line"),
    [
        (["ok", "warn"], card.EXIT_OK, "ready"),
        (["ok", "fail"], card.EXIT_NOT_READY, "not ready: 1 problem(s) to fix"),
    ],
)
def test_doctor_exit_code_follows_failures_not_warnings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    statuses: list[str],
    expected_code: int,
    last_line: str,
) -> None:
    """Let warnings through and stop only on failures."""
    checks = [
        Check(f"check{index}", status, "detail", "fix it")  # type: ignore[arg-type]
        for index, status in enumerate(statuses)
    ]
    monkeypatch.setattr("card_framework.api.doctor_checks", lambda **_: checks)

    code = card.main(["doctor"])

    out = capsys.readouterr().out
    assert code == expected_code
    assert "[ok] check0: detail" in out
    assert out.splitlines()[-1] == last_line


def test_doctor_json_lists_checks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emit the checks as JSON for scripts and bug reports."""
    monkeypatch.setattr(
        "card_framework.api.doctor_checks",
        lambda **_: [Check("FFmpeg", "ok", "/usr/bin/ffmpeg")],
    )

    assert card.main(["doctor", "--json"]) == card.EXIT_OK
    assert json.loads(capsys.readouterr().out) == [
        {"name": "FFmpeg", "status": "ok", "detail": "/usr/bin/ffmpeg", "fix": ""}
    ]


def test_version_flag_prints_the_program_name(capsys: pytest.CaptureFixture[str]) -> None:
    """Answer --version without loading the pipeline."""
    with pytest.raises(SystemExit) as exit_info:
        card.main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("card-framework ")
