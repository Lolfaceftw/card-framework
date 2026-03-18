"""Coverage for the SPoRC human transcript sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from card_framework.benchmark.summarizer_critic.sporc.human_sampler import (
    TranscriptLine,
    _build_episode_plan,
    _compute_target_words,
    _parse_selection_input,
    run_human_sampling_session,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    PreparedBenchmarkSample,
)


def _write_manifest(tmp_path: Path) -> Path:
    transcript_path = tmp_path / "transcripts" / "episode.transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0,
                        "end_time": 1000,
                        "text": "alpha beta gamma",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 1000,
                        "end_time": 2000,
                        "text": "delta epsilon zeta",
                    },
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 2000,
                        "end_time": 3000,
                        "text": "eta theta iota",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 3000,
                        "end_time": 4000,
                        "text": "kappa lambda mu",
                    },
                ],
                "metadata": {
                    "dataset": "sporc",
                    "episode_title": "Sampler Episode",
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(tmp_path / "speakerTurnData.jsonl.gz"),
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "sporc_00001_episode",
                        "dataset": "sporc_full",
                        "episode_key": "episode://sampler",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "Reference summary.",
                        "target_seconds": 45,
                        "metadata": {"episode_title": "Sampler Episode"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_compute_target_words_uses_configurable_wpm_and_factor() -> None:
    assert _compute_target_words(30, words_per_minute=170.0, sample_factor=5.0) == 425
    assert _compute_target_words(60, words_per_minute=170.0, sample_factor=5.0) == 850
    assert _compute_target_words(300, words_per_minute=170.0, sample_factor=5.0) == 4250


def test_build_episode_plan_creates_word_balanced_windows(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = PreparedBenchmarkSample(
        sample_id=manifest_payload["samples"][0]["sample_id"],
        dataset=manifest_payload["samples"][0]["dataset"],
        episode_key=manifest_payload["samples"][0]["episode_key"],
        transcript_path=manifest_payload["samples"][0]["transcript_path"],
        reference_summary=manifest_payload["samples"][0]["reference_summary"],
        target_seconds=manifest_payload["samples"][0]["target_seconds"],
        metadata=dict(manifest_payload["samples"][0]["metadata"]),
    )

    episode_plan = _build_episode_plan(
        sample,
        target_words=12,
        max_windows_per_episode=2,
    )

    assert episode_plan.transcript_word_count == 12
    assert episode_plan.target_samples == 4
    assert [window.max_samples for window in episode_plan.windows] == [2, 2]
    assert [window.start_line_number for window in episode_plan.windows] == [1, 3]
    assert [window.end_line_number for window in episode_plan.windows] == [2, 4]


def test_parse_selection_input_supports_lines_and_overrides() -> None:
    window_lines = (
        TranscriptLine(
            line_number=1,
            speaker="SPEAKER_00",
            text="alpha beta gamma",
            word_count=3,
            start_time=0,
            end_time=1000,
        ),
        TranscriptLine(
            line_number=2,
            speaker="SPEAKER_01",
            text="delta epsilon zeta",
            word_count=3,
            start_time=1000,
            end_time=2000,
        ),
    )

    parsed = _parse_selection_input(
        '1,"/override [SPEAKER_04] replacement, segment"',
        max_samples=2,
        window_lines=window_lines,
    )

    assert len(parsed) == 2
    assert parsed[0].selection_type == "line"
    assert parsed[0].line_number == 1
    assert parsed[1].selection_type == "override"
    assert parsed[1].speaker == "SPEAKER_04"
    assert parsed[1].text == "replacement, segment"


def test_run_human_sampling_session_pauses_and_resumes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    args = argparse.Namespace(
        dataset_manifest=str(manifest_path),
        sampling_root=str(sampling_root),
        resume_session="",
        duration_preset="30s",
        episodes=1,
        words_per_minute=6.0,
        sample_factor=1.0,
        max_windows_per_episode=1,
    )

    def _raise_keyboard_interrupt(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.human_sampler.Prompt.ask",
        _raise_keyboard_interrupt,
    )

    paused_exit_code = run_human_sampling_session(args, console=console)

    assert paused_exit_code == 130
    session_dirs = list(sampling_root.iterdir())
    assert len(session_dirs) == 1
    session_id = session_dirs[0].name
    paused_summary = json.loads(
        (session_dirs[0] / "session_summary.json").read_text(encoding="utf-8")
    )
    assert paused_summary["status"] == "paused"

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.human_sampler.Prompt.ask",
        lambda *args, **kwargs: "1",
    )
    resumed_args = argparse.Namespace(
        dataset_manifest=str(manifest_path),
        sampling_root=str(sampling_root),
        resume_session=session_id,
        duration_preset="",
        episodes=0,
        words_per_minute=0.0,
        sample_factor=0.0,
        max_windows_per_episode=0,
    )

    completed_exit_code = run_human_sampling_session(resumed_args, console=console)

    assert completed_exit_code == 0
    completed_summary = json.loads(
        (session_dirs[0] / "session_summary.json").read_text(encoding="utf-8")
    )
    assert completed_summary["status"] == "completed"
    csv_lines = (session_dirs[0] / "selected_samples.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(csv_lines) == 2
    assert "line_number" in csv_lines[0]
    assert ",1," in csv_lines[1]
