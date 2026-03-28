"""Coverage for the SPoRC Anthropic-backed transcript sampler."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from rich.console import Console

from card_framework.agents.dtos import AssistantMessage, Function, ToolCall
from card_framework.benchmark.summarizer_critic.sporc.llm_sampler import (
    _resolve_settings,
    run_llm_sampling_session,
)
from card_framework.orchestration.transcript import Transcript


def _write_turn_data(tmp_path: Path, *, episode_key: str, turns: list[dict[str, Any]]) -> Path:
    turn_data_path = tmp_path / "speakerTurnData.jsonl.gz"
    with gzip.open(turn_data_path, "wt", encoding="utf-8") as handle:
        for turn in turns:
            handle.write(
                json.dumps(
                    {
                        "episode_url": episode_key,
                        "speaker": turn["speaker"],
                        "text": turn["text"],
                        "start": turn["start"],
                        "end": turn["end"],
                    }
                )
                + "\n"
            )
    return turn_data_path


def _write_manifest(tmp_path: Path) -> Path:
    turn_data_path = _write_turn_data(
        tmp_path,
        episode_key="episode://sampler",
        turns=[
            {
                "speaker": "SPEAKER_00",
                "text": "alpha beta gamma",
                "start": 0.0,
                "end": 1.0,
            },
            {
                "speaker": "SPEAKER_00",
                "text": "delta epsilon",
                "start": 1.0,
                "end": 2.0,
            },
            {
                "speaker": "SPEAKER_01",
                "text": "zeta eta theta iota kappa",
                "start": 2.0,
                "end": 3.0,
            },
        ],
    )
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
                        "text": "alpha beta gamma delta epsilon",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 1000,
                        "end_time": 2000,
                        "text": "zeta eta theta iota kappa",
                    },
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 2000,
                        "end_time": 3000,
                        "text": "lambda mu nu xi omicron",
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
                "turn_data_path": str(turn_data_path),
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


def _write_long_duration_manifest(tmp_path: Path) -> Path:
    turn_data_path = _write_turn_data(
        tmp_path,
        episode_key="episode://sampler",
        turns=[
            {
                "speaker": "SPEAKER_00",
                "text": "alpha beta gamma",
                "start": 0.0,
                "end": 15.0,
            },
            {
                "speaker": "SPEAKER_00",
                "text": "delta epsilon",
                "start": 15.0,
                "end": 30.0,
            },
            {
                "speaker": "SPEAKER_01",
                "text": "zeta eta theta iota kappa",
                "start": 30.0,
                "end": 45.0,
            },
        ],
    )
    transcript_path = tmp_path / "transcripts" / "long_duration.transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0,
                        "end_time": 15000,
                        "text": "alpha beta gamma delta epsilon",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 15000,
                        "end_time": 30000,
                        "text": "zeta eta theta iota kappa",
                    },
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 30000,
                        "end_time": 45000,
                        "text": "lambda mu nu xi omicron",
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
    manifest_path = tmp_path / "long_duration_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(turn_data_path),
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


def _write_episode_assets_manifest(tmp_path: Path) -> Path:
    manifest_path = _write_manifest(tmp_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_payload = manifest_payload["samples"][0]
    summary_path = tmp_path / "episode_assets" / "prepared_episodes_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "dataset_manifest_path": str(manifest_path),
                "output_root": str(summary_path.parent),
                "episode_count": 1,
                "episodes": [
                    {
                        "sample_id": sample_payload["sample_id"],
                        "episode_title": sample_payload["metadata"]["episode_title"],
                        "episode_root": str(summary_path.parent / sample_payload["sample_id"]),
                        "transcript_path": sample_payload["transcript_path"],
                        "source_transcript_path": sample_payload["transcript_path"],
                        "sampled_transcript_path": sample_payload["transcript_path"],
                        "source_audio_url": "https://example.com/audio.mp3",
                        "source_audio_path": str(tmp_path / "audio.mp3"),
                        "speaker_samples_manifest_path": str(
                            tmp_path / "speaker_samples" / "manifest.json"
                        ),
                        "segment_files": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def _write_fragmented_manifest(tmp_path: Path) -> Path:
    turn_data_path = _write_turn_data(
        tmp_path,
        episode_key="episode://fragmented",
        turns=[
            {
                "speaker": "SPEAKER_00",
                "text": "alpha beta",
                "start": 0.0,
                "end": 1.0,
            },
            {
                "speaker": "SPEAKER_00",
                "text": "gamma delta",
                "start": 1.0,
                "end": 2.0,
            },
            {
                "speaker": "SPEAKER_01",
                "text": "epsilon zeta",
                "start": 2.0,
                "end": 3.0,
            },
        ],
    )
    transcript_path = tmp_path / "transcripts" / "fragmented.transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0,
                        "end_time": 1000,
                        "text": "alpha beta",
                    },
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 1000,
                        "end_time": 2000,
                        "text": "gamma delta",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 2000,
                        "end_time": 3000,
                        "text": "epsilon zeta",
                    },
                ],
                "metadata": {
                    "dataset": "sporc",
                    "episode_title": "Fragmented Episode",
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "fragmented_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(turn_data_path),
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "sporc_fragmented_episode",
                        "dataset": "sporc_full",
                        "episode_key": "episode://fragmented",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "Reference summary.",
                        "target_seconds": 45,
                        "metadata": {"episode_title": "Fragmented Episode"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_seam_manifest(tmp_path: Path) -> Path:
    turn_data_path = _write_turn_data(
        tmp_path,
        episode_key="episode://seam",
        turns=[
            {
                "speaker": "SPEAKER_00",
                "text": "alpha beta",
                "start": 0.0,
                "end": 1.0,
            },
            {
                "speaker": "SPEAKER_01",
                "text": "gamma",
                "start": 1.0,
                "end": 2.0,
            },
            {
                "speaker": "SPEAKER_00",
                "text": "delta epsilon",
                "start": 2.0,
                "end": 3.0,
            },
        ],
    )
    transcript_path = tmp_path / "transcripts" / "seam.transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0,
                        "end_time": 1000,
                        "text": "alpha beta",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 1000,
                        "end_time": 2000,
                        "text": "gamma",
                    },
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 2000,
                        "end_time": 3000,
                        "text": "delta epsilon",
                    },
                ],
                "metadata": {
                    "dataset": "sporc",
                    "episode_title": "Seam Episode",
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "seam_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(turn_data_path),
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "sporc_seam_episode",
                        "dataset": "sporc_full",
                        "episode_key": "episode://seam",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "Reference summary.",
                        "target_seconds": 45,
                        "metadata": {"episode_title": "Seam Episode"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_multi_manifest(tmp_path: Path, *, episode_count: int = 4) -> Path:
    manifest_samples: list[dict[str, Any]] = []
    transcripts_root = tmp_path / "transcripts"
    transcripts_root.mkdir(parents=True, exist_ok=True)
    turn_data_path = tmp_path / "speakerTurnData.jsonl.gz"
    with gzip.open(turn_data_path, "wt", encoding="utf-8") as handle:
        for index in range(1, episode_count + 1):
            for turn in (
                {
                    "speaker": "SPEAKER_00",
                    "text": f"episode {index} alpha beta gamma delta",
                    "start": 0.0,
                    "end": 1.0,
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": f"episode {index} zeta eta theta iota",
                    "start": 1.0,
                    "end": 2.0,
                },
            ):
                handle.write(
                    json.dumps(
                        {
                            "episode_url": f"episode://multi/{index}",
                            "speaker": turn["speaker"],
                            "text": turn["text"],
                            "start": turn["start"],
                            "end": turn["end"],
                        }
                    )
                    + "\n"
                )
    for index in range(1, episode_count + 1):
        transcript_path = transcripts_root / f"episode_{index}.transcript.json"
        transcript_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker": "SPEAKER_00",
                            "start_time": 0,
                            "end_time": 1000,
                            "text": f"episode {index} alpha beta gamma delta",
                        },
                        {
                            "speaker": "SPEAKER_01",
                            "start_time": 1000,
                            "end_time": 2000,
                            "text": f"episode {index} zeta eta theta iota",
                        },
                    ],
                    "metadata": {
                        "dataset": "sporc",
                        "episode_title": f"Episode {index}",
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest_samples.append(
            {
                "sample_id": f"sporc_multi_{index:05d}",
                "dataset": "sporc_full",
                "episode_key": f"episode://multi/{index}",
                "transcript_path": str(transcript_path),
                "reference_summary": "Reference summary.",
                "target_seconds": 45,
                "metadata": {"episode_title": f"Episode {index}"},
            }
        )
    manifest_path = tmp_path / "multi_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(turn_data_path),
                "sample_count": episode_count,
                "samples": manifest_samples,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


class _NoOpRichCallback:
    """Suppress terminal streaming noise during tests."""

    def on_start(self, agent_name: str) -> None:
        del agent_name

    def on_thought_token(self, token: str) -> None:
        del token

    def on_content_token(self, token: str) -> None:
        del token

    def on_complete(self) -> None:
        return None


class _FakeProvider:
    """Fake Anthropic provider that emits predetermined responses."""

    planned_responses: list[dict[str, Any] | Exception] = []
    instances: list["_FakeProvider"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = dict(kwargs)
        self.calls: list[dict[str, Any]] = []
        self.last_response_details: dict[str, Any] = {}
        self._response_callback: Any = None
        self._client = None
        _FakeProvider.instances.append(self)

    def set_response_callback(self, callback: Any) -> None:
        self._response_callback = callback

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any],
        max_tokens: int,
    ) -> AssistantMessage:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "max_tokens": max_tokens,
            }
        )
        planned = _FakeProvider.planned_responses.pop(0)
        if isinstance(planned, Exception):
            raise planned

        if self._response_callback is not None:
            self._response_callback.on_start("Summarizer")
            if planned.get("thought"):
                self._response_callback.on_thought_token(str(planned["thought"]))
            if planned.get("content"):
                self._response_callback.on_content_token(str(planned["content"]))
            self._response_callback.on_complete()

        tool_name = str(planned.get("tool_name") or "submit_samples")
        tool_arguments = json.dumps(planned["tool_payload"])
        assistant_message = AssistantMessage(
            content=str(planned.get("content", "")),
            reasoning_content=str(planned.get("thought", "")) or None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=Function(
                        name=tool_name,
                        arguments=tool_arguments,
                    ),
                )
            ],
        )
        self.last_response_details = {
            "provider": "anthropic",
            "response": {
                "stop_reason": "tool_use",
                "content": assistant_message.content,
                "reasoning_content": assistant_message.reasoning_content,
                "tool_calls": assistant_message.model_dump().get("tool_calls"),
            },
        }
        return assistant_message


def _build_args(
    *,
    manifest_path: Path,
    sampling_root: Path,
    samples: int = 0,
    disable_thinking: bool = False,
    select_episode: str = "",
    duration_preset: str = "30s",
    text_only: bool = False,
    provider: str = "anthropic",
    transcribe_enabled: bool = False,
    audio_config: str = "",
    audio_device: str = "",
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_manifest=str(manifest_path),
        provider=provider,
        sampling_root=str(sampling_root),
        duration_preset=duration_preset,
        samples=samples,
        text_only=text_only,
        transcribe_enabled=transcribe_enabled,
        audio_config=audio_config,
        audio_device=audio_device,
        episodes=1,
        select_episode=select_episode,
        words_per_minute=20.0,
        sample_factor=1.0,
        model="claude-opus-4-6",
        thinking_effort="medium",
        disable_thinking=disable_thinking,
        max_output_tokens=2048,
    )


def _build_stage1_transcript(
    tmp_path: Path,
    *,
    stem: str,
    payload: dict[str, Any],
) -> tuple[Transcript, Path, Path]:
    transcript_path = (tmp_path / f"{stem}.stage1.transcript.json").resolve()
    transcript_path.write_text(json.dumps(payload), encoding="utf-8")
    source_audio_path = (tmp_path / f"{stem}.mp3").resolve()
    source_audio_path.write_bytes(b"ID3")
    return Transcript.from_mapping(payload), transcript_path, source_audio_path


def _anchored_sample(
    *,
    order: int,
    segment_text: str,
    rationale: str,
    estimated_words: int,
    source_start_index: int,
    source_end_index: int,
    speaker_mode: str = "single_speaker_repair",
    speaker_labels: list[str] | None = None,
    source_start_timestamp: str | None = None,
    source_end_timestamp: str | None = None,
) -> dict[str, Any]:
    labels = list(speaker_labels or [])
    if not labels:
        raise ValueError("speaker_labels must be provided for test samples")
    if source_start_timestamp is None:
        source_start_timestamp = f"00:00:{source_start_index:02d}"
    if source_end_timestamp is None:
        source_end_timestamp = f"00:00:{source_end_index + 1:02d}"
    return {
        "order": order,
        "speaker_mode": speaker_mode,
        "speaker_labels": labels,
        "source_start_index": source_start_index,
        "source_end_index": source_end_index,
        "source_start_timestamp": source_start_timestamp,
        "source_end_timestamp": source_end_timestamp,
        "segment_text": segment_text,
        "rationale": rationale,
        "estimated_words": estimated_words,
    }


def test_run_llm_sampling_session_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_long_duration_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "plan",
            "content": "I will pick two supporting excerpts.",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "target_words": 10,
                "selected_words": 10,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Introduces the first key idea.",
                        estimated_words=5,
                    ),
                    _anchored_sample(
                        order=2,
                        speaker_labels=["SPEAKER_01"],
                        source_start_index=1,
                        source_end_index=1,
                        segment_text="zeta eta theta iota kappa",
                        rationale="Captures the second important turn.",
                        estimated_words=5,
                    ),
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(manifest_path=manifest_path, sampling_root=sampling_root),
        console=console,
    )

    assert exit_code == 0
    run_dirs = list(sampling_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["selected_sample_count"] == 2
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["status"] == "completed"
    selected_segments_text = (
        run_dir / "episodes" / "sporc_00001_episode" / "selected_segments.txt"
    ).read_text(encoding="utf-8")
    assert "[SPEAKER_00]" in selected_segments_text
    assert "Source: 0-0 (00:00:00-00:00:01) | single_speaker_repair" in selected_segments_text
    stream_lines = (
        run_dir / "episodes" / "sporc_00001_episode" / "stream.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert any('"event": "thinking_token"' in line for line in stream_lines)
    assert any('"event": "content_token"' in line for line in stream_lines)
    exported_rows = [
        json.loads(line)
        for line in (run_dir / "selected_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(exported_rows) == 2
    assert exported_rows[0]["speaker_mode"] == "single_speaker_repair"
    assert exported_rows[0]["speaker_labels"] == ["SPEAKER_00"]
    assert exported_rows[0]["source_start_index"] == 0
    assert exported_rows[0]["duration_ms"] == 15000
    assert exported_rows[0]["source_end_timestamp"] == "00:00:01"
    request_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "request.json").read_text(
            encoding="utf-8"
        )
    )
    request_messages = request_doc["attempt_requests"][0]["messages"]
    assert "[#0 00:00:00-00:00:15] [SPEAKER_00] alpha beta gamma delta epsilon" in str(
        request_messages[1]["content"]
    )


def test_run_llm_sampling_session_supports_fixed_sample_count_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "fixed-count plan",
            "content": "Returning exactly two excerpts.",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 2,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Important opening cue.",
                        estimated_words=5,
                    ),
                    _anchored_sample(
                        order=2,
                        speaker_labels=["SPEAKER_01"],
                        source_start_index=1,
                        source_end_index=1,
                        segment_text="zeta eta theta iota kappa",
                        rationale="Second important cue.",
                        estimated_words=5,
                    ),
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=2,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["sampling_mode"] == "fixed_samples"
    assert summary["target_samples_per_episode"] == 2
    assert summary["target_words_per_episode"] is None
    exported_rows = (
        run_dir / "selected_samples.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(exported_rows) == 2
    request_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "request.json").read_text(
            encoding="utf-8"
        )
    )
    schema_properties = request_doc["attempt_requests"][0]["tools"][0]["function"][
        "parameters"
    ]["properties"]["samples"]["items"]["properties"]
    assert "speaker_mode" in schema_properties
    assert "source_start_index" in schema_properties
    assert "source_end_timestamp" in schema_properties
    assert (
        request_doc["attempt_requests"][0]["tools"][1]["function"]["name"]
        == "measure_selection_duration"
    )


def test_run_llm_sampling_session_text_only_mode_uses_raw_turn_spans(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 2,
                "samples": [
                    {
                        "order": 1,
                        "source_start_index": 0,
                        "source_end_index": 1,
                        "rationale": "Keep the opening fragmented speaker line together.",
                    },
                    {
                        "order": 2,
                        "source_start_index": 2,
                        "source_end_index": 2,
                        "rationale": "Keep the next speaker turn as its own segment.",
                    },
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=2,
            text_only=True,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["text_only"] is True
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["status"] == "completed"
    assert response_doc["selected_samples"][0]["speaker_mode"] == "raw_transcript_span"
    assert response_doc["selected_samples"][0]["source_segment_count"] == 2
    assert (
        response_doc["selected_samples"][0]["segment_text"]
        == "[SPEAKER_00] alpha beta gamma delta epsilon"
    )
    combined_text = (
        run_dir / "episodes" / "sporc_00001_episode" / "combined_segments.txt"
    ).read_text(encoding="utf-8")
    assert "[SPEAKER_00] alpha beta gamma delta epsilon" in combined_text
    request_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "request.json").read_text(
            encoding="utf-8"
        )
    )
    request_prompt = str(request_doc["attempt_requests"][0]["messages"][1]["content"])
    assert "Each line below is a raw speaker-turn fragment." in request_prompt
    assert "[#0 00:00:00-00:00:01] [SPEAKER_00] alpha beta gamma" in request_prompt
    assert "[#1 00:00:01-00:00:02] [SPEAKER_00] delta epsilon" in request_prompt


def test_run_llm_sampling_session_supports_vllm_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Opening turn.",
                        estimated_words=5,
                    )
                ],
            },
        }
    ]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_URL", "http://vllm.example.test:8000/v1")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_vllm_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
            provider="vllm",
        ),
        console=console,
    )

    assert exit_code == 0
    assert _FakeProvider.instances[0].init_kwargs["base_url"] == "http://vllm.example.test:8000/v1"
    summary = json.loads(
        (next(iter(sampling_root.iterdir())) / "run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["provider"] == "vllm"


def test_run_llm_sampling_session_accepts_episode_assets_summary_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_episode_assets_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Opening line from episode-assets transcript.",
                        estimated_words=5,
                    )
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["manifest_path"] == str(manifest_path.resolve())
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["status"] == "completed"


def test_run_llm_sampling_session_text_only_mode_supports_transcribed_stage1_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    {
                        "order": 1,
                        "source_start_index": 0,
                        "source_end_index": 1,
                        "rationale": "Keep the overlapping host line together.",
                    },
                ],
            },
        }
    ]
    stage1_transcript, stage1_transcript_path, stage1_audio_path = _build_stage1_transcript(
        tmp_path,
        stem="sampler",
        payload={
            "segments": [
                {
                    "speaker": "HOST",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "fresh opening",
                    "overlap_speakers": ["GUEST"],
                },
                {
                    "speaker": "HOST",
                    "start_time": 1000,
                    "end_time": 2000,
                    "text": "still talking",
                },
            ],
            "metadata": {
                "diarization_turns": [
                    {"speaker": "HOST", "start_time": 0, "end_time": 2000},
                    {"speaker": "GUEST", "start_time": 0, "end_time": 700},
                ],
            },
        },
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_sampler_audio_runtime",
        lambda **kwargs: {
            "audio_cfg": {
                "device": "cpu",
                "diarization": {"provider": "nemo"},
            },
            "audio_device": "cpu",
            "audio_diarizer_provider": "nemo",
            "platform_fallback_applied": False,
        },
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._ensure_stage_one_source_transcript",
        lambda **kwargs: (stage1_transcript, stage1_transcript_path, stage1_audio_path),
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
            text_only=True,
            transcribe_enabled=True,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["source_mode"] == "stage1_transcribed"
    request_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "request.json").read_text(
            encoding="utf-8"
        )
    )
    request_prompt = str(request_doc["attempt_requests"][0]["messages"][1]["content"])
    assert "Stage-1 transcript line count: 2" in request_prompt
    assert "[#0 00:00:00-00:00:01] [HOST] [overlap: GUEST] fresh opening" in request_prompt
    combined_text = (
        run_dir / "episodes" / "sporc_00001_episode" / "combined_segments.txt"
    ).read_text(encoding="utf-8")
    assert "[HOST] fresh opening still talking" in combined_text


def test_run_llm_sampling_session_reconstructed_mode_uses_transcribed_stage1_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["HOST"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="fresh opening",
                        rationale="Use the stage-1 opening line.",
                        estimated_words=2,
                    )
                ],
            },
        }
    ]
    stage1_transcript, stage1_transcript_path, stage1_audio_path = _build_stage1_transcript(
        tmp_path,
        stem="sampler_reconstructed",
        payload={
            "segments": [
                {
                    "speaker": "HOST",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "fresh opening",
                    "overlap_speakers": ["GUEST"],
                },
                {
                    "speaker": "GUEST",
                    "start_time": 1200,
                    "end_time": 1800,
                    "text": "reply line",
                },
            ],
            "metadata": {},
        },
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_sampler_audio_runtime",
        lambda **kwargs: {
            "audio_cfg": {
                "device": "cpu",
                "diarization": {"provider": "nemo"},
            },
            "audio_device": "cpu",
            "audio_diarizer_provider": "nemo",
            "platform_fallback_applied": False,
        },
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._ensure_stage_one_source_transcript",
        lambda **kwargs: (stage1_transcript, stage1_transcript_path, stage1_audio_path),
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
            transcribe_enabled=True,
        ),
        console=console,
    )

    assert exit_code == 0
    request_doc = json.loads(
        (
            next(iter(sampling_root.iterdir()))
            / "episodes"
            / "sporc_00001_episode"
            / "request.json"
        ).read_text(encoding="utf-8")
    )
    request_prompt = str(request_doc["attempt_requests"][0]["messages"][1]["content"])
    assert "Transcript source: fresh stage-1 transcript built from source-separated audio" in request_prompt
    assert "[#0 00:00:00-00:00:01] [HOST] [overlap: GUEST] fresh opening" in request_prompt


def test_run_llm_sampling_session_select_episode_range_loads_manifest_slice(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_multi_manifest(tmp_path, episode_count=4)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "selected slice",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://multi/2",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="episode 2 alpha beta gamma delta",
                        rationale="Second manifest episode.",
                        estimated_words=6,
                    )
                ],
            },
        },
        {
            "thought": "selected slice",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://multi/3",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="episode 3 alpha beta gamma delta",
                        rationale="Third manifest episode.",
                        estimated_words=6,
                    )
                ],
            },
        },
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
            select_episode="[2,3]",
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["selected_episode_range"] == [2, 3]
    assert summary["episodes_requested"] == 2
    assert summary["total_episode_count"] == 2
    assert [episode["manifest_episode_number"] for episode in summary["episodes"]] == [2, 3]
    assert [episode["sample_id"] for episode in summary["episodes"]] == [
        "sporc_multi_00002",
        "sporc_multi_00003",
    ]
    assert "Episode 2: Episode 2" in console.export_text()


def test_run_llm_sampling_session_disable_thinking_sets_temperature_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "Using exact transcript segments only.",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Key opening line.",
                        estimated_words=5,
                    )
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
            disable_thinking=True,
        ),
        console=console,
    )

    assert exit_code == 0
    assert len(_FakeProvider.instances) == 1
    assert _FakeProvider.instances[0].init_kwargs["enable_thinking"] is False
    assert _FakeProvider.instances[0].init_kwargs["temperature"] == 0.0


def test_run_llm_sampling_session_accepts_tool_only_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "select exact transcript segments",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Key opening line.",
                        estimated_words=5,
                    )
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["status"] == "completed"


def test_run_llm_sampling_session_supports_duration_measurement_tool_roundtrip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "measure exact timing before finalizing",
            "content": "",
            "tool_name": "measure_selection_duration",
            "tool_payload": {
                "samples": [
                    {
                        "order": 1,
                        "source_start_index": 0,
                        "source_end_index": 0,
                    },
                    {
                        "order": 2,
                        "source_start_index": 1,
                        "source_end_index": 1,
                    },
                ]
            },
        },
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 2,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Introduces the first key idea.",
                        estimated_words=5,
                    ),
                    _anchored_sample(
                        order=2,
                        speaker_labels=["SPEAKER_01"],
                        source_start_index=1,
                        source_end_index=1,
                        segment_text="zeta eta theta iota kappa",
                        rationale="Captures the second important turn.",
                        estimated_words=5,
                    ),
                ],
            },
        },
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=2,
        ),
        console=console,
    )

    assert exit_code == 0
    assert len(_FakeProvider.instances) == 1
    assert len(_FakeProvider.instances[0].calls) == 2
    second_messages = _FakeProvider.instances[0].calls[1]["messages"]
    assert any(
        message.get("role") == "tool"
        and '"total_duration_ms": 2000' in str(message.get("content"))
        for message in second_messages
    )
    assert any(
        message.get("role") == "tool"
        and '"remaining_seconds_to_target": 28.0' in str(message.get("content"))
        for message in second_messages
    )
    run_dir = next(iter(sampling_root.iterdir()))
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["selected_duration_ms"] == 2000
    assert response_doc["attempts"][0]["tool_results"][0]["tool_name"] == (
        "measure_selection_duration"
    )
    assert response_doc["attempts"][0]["tool_results"][0]["result"][
        "total_duration_ms"
    ] == 2000
    assert response_doc["attempts"][0]["tool_results"][0]["result"]["budget"][
        "remaining_seconds_to_target"
    ] == 28.0


def test_run_llm_sampling_session_accepts_last_measured_selection_at_tool_round_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_long_duration_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "measure exact timing before finalizing",
            "content": "",
            "tool_name": "measure_selection_duration",
            "tool_payload": {
                "samples": [
                    {
                        "order": 1,
                        "source_start_index": 0,
                        "source_end_index": 1,
                    }
                ]
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    args = _build_args(
        manifest_path=manifest_path,
        sampling_root=sampling_root,
        samples=0,
        duration_preset="30s",
    )
    args.max_tool_rounds = 1

    exit_code = run_llm_sampling_session(
        args,
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["status"] == "completed"
    assert response_doc["selected_duration_ms"] == 30000
    assert response_doc["selected_samples"]
    assert response_doc["attempts"][0]["validation"]["status"] == "completed"
    assert "tool-round limit" in response_doc["attempts"][0]["validation"]["message"]


def test_run_llm_sampling_session_preserves_last_measured_failure_reason_at_tool_round_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_long_duration_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "measure exact timing before finalizing",
            "content": "",
            "tool_name": "measure_selection_duration",
            "tool_payload": {
                "samples": [
                    {
                        "order": 1,
                        "source_start_index": 0,
                        "source_end_index": 2,
                    }
                ]
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    args = _build_args(
        manifest_path=manifest_path,
        sampling_root=sampling_root,
        samples=0,
        duration_preset="30s",
    )
    args.max_tool_rounds = 1

    exit_code = run_llm_sampling_session(
        args,
        console=console,
    )

    assert exit_code == 1
    run_dir = next(iter(sampling_root.iterdir()))
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["status"] == "failed_validation"
    assert response_doc["attempts"][0]["validation"]["status"] == "failed_validation"
    assert (
        "outside the allowed tolerance band"
        in response_doc["attempts"][0]["validation"]["message"]
    )


def test_run_llm_sampling_session_accepts_combined_fragmented_same_speaker_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_fragmented_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "combine adjacent fragments",
            "content": "I am selecting one combined fragmented quote.",
            "tool_payload": {
                "episode_key": "episode://fragmented",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=1,
                        segment_text="alpha beta gamma delta",
                        rationale="This speaker thought is split across adjacent fragments.",
                        estimated_words=4,
                    )
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0


def test_run_llm_sampling_session_accepts_single_speaker_repair_across_short_cross_speaker_seam(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_seam_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "repair cross-speaker seam",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://seam",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=2,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Short off-speaker seam repaired back into one utterance.",
                        estimated_words=5,
                    )
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0


def test_run_llm_sampling_session_accepts_grounded_multi_speaker_reconstruction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_fragmented_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "grounded reconstructed exchange",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://fragmented",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_mode="multi_speaker_exchange",
                        speaker_labels=["SPEAKER_00", "SPEAKER_01"],
                        source_start_index=0,
                        source_end_index=2,
                        segment_text="alpha beta gamma delta epsilon zeta",
                        rationale="Keeps the short exchange readable while staying in one bounded span.",
                        estimated_words=6,
                    )
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    response_doc = json.loads(
        (
            run_dir / "episodes" / "sporc_fragmented_episode" / "response.json"
        ).read_text(encoding="utf-8")
    )
    assert response_doc["selected_samples"][0]["speaker_label"] == "SPEAKER_00"
    selected_segments_text = (
        run_dir / "episodes" / "sporc_fragmented_episode" / "selected_segments.txt"
    ).read_text(encoding="utf-8")
    assert "[SPEAKER_00]" in selected_segments_text
    assert "SPEAKER_00 + SPEAKER_01" not in selected_segments_text


def test_run_llm_sampling_session_prompt_preserves_disfluencies_with_readability_cleanup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta",
                        rationale="Readable excerpt.",
                        estimated_words=2,
                    )
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0
    messages = _FakeProvider.instances[0].calls[0]["messages"]
    assert "grounded reconstruction from a noisy transcript" in str(messages[0]["content"])
    assert "You have access to a local timing tool named `measure_selection_duration`." in str(
        messages[0]["content"]
    )
    assert "remaining seconds to target" in str(messages[0]["content"])
    assert "finish with one final `submit_samples` tool call" in str(messages[0]["content"])
    assert "Include `source_start_index`, `source_end_index`, `source_start_timestamp`" in str(
        messages[0]["content"]
    )
    assert "accurate depiction of the episode without requiring them to listen to the full source audio" in str(
        messages[0]["content"]
    )
    assert "Target sampled source duration for human evaluation: about 30 seconds" in str(
        messages[1]["content"]
    )
    assert "accurate depiction of what matters in the episode without listening to the full podcast" in str(
        messages[1]["content"]
    )
    assert "Return source indices and timestamps for every sample when you can." in str(messages[1]["content"])
    assert "Do not do duration arithmetic yourself." in str(
        messages[1]["content"]
    )
    assert "`remaining_seconds_to_target`, `in_budget`, and `recommended_action`" in str(
        messages[1]["content"]
    )


def test_run_llm_sampling_session_accepts_15m_duration_preset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Key opening line.",
                        estimated_words=5,
                    )
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
            duration_preset="15m",
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["duration_preset"] == "15m"
    assert summary["target_seconds_per_episode"] == 900


def test_run_llm_sampling_session_accepts_10m_duration_preset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Key opening line.",
                        estimated_words=5,
                    )
                ],
            },
        }
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
            duration_preset="10m",
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["duration_preset"] == "10m"
    assert summary["target_seconds_per_episode"] == 600


def test_resolve_settings_respects_max_tool_rounds(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    args = _build_args(manifest_path=manifest_path, sampling_root=sampling_root)
    args.max_tool_rounds = 7

    settings = _resolve_settings(args)

    assert settings["max_tool_rounds"] == 7


def test_run_llm_sampling_session_fails_invalid_submission_without_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "too short",
            "content": "First attempt is short.",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "target_words": 10,
                "selected_words": 2,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta",
                        rationale="Too short on purpose.",
                        estimated_words=2,
                    )
                ],
            },
        },
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(manifest_path=manifest_path, sampling_root=sampling_root),
        console=console,
    )

    assert exit_code == 1
    assert len(_FakeProvider.instances) == 1
    assert len(_FakeProvider.instances[0].calls) == 1
    response_doc = json.loads(
        (
            sampling_root
            / next(iter(sampling_root.iterdir())).name
            / "episodes"
            / "sporc_00001_episode"
            / "response.json"
        ).read_text(encoding="utf-8")
    )
    assert response_doc["status"] == "failed_validation"
    assert len(response_doc["attempts"]) == 1


def test_run_llm_sampling_session_rejects_text_only_episode_assets_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_episode_assets_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
            text_only=True,
        ),
        console=console,
    )

    assert exit_code == 1
    assert "cannot read `episode_assets/prepared_episodes_manifest.json` directly" in console.export_text()


def test_run_llm_sampling_session_accepts_missing_source_anchors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    {
                        "order": 1,
                        "speaker_mode": "single_speaker_repair",
                        "speaker_labels": ["SPEAKER_00"],
                        "segment_text": "alpha beta gamma delta epsilon",
                        "rationale": "Missing anchors on purpose.",
                        "estimated_words": 5,
                    }
                ],
            },
        },
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["status"] == "completed"
    assert response_doc["selected_samples"][0]["source_start_index"] == -1


def test_run_llm_sampling_session_accepts_reversed_source_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=1,
                        source_end_index=0,
                        segment_text="alpha beta gamma delta epsilon",
                        rationale="Broken range on purpose.",
                        estimated_words=5,
                    )
                ],
            },
        },
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["status"] == "completed"
    assert response_doc["selected_samples"][0]["source_start_index"] == 1
    assert response_doc["selected_samples"][0]["source_end_index"] == 0


def test_run_llm_sampling_session_allows_reconstructed_text_inside_cited_span(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=0,
                        segment_text="alpha beta late gamma delta epsilon",
                        rationale="Adds an invented token whose letters happened to fit the old character-subsequence check.",
                        estimated_words=6,
                    )
                ],
            },
        },
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["status"] == "completed"


def test_run_llm_sampling_session_accepts_non_integer_source_anchor_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    {
                        "order": 1,
                        "speaker_mode": "single_speaker_repair",
                        "speaker_labels": ["SPEAKER_00"],
                        "source_start_index": "zero",
                        "source_end_index": 0,
                        "source_start_timestamp": "00:00:00",
                        "source_end_timestamp": "00:00:01",
                        "segment_text": "alpha beta gamma delta epsilon",
                        "rationale": "Bad anchor type on purpose.",
                        "estimated_words": 5,
                    }
                ],
            },
        },
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    response_doc = json.loads(
        (run_dir / "episodes" / "sporc_00001_episode" / "response.json").read_text(
            encoding="utf-8"
        )
    )
    assert response_doc["status"] == "completed"
    assert response_doc["selected_samples"][0]["source_start_index"] == -1


def test_run_llm_sampling_session_accepts_single_speaker_claim_over_multi_speaker_span(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_fragmented_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://fragmented",
                "requested_samples": 1,
                "samples": [
                    _anchored_sample(
                        order=1,
                        speaker_labels=["SPEAKER_00"],
                        source_start_index=0,
                        source_end_index=2,
                        segment_text="alpha beta gamma delta epsilon zeta",
                        rationale="Incorrectly flattens the exchange.",
                        estimated_words=6,
                    )
                ],
            },
        },
    ]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    exit_code = run_llm_sampling_session(
        _build_args(
            manifest_path=manifest_path,
            sampling_root=sampling_root,
            samples=1,
        ),
        console=console,
    )

    assert exit_code == 0
    run_dir = next(iter(sampling_root.iterdir()))
    response_doc = json.loads(
        (
            run_dir / "episodes" / "sporc_fragmented_episode" / "response.json"
        ).read_text(encoding="utf-8")
    )
    assert response_doc["status"] == "completed"


def test_run_llm_sampling_session_fails_fast_without_api_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )

    exit_code = run_llm_sampling_session(
        _build_args(manifest_path=manifest_path, sampling_root=sampling_root),
        console=console,
    )

    assert exit_code == 1
    assert not sampling_root.exists()
    assert "ANTHROPIC_API_KEY is missing" in console.export_text()
