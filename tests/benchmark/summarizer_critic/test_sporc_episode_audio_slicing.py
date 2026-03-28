"""Coverage for the SPoRC episode-assets audio-slicing workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console

from card_framework.agents.dtos import AssistantMessage, Function, ToolCall
from card_framework.benchmark.summarizer_critic.sporc.episode_audio_slicing import (
    _RenderResult,
    _selected_rows_to_audio_slices,
    run_episode_audio_slicing_session,
)


def _write_episode_assets_manifest(tmp_path: Path) -> Path:
    source_transcript_path = tmp_path / "prepared_full.transcript.json"
    source_transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0,
                        "end_time": 300000,
                        "text": "source alpha beta gamma",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 300000,
                        "end_time": 600000,
                        "text": "source delta epsilon zeta",
                    },
                ],
                "metadata": {
                    "dataset": "sporc_episode_assets",
                    "episode_title": "Source Episode",
                },
            }
        ),
        encoding="utf-8",
    )
    sampled_transcript_path = tmp_path / "sampled.transcript.json"
    sampled_transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_99",
                        "start_time": 0,
                        "end_time": 1000,
                        "text": "sampled transcript text only",
                    }
                ],
                "metadata": {
                    "dataset": "sporc_episode_assets",
                    "episode_title": "Sampled Episode",
                },
            }
        ),
        encoding="utf-8",
    )
    source_audio_path = tmp_path / "audio.mp3"
    source_audio_path.write_bytes(b"ID3")
    summary_path = tmp_path / "episode_assets" / "prepared_episodes_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "dataset_manifest_path": str(tmp_path / "unused.json"),
                "output_root": str(summary_path.parent),
                "episode_count": 1,
                "episodes": [
                    {
                        "sample_id": "sporc_audio_episode_00001",
                        "episode_title": "Source Episode",
                        "episode_root": str(summary_path.parent / "episode_1"),
                        "transcript_path": str(sampled_transcript_path),
                        "source_transcript_path": str(source_transcript_path),
                        "sampled_transcript_path": str(sampled_transcript_path),
                        "source_audio_url": "https://example.com/audio.mp3",
                        "source_audio_path": str(source_audio_path),
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
    """Fake Anthropic provider that emits one successful tool-use response."""

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
        del tools, tool_choice, max_tokens
        self.calls.append({"messages": messages})
        if self._response_callback is not None:
            self._response_callback.on_start("Summarizer")
            self._response_callback.on_complete()
        assistant_message = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=Function(
                        name="submit_samples",
                        arguments=json.dumps(
                            {
                                "episode_key": "sporc_audio_episode_00001",
                                "target_seconds": 600,
                                "selected_seconds": 600.0,
                                "samples": [
                                    {
                                        "order": 1,
                                        "speaker_mode": "single_speaker_repair",
                                        "speaker_labels": ["SPEAKER_00"],
                                        "source_start_index": 0,
                                        "source_end_index": 0,
                                        "source_start_timestamp": "00:00:00",
                                        "source_end_timestamp": "00:05:00",
                                        "segment_text": "source alpha beta gamma",
                                        "rationale": "Opens the episode cleanly.",
                                        "estimated_words": 4,
                                    },
                                    {
                                        "order": 2,
                                        "speaker_mode": "single_speaker_repair",
                                        "speaker_labels": ["SPEAKER_01"],
                                        "source_start_index": 1,
                                        "source_end_index": 1,
                                        "source_start_timestamp": "00:05:00",
                                        "source_end_timestamp": "00:10:00",
                                        "segment_text": "source delta epsilon zeta",
                                        "rationale": "Carries the second half.",
                                        "estimated_words": 4,
                                    },
                                ],
                            }
                        ),
                    ),
                )
            ],
        )
        self.last_response_details = {
            "provider": "anthropic",
            "response": {
                "stop_reason": "tool_use",
                "content": "",
                "reasoning_content": None,
                "tool_calls": assistant_message.model_dump().get("tool_calls"),
                "usage": {"input_tokens": 1000, "output_tokens": 200},
            },
        }
        return assistant_message


def _build_args(*, manifest_path: Path, output_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_manifest=str(manifest_path),
        output_root=str(output_root),
        duration_preset="10m",
        episodes=1,
        select_episode="",
        provider="anthropic",
        model="claude-sonnet-4-6",
        thinking_effort="medium",
        disable_thinking=True,
        max_output_tokens=2048,
        max_tool_rounds=4,
        audio_format="mp3",
        sample_rate_hz=44100,
        channels=1,
        edge_fade_ms=20,
        mp3_bitrate="192k",
        timeout_seconds=30,
        max_estimated_cost_usd=0.0,
        words_per_minute=170.0,
        sample_factor=5.0,
    )


def test_selected_rows_to_audio_slices_orders_by_selection_order() -> None:
    rows = [
        {"selection_order": 2, "start_time": 3000, "end_time": 5000},
        {"selection_order": 1, "start_time": 0, "end_time": 2500},
    ]

    slices = _selected_rows_to_audio_slices(rows)

    assert [(item.start_time_ms, item.end_time_ms) for item in slices] == [
        (0, 2500),
        (3000, 5000),
    ]


def test_run_episode_audio_slicing_session_uses_source_transcript_and_writes_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_episode_assets_manifest(tmp_path)
    output_root = tmp_path / "runs"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    def _fake_render_episode_audio(**kwargs: Any) -> _RenderResult:
        output_dir = Path(kwargs["output_dir"])
        output_path = (output_dir / "final.mp3").resolve()
        output_path.write_bytes(b"ID3")
        return _RenderResult(
            output_path=output_path,
            output_format="mp3",
            source_audio_path=Path(kwargs["source_audio_path"]).resolve(),
            slice_count=2,
            selected_duration_ms=600000,
            rendered_duration_ms=600000,
        )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.episode_audio_slicing._render_episode_audio",
        _fake_render_episode_audio,
    )

    exit_code = run_episode_audio_slicing_session(
        _build_args(manifest_path=manifest_path, output_root=output_root),
        console=console,
    )

    assert exit_code == 0
    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["rendered_audio_count"] == 1
    assert summary["estimated_cost_usd"] == 0.006
    episode_summary = summary["episodes"][0]
    assert episode_summary["rendered_audio_path"].endswith("final.mp3")
    assert episode_summary["rendered_audio_duration_ms"] == 600000
    assert episode_summary["source_audio_path"].endswith("audio.mp3")
    experiments_text = (run_dir / "experiments.md").read_text(encoding="utf-8")
    assert "Estimated cost (USD): 0.006000" in experiments_text
    selected_rows = [
        json.loads(line)
        for line in (run_dir / "selected_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(selected_rows) == 2
    provider_call = _FakeProvider.instances[0].calls[0]
    user_prompt = provider_call["messages"][1]["content"]
    assert "source alpha beta gamma" in user_prompt
    assert "sampled transcript text only" not in user_prompt


def test_run_episode_audio_slicing_session_persists_render_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_episode_assets_manifest(tmp_path)
    output_root = tmp_path / "runs"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler._load_anthropic_provider_class",
        lambda: _FakeProvider,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.llm_sampler.RichConsoleResponseCallback",
        _NoOpRichCallback,
    )

    def _failing_render_episode_audio(**kwargs: Any) -> _RenderResult:
        del kwargs
        raise RuntimeError("missing measured timestamps")

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.episode_audio_slicing._render_episode_audio",
        _failing_render_episode_audio,
    )

    exit_code = run_episode_audio_slicing_session(
        _build_args(manifest_path=manifest_path, output_root=output_root),
        console=console,
    )

    assert exit_code == 1
    run_dir = next(iter(output_root.iterdir()))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed_with_failures"
    episode_summary = summary["episodes"][0]
    assert episode_summary["status"] == "failed_render"
    assert "missing measured timestamps" in episode_summary["render_error"]
