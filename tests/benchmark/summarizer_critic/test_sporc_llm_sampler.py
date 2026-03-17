"""Coverage for the SPoRC Anthropic-backed transcript sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console

from card_framework.agents.dtos import AssistantMessage, Function, ToolCall
from card_framework.benchmark.summarizer_critic.sporc.llm_sampler import (
    run_llm_sampling_session,
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


def _write_fragmented_manifest(tmp_path: Path) -> Path:
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
                "turn_data_path": str(tmp_path / "speakerTurnData.jsonl.gz"),
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


def _write_multi_manifest(tmp_path: Path, *, episode_count: int = 4) -> Path:
    manifest_samples: list[dict[str, Any]] = []
    transcripts_root = tmp_path / "transcripts"
    transcripts_root.mkdir(parents=True, exist_ok=True)
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
                "turn_data_path": str(tmp_path / "speakerTurnData.jsonl.gz"),
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

        tool_arguments = json.dumps(planned["tool_payload"])
        assistant_message = AssistantMessage(
            content=str(planned.get("content", "")),
            reasoning_content=str(planned.get("thought", "")) or None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=Function(
                        name="submit_samples",
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
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_manifest=str(manifest_path),
        sampling_root=str(sampling_root),
        duration_preset="30s",
        samples=samples,
        episodes=1,
        select_episode=select_episode,
        words_per_minute=20.0,
        sample_factor=1.0,
        model="claude-opus-4-6",
        thinking_effort="medium",
        disable_thinking=disable_thinking,
        max_output_tokens=2048,
    )


def test_run_llm_sampling_session_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
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
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "alpha beta gamma delta epsilon",
                        "rationale": "Introduces the first key idea.",
                        "estimated_words": 5,
                    },
                    {
                        "order": 2,
                        "speaker_label": "SPEAKER_01",
                        "segment_text": "zeta eta theta iota kappa",
                        "rationale": "Captures the second important turn.",
                        "estimated_words": 5,
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
    stream_lines = (
        run_dir / "episodes" / "sporc_00001_episode" / "stream.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert any('"event": "thinking_token"' in line for line in stream_lines)
    assert any('"event": "content_token"' in line for line in stream_lines)
    exported_rows = (
        run_dir / "selected_samples.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(exported_rows) == 2


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
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "alpha beta gamma delta epsilon",
                        "rationale": "Important opening cue.",
                        "estimated_words": 5,
                    },
                    {
                        "order": 2,
                        "speaker_label": "SPEAKER_01",
                        "segment_text": "zeta eta theta iota kappa",
                        "rationale": "Second important cue.",
                        "estimated_words": 5,
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
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "episode 2 alpha beta gamma delta",
                        "rationale": "Second manifest episode.",
                        "estimated_words": 6,
                    }
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
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "episode 3 alpha beta gamma delta",
                        "rationale": "Third manifest episode.",
                        "estimated_words": 6,
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
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "alpha beta gamma delta epsilon",
                        "rationale": "Key opening line.",
                        "estimated_words": 5,
                    }
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
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "alpha beta gamma delta epsilon",
                        "rationale": "Key opening line.",
                        "estimated_words": 5,
                    }
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
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "alpha beta gamma delta",
                        "rationale": "This speaker thought is split across adjacent fragments.",
                        "estimated_words": 4,
                    }
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


def test_run_llm_sampling_session_accepts_non_verbatim_segment_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    sampling_root = tmp_path / "sampling"
    console = Console(record=True, width=120)
    _FakeProvider.instances.clear()
    _FakeProvider.planned_responses = [
        {
            "thought": "grounded but not exact",
            "content": "",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "requested_samples": 1,
                "samples": [
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "alpha beta",
                        "rationale": "Condenses the opening idea.",
                        "estimated_words": 2,
                    }
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
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "alpha beta",
                        "rationale": "Readable excerpt.",
                        "estimated_words": 2,
                    }
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
    assert "including disfluencies, repetitions, and filler words" in str(
        messages[0]["content"]
    )
    assert "understandable to a human reviewer" in str(messages[1]["content"])


def test_run_llm_sampling_session_retries_one_invalid_submission(
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
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "alpha beta",
                        "rationale": "Too short on purpose.",
                        "estimated_words": 2,
                    }
                ],
            },
        },
        {
            "thought": "fixed",
            "content": "Second attempt is in budget.",
            "tool_payload": {
                "episode_key": "episode://sampler",
                "target_words": 10,
                "selected_words": 10,
                "samples": [
                    {
                        "order": 1,
                        "speaker_label": "SPEAKER_00",
                        "segment_text": "alpha beta gamma delta epsilon",
                        "rationale": "Restores the opening context.",
                        "estimated_words": 5,
                    },
                    {
                        "order": 2,
                        "speaker_label": "SPEAKER_01",
                        "segment_text": "zeta eta theta iota kappa",
                        "rationale": "Adds the next key exchange.",
                        "estimated_words": 5,
                    },
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

    assert exit_code == 0
    assert len(_FakeProvider.instances) == 1
    assert len(_FakeProvider.instances[0].calls) == 2
    response_doc = json.loads(
        (
            sampling_root
            / next(iter(sampling_root.iterdir())).name
            / "episodes"
            / "sporc_00001_episode"
            / "response.json"
        ).read_text(encoding="utf-8")
    )
    assert response_doc["status"] == "completed"
    assert len(response_doc["attempts"]) == 2


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
