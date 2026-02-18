"""Unit tests for DeepSeek tool-calling loop and local constraint tool."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from openai import APIStatusError

from audio2script_and_summarizer import summarizer_deepseek as deepseek_summary


def _tool_line(
    text: str,
    *,
    speaker: str = "Speaker 0",
    source_ids: list[str] | None = None,
) -> deepseek_summary.DialogueLinePayload:
    """Build a minimal tool-line payload for tests."""
    return {
        "speaker": speaker,
        "text": text,
        "emo_text": "Neutral",
        "emo_alpha": 0.6,
        "source_segment_ids": source_ids or ["seg_00000"],
    }


def _sample_payload() -> dict[str, list[dict[str, object]]]:
    """Build a strict JSON payload that matches the script schema."""
    return {
        "dialogue": [
            {
                "speaker": "Speaker 0",
                "text": "hello world",
                "emo_text": "Neutral",
                "emo_alpha": 0.6,
                "source_segment_ids": ["seg_00000"],
            }
        ]
    }


def _segments() -> list[deepseek_summary.TranscriptSegment]:
    """Build minimal transcript segments for tool-driven reads."""
    return [
        deepseek_summary.TranscriptSegment(
            segment_id="seg_00000",
            speaker="Speaker 0",
            text="hello world",
            start_time=0.0,
            end_time=1.0,
        ),
        deepseek_summary.TranscriptSegment(
            segment_id="seg_00001",
            speaker="Speaker 0",
            text="next sentence",
            start_time=1.0,
            end_time=2.0,
        ),
    ]


def _tool_call_delta(
    *,
    index: int,
    tool_name: str | None = None,
    arguments: str | None = None,
    tool_call_id: str | None = None,
) -> SimpleNamespace:
    """Build a streamed tool-call delta object."""
    function_payload = SimpleNamespace(
        name=tool_name,
        arguments=arguments,
    )
    return SimpleNamespace(
        index=index,
        id=tool_call_id,
        type="function",
        function=function_payload,
    )


def _stream_chunk(
    *,
    finish_reason: str | None = None,
    reasoning: str | None = None,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    """Build one streamed completion chunk for tool-loop tests."""
    delta_payload = SimpleNamespace(
        reasoning_content=reasoning,
        content=content,
        tool_calls=tool_calls,
    )
    choice_payload = SimpleNamespace(finish_reason=finish_reason, delta=delta_payload)
    return SimpleNamespace(choices=[choice_payload])


def _api_status_error(status_code: int, message: str) -> APIStatusError:
    """Create an APIStatusError with deterministic request/response bodies."""
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": message})
    return APIStatusError(message=message, response=response, body={"error": message})


def test_evaluate_script_constraints_tool_fails_on_budget_overrun() -> None:
    """Return fail status when generated dialogue exceeds target words."""
    result = deepseek_summary._evaluate_script_constraints_tool(  # noqa: SLF001
        dialogue_payload=[_tool_line("one two three four")],
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0"},
        word_budget=2,
        word_budget_tolerance=0.0,
        source_word_count=20,
    )

    assert result["status"] == "fail"
    assert result["word_budget"]["total_words"] == 4
    assert result["word_budget"]["in_range"] is False
    assert result["validation"]["is_valid"] is True


def test_evaluate_script_constraints_tool_normalizes_recoverable_segment_ids() -> None:
    """Normalize recoverable short source IDs before speaker validation."""
    result = deepseek_summary._evaluate_script_constraints_tool(  # noqa: SLF001
        dialogue_payload=[_tool_line("hello world", source_ids=["seg_0004"])],
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00004": "Speaker 0"},
        word_budget=None,
        word_budget_tolerance=0.0,
        source_word_count=20,
    )

    assert result["validation"]["is_valid"] is True
    assert result["validation"]["issues"] == []
    assert result["repaired_dialogue"][0]["source_segment_ids"] == ["seg_00004"]


def test_build_system_prompt_enforces_zero_padded_segment_ids() -> None:
    """Include explicit zero-padding segment-ID rule in prompt contract."""
    prompt = deepseek_summary._build_system_prompt(  # noqa: SLF001
        allowed_speakers={"Speaker 0"},
        target_min_lines=8,
        target_max_lines=12,
        word_budget=120,
        target_minutes=None,
        avg_wpm=None,
        word_budget_tolerance=0.05,
        require_tool_call=True,
        tool_mode="full_agentic",
        max_repeated_write_overwrites=2,
    )

    assert "example: seg_00004" in prompt
    assert "never seg_0004" not in prompt
    assert "batched `lines` payloads" in prompt
    assert "Prioritize broad transcript coverage early" in prompt
    assert "full candidate dialogue list" in prompt
    assert "Prefer one material rewrite" in prompt


def test_stream_assistant_turn_invokes_chunk_callback() -> None:
    """Invoke per-chunk callback for heartbeat-style side-channel updates."""

    class _FakeCompletions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            return [
                _stream_chunk(reasoning="r1"),
                _stream_chunk(content='{"dialogue":'),
                _stream_chunk(finish_reason="stop", content="[]}"),
            ]

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    chunk_ticks = 0

    def _tick() -> None:
        nonlocal chunk_ticks
        chunk_ticks += 1

    turn = deepseek_summary._stream_assistant_turn(  # noqa: SLF001
        client=cast(object, fake_client),
        request_payload={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        stream_event_callback=None,
        on_chunk_callback=_tick,
    )

    assert chunk_ticks == 3
    assert turn.finish_reason == "stop"
    assert turn.content == '{"dialogue":[]}'


def test_request_completion_tool_loop_runs_until_pass() -> None:
    """Execute multi-round tool loop and accept JSON only after pass result."""

    class _FakeCompletions:
        def __init__(self) -> None:
            self.call_count = 0
            self.seen_stream_values: list[object] = []

        def create(self, **kwargs: object) -> object:
            self.call_count += 1
            self.seen_stream_values.append(kwargs.get("stream"))
            if self.call_count == 1:
                tool_arguments = json.dumps(
                    {
                        "dialogue": [
                            _tool_line("hello world this line exceeds the target budget")
                        ]
                    }
                )
                split_at = max(1, len(tool_arguments) // 2)
                return [
                    _stream_chunk(reasoning="check constraints"),
                    _stream_chunk(
                        tool_calls=[
                            _tool_call_delta(
                                index=0,
                                tool_name=deepseek_summary.EVALUATE_SCRIPT_TOOL_NAME,
                                arguments=tool_arguments[:split_at],
                                tool_call_id="call_1",
                            )
                        ]
                    ),
                    _stream_chunk(
                        finish_reason="tool_calls",
                        tool_calls=[
                            _tool_call_delta(
                                index=0,
                                arguments=tool_arguments[split_at:],
                            )
                        ],
                    ),
                ]
            if self.call_count == 2:
                tool_arguments = json.dumps(
                    {
                        "dialogue": [
                            _tool_line("hello this line has natural spoken pacing")
                        ]
                    }
                )
                write_arguments = json.dumps(
                    {
                        "mode": "overwrite",
                        "content": json.dumps(_sample_payload()),
                    }
                )
                eval_split_at = max(1, len(tool_arguments) // 2)
                write_split_at = max(1, len(write_arguments) // 2)
                return [
                    _stream_chunk(reasoning="try again"),
                    _stream_chunk(
                        tool_calls=[
                            _tool_call_delta(
                                index=0,
                                tool_name=deepseek_summary.EVALUATE_SCRIPT_TOOL_NAME,
                                arguments=tool_arguments[:eval_split_at],
                                tool_call_id="call_2",
                            ),
                            _tool_call_delta(
                                index=1,
                                tool_name=deepseek_summary.WRITE_OUTPUT_SEGMENT_TOOL_NAME,
                                arguments=write_arguments[:write_split_at],
                                tool_call_id="call_2_write",
                            )
                        ]
                    ),
                    _stream_chunk(
                        finish_reason="tool_calls",
                        tool_calls=[
                            _tool_call_delta(
                                index=0,
                                arguments=tool_arguments[eval_split_at:],
                            ),
                            _tool_call_delta(
                                index=1,
                                arguments=write_arguments[write_split_at:],
                            )
                        ],
                    ),
                ]
            return [
                _stream_chunk(reasoning="final"),
                _stream_chunk(
                    finish_reason="stop",
                    content=deepseek_summary.STAGED_OUTPUT_READY_MARKER,
                ),
            ]

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    stream_events: list[dict[str, object]] = []
    settings = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-reasoner",
        max_completion_tokens=64000,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=None,
        auto_beta=False,
        agent_tool_loop=True,
        agent_tool_mode="constraints_only",
        agent_max_tool_rounds=10,
    )

    parsed_script, used_repair, tool_rounds, tool_call_counts = deepseek_summary._request_deepseek_completion(  # noqa: SLF001
        client=cast(object, fake_client),
        settings=settings,
        transcript_text="[seg_00000|Speaker 0]: hello world",
        transcript_segments=_segments(),
        transcript_manifest="segment_count=2",
        system_prompt="test prompt",
        retry_context=None,
        endpoint_mode="stable",
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0"},
        word_budget=7,
        word_budget_tolerance=0.0,
        source_word_count=20,
        stream_event_callback=stream_events.append,
    )

    assert parsed_script.dialogue[0].text == "hello world"
    assert used_repair is False
    assert tool_rounds == 3
    assert tool_call_counts[deepseek_summary.EVALUATE_SCRIPT_TOOL_NAME] == 2
    assert tool_call_counts[deepseek_summary.WRITE_OUTPUT_SEGMENT_TOOL_NAME] == 1
    assert all(value is True for value in completions.seen_stream_values)
    event_names = [str(event.get("event", "")) for event in stream_events]
    assert "start" in event_names
    assert "done" in event_names
    token_phases = [
        str(event.get("phase", ""))
        for event in stream_events
        if event.get("event") == "token"
    ]
    assert "status" in token_phases
    assert "reasoning" in token_phases
    assert "answer" in token_phases
    assert "tool_call" in token_phases
    assert "tool_result" in token_phases


def test_request_completion_tool_loop_requires_pass_before_final_json() -> None:
    """Reject final JSON when no passing tool result was produced in loop."""

    class _FakeCompletions:
        def __init__(self) -> None:
            self.call_count = 0

        def create(self, **kwargs: object) -> object:
            self.call_count += 1
            return [
                _stream_chunk(reasoning="skip tool"),
                _stream_chunk(
                    finish_reason="stop",
                    content=json.dumps(_sample_payload()),
                ),
            ]

    settings = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-reasoner",
        max_completion_tokens=64000,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=None,
        auto_beta=False,
        agent_tool_loop=True,
        agent_tool_mode="constraints_only",
        agent_max_tool_rounds=1,
    )
    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    stream_events: list[dict[str, object]] = []

    with pytest.raises(deepseek_summary.ToolLoopExhaustedError):
        deepseek_summary._request_deepseek_completion(  # noqa: SLF001
            client=cast(object, fake_client),
            settings=settings,
            transcript_text="[seg_00000|Speaker 0]: hello world",
            transcript_segments=_segments(),
            transcript_manifest="segment_count=2",
            system_prompt="test prompt",
            retry_context=None,
            endpoint_mode="stable",
            allowed_speakers={"Speaker 0"},
            segment_speaker_map={"seg_00000": "Speaker 0"},
            word_budget=2,
            word_budget_tolerance=0.0,
            source_word_count=20,
            stream_event_callback=stream_events.append,
        )
    assert completions.call_count == 1
    status_messages = [
        str(event.get("text", ""))
        for event in stream_events
        if event.get("event") == "token" and event.get("phase") == "status"
    ]
    assert any("before a passing tool result" in message for message in status_messages)


def test_request_completion_full_agentic_requires_read_tool() -> None:
    """Reject final JSON in full-agentic mode until transcript read tool is used."""

    class _FakeCompletions:
        def __init__(self) -> None:
            self.call_count = 0

        def create(self, **kwargs: object) -> object:
            self.call_count += 1
            return [
                _stream_chunk(reasoning="skip read tool"),
                _stream_chunk(
                    finish_reason="stop",
                    content=json.dumps(_sample_payload()),
                ),
            ]

    settings = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-chat",
        max_completion_tokens=8192,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=0.2,
        auto_beta=False,
        agent_tool_loop=True,
        agent_tool_mode="full_agentic",
        agent_max_tool_rounds=1,
        agent_read_max_lines=10,
    )
    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    stream_events: list[dict[str, object]] = []

    with pytest.raises(deepseek_summary.ToolLoopExhaustedError):
        deepseek_summary._request_deepseek_completion(  # noqa: SLF001
            client=cast(object, fake_client),
            settings=settings,
            transcript_text="[seg_00000|Speaker 0]: hello world",
            transcript_segments=_segments(),
            transcript_manifest="segment_count=2",
            system_prompt="test prompt",
            retry_context=None,
            endpoint_mode="stable",
            allowed_speakers={"Speaker 0"},
            segment_speaker_map={"seg_00000": "Speaker 0", "seg_00001": "Speaker 0"},
            word_budget=None,
            word_budget_tolerance=0.05,
            source_word_count=20,
            stream_event_callback=stream_events.append,
        )
    assert completions.call_count == 1
    status_messages = [
        str(event.get("text", ""))
        for event in stream_events
        if event.get("event") == "token" and event.get("phase") == "status"
    ]
    assert any("read_transcript_lines was not called" in message for message in status_messages)


def test_request_completion_requires_write_tool_before_final_json() -> None:
    """Reject finalization until write_output_segment succeeds at least once."""

    class _FakeCompletions:
        def __init__(self) -> None:
            self.call_count = 0

        def create(self, **kwargs: object) -> object:
            self.call_count += 1
            if self.call_count == 1:
                tool_arguments = json.dumps(
                    {
                        "dialogue": [
                            _tool_line("hello this line has natural spoken pacing")
                        ]
                    }
                )
                split_at = max(1, len(tool_arguments) // 2)
                return [
                    _stream_chunk(reasoning="evaluate pass"),
                    _stream_chunk(
                        tool_calls=[
                            _tool_call_delta(
                                index=0,
                                tool_name=deepseek_summary.EVALUATE_SCRIPT_TOOL_NAME,
                                arguments=tool_arguments[:split_at],
                                tool_call_id="call_eval",
                            )
                        ],
                    ),
                    _stream_chunk(
                        finish_reason="tool_calls",
                        tool_calls=[
                            _tool_call_delta(
                                index=0,
                                arguments=tool_arguments[split_at:],
                            )
                        ],
                    ),
                ]
            return [
                _stream_chunk(reasoning="skip write"),
                _stream_chunk(
                    finish_reason="stop",
                    content=json.dumps(_sample_payload()),
                ),
            ]

    settings = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-reasoner",
        max_completion_tokens=64000,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=None,
        auto_beta=False,
        agent_tool_loop=True,
        agent_tool_mode="constraints_only",
        agent_max_tool_rounds=2,
    )
    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    stream_events: list[dict[str, object]] = []

    with pytest.raises(deepseek_summary.ToolLoopExhaustedError):
        deepseek_summary._request_deepseek_completion(  # noqa: SLF001
            client=cast(object, fake_client),
            settings=settings,
            transcript_text="[seg_00000|Speaker 0]: hello world",
            transcript_segments=_segments(),
            transcript_manifest="segment_count=2",
            system_prompt="test prompt",
            retry_context=None,
            endpoint_mode="stable",
            allowed_speakers={"Speaker 0"},
            segment_speaker_map={"seg_00000": "Speaker 0"},
            word_budget=None,
            word_budget_tolerance=0.0,
            source_word_count=20,
            stream_event_callback=stream_events.append,
        )
    assert completions.call_count == 2
    status_messages = [
        str(event.get("text", ""))
        for event in stream_events
        if event.get("event") == "token" and event.get("phase") == "status"
    ]
    assert any(
        "write_output_segment has no successful writes yet" in message
        for message in status_messages
    )


def test_request_completion_auto_salvages_staged_output_on_round_exhaustion() -> None:
    """Accept staged JSON when round limit is reached and local checks pass."""

    class _FakeCompletions:
        def __init__(self) -> None:
            self.call_count = 0

        def create(self, **kwargs: object) -> object:
            self.call_count += 1
            payload = {
                "dialogue": [
                    {
                        "speaker": "Speaker 0",
                        "text": "hello world this sentence has natural spoken pacing today",
                        "emo_text": "Neutral",
                        "emo_alpha": 0.6,
                        "source_segment_ids": ["seg_00000"],
                    }
                ]
            }
            write_arguments = json.dumps(
                {
                    "mode": "overwrite",
                    "content": json.dumps(payload),
                }
            )
            split_at = max(1, len(write_arguments) // 2)
            return [
                _stream_chunk(reasoning="stage output"),
                _stream_chunk(
                    tool_calls=[
                        _tool_call_delta(
                            index=0,
                            tool_name=deepseek_summary.WRITE_OUTPUT_SEGMENT_TOOL_NAME,
                            arguments=write_arguments[:split_at],
                            tool_call_id="call_write",
                        )
                    ],
                ),
                _stream_chunk(
                    finish_reason="tool_calls",
                    tool_calls=[
                        _tool_call_delta(
                            index=0,
                            arguments=write_arguments[split_at:],
                        )
                    ],
                ),
            ]

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-reasoner",
        max_completion_tokens=64000,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=None,
        auto_beta=False,
        agent_tool_loop=True,
        agent_tool_mode="constraints_only",
        agent_max_tool_rounds=1,
    )

    parsed_script, used_repair, tool_rounds, tool_call_counts = deepseek_summary._request_deepseek_completion(  # noqa: SLF001
        client=cast(object, fake_client),
        settings=settings,
        transcript_text="[seg_00000|Speaker 0]: hello world",
        transcript_segments=_segments(),
        transcript_manifest="segment_count=2",
        system_prompt="test prompt",
        retry_context=None,
        endpoint_mode="stable",
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0", "seg_00001": "Speaker 0"},
        word_budget=None,
        word_budget_tolerance=0.0,
        source_word_count=20,
    )

    assert completions.call_count == 1
    assert tool_rounds == 1
    assert tool_call_counts[deepseek_summary.WRITE_OUTPUT_SEGMENT_TOOL_NAME] == 1
    assert used_repair is False
    assert parsed_script.dialogue[0].text.startswith("hello world this sentence")


def test_request_completion_retry_resume_detects_reread_from_start() -> None:
    """Emit retry-resume telemetry when a resumed retry rereads index zero."""

    class _FakeCompletions:
        def __init__(self) -> None:
            self.call_count = 0
            self.last_messages: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.call_count += 1
            self.last_messages = cast(list[dict[str, object]], kwargs["messages"])
            read_arguments = json.dumps({"start_index": 0, "end_index": 1})
            split_at = max(1, len(read_arguments) // 2)
            return [
                _stream_chunk(reasoning="resume retry"),
                _stream_chunk(
                    tool_calls=[
                        _tool_call_delta(
                            index=0,
                            tool_name=deepseek_summary.READ_TRANSCRIPT_LINES_TOOL_NAME,
                            arguments=read_arguments[:split_at],
                            tool_call_id="call_read",
                        )
                    ],
                ),
                _stream_chunk(
                    finish_reason="tool_calls",
                    tool_calls=[
                        _tool_call_delta(
                            index=0,
                            arguments=read_arguments[split_at:],
                        )
                    ],
                ),
            ]

    continuation = deepseek_summary.RetryContinuationState(
        read_ranges=[(0, 20), (21, 40)],
        max_read_index=40,
        write_tool_succeeded=False,
        latest_constraints_status="fail",
        last_validation_issues=["No staged JSON was available after tool-loop round exhaustion."],
        staged_output_present=False,
        staged_output_valid_json=False,
    )
    retry_context = deepseek_summary.RetryContext(
        attempt_index=1,
        endpoint_mode="stable",
        error_type="tool_loop_exhausted",
        error_digest="Tool loop exhausted: max_tool_rounds_reached_no_staged_output",
        continuation=continuation,
    )
    settings = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-chat",
        max_completion_tokens=8192,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=0.2,
        auto_beta=False,
        agent_tool_loop=True,
        agent_tool_mode="full_agentic",
        agent_max_tool_rounds=1,
        agent_loop_exhaustion_policy="fail_fast",
    )
    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    stream_events: list[dict[str, object]] = []

    with pytest.raises(deepseek_summary.ToolLoopExhaustedError):
        deepseek_summary._request_deepseek_completion(  # noqa: SLF001
            client=cast(object, fake_client),
            settings=settings,
            transcript_text="[seg_00000|Speaker 0]: hello world",
            transcript_segments=_segments(),
            transcript_manifest="segment_count=2",
            system_prompt="test prompt",
            retry_context=retry_context,
            endpoint_mode="stable",
            allowed_speakers={"Speaker 0"},
            segment_speaker_map={"seg_00000": "Speaker 0", "seg_00001": "Speaker 0"},
            word_budget=None,
            word_budget_tolerance=0.05,
            source_word_count=20,
            stream_event_callback=stream_events.append,
        )

    system_messages = [
        cast(str, message.get("content", ""))
        for message in completions.last_messages
        if message.get("role") == "system"
    ]
    assert any("Retry resume guard" in message for message in system_messages)
    retry_events = [
        event for event in stream_events if event.get("event") == "retry_resume_telemetry"
    ]
    assert any(event.get("retry_resume_used") is True for event in retry_events)
    assert any(
        event.get("retry_read_from_start_detected") is True for event in retry_events
    )


def test_request_completion_blocks_repeated_identical_overwrite_writes() -> None:
    """Reject repeated identical overwrite writes and emit a revision hint."""

    class _FakeCompletions:
        def __init__(self) -> None:
            self.call_count = 0

        def create(self, **kwargs: object) -> object:
            self.call_count += 1
            payload = {
                "dialogue": [
                    {
                        "speaker": "Speaker 0",
                        "text": "hello world this sentence has natural spoken pacing today",
                        "emo_text": "Neutral",
                        "emo_alpha": 0.6,
                        "source_segment_ids": ["seg_00000"],
                    }
                ]
            }
            write_arguments = json.dumps(
                {
                    "mode": "overwrite",
                    "content": json.dumps(payload),
                }
            )
            split_at = max(1, len(write_arguments) // 2)
            return [
                _stream_chunk(reasoning="stage output repeatedly"),
                _stream_chunk(
                    tool_calls=[
                        _tool_call_delta(
                            index=0,
                            tool_name=deepseek_summary.WRITE_OUTPUT_SEGMENT_TOOL_NAME,
                            arguments=write_arguments[:split_at],
                            tool_call_id=f"call_write_{self.call_count}",
                        )
                    ],
                ),
                _stream_chunk(
                    finish_reason="tool_calls",
                    tool_calls=[
                        _tool_call_delta(
                            index=0,
                            arguments=write_arguments[split_at:],
                        )
                    ],
                ),
            ]

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    stream_events: list[dict[str, object]] = []
    settings = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-reasoner",
        max_completion_tokens=64000,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=None,
        auto_beta=False,
        agent_tool_loop=True,
        agent_tool_mode="constraints_only",
        agent_max_tool_rounds=2,
        agent_max_repeated_write_overwrites=1,
    )

    parsed_script, _, tool_rounds, tool_call_counts = deepseek_summary._request_deepseek_completion(  # noqa: SLF001
        client=cast(object, fake_client),
        settings=settings,
        transcript_text="[seg_00000|Speaker 0]: hello world",
        transcript_segments=_segments(),
        transcript_manifest="segment_count=2",
        system_prompt="test prompt",
        retry_context=None,
        endpoint_mode="stable",
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0", "seg_00001": "Speaker 0"},
        word_budget=None,
        word_budget_tolerance=0.0,
        source_word_count=20,
        stream_event_callback=stream_events.append,
    )

    assert completions.call_count == 2
    assert tool_rounds == 2
    assert tool_call_counts[deepseek_summary.WRITE_OUTPUT_SEGMENT_TOOL_NAME] == 2
    assert parsed_script.dialogue[0].speaker == "Speaker 0"
    tool_result_text = [
        str(event.get("text", ""))
        for event in stream_events
        if event.get("event") == "token" and event.get("phase") == "tool_result"
    ]
    assert any("Repeated overwrite detected" in text for text in tool_result_text)


def test_read_transcript_lines_tool_clamps_to_bounds() -> None:
    """Clamp read ranges to transcript bounds and max_lines limit."""
    result = deepseek_summary._read_transcript_lines_tool(  # noqa: SLF001
        transcript_segments=_segments(),
        start_index=-5,
        end_index=50,
        max_lines=1,
    )
    assert result["status"] == "ok"
    assert result["returned_count"] == 1
    assert result["returned_start_index"] == 0
    assert result["returned_end_index"] == 0


def test_count_words_tool_counts_lines_payload() -> None:
    """Count words from explicit lines payload."""
    result = deepseek_summary._count_words_tool(  # noqa: SLF001
        text=None,
        lines=["hello world", "three word line"],
    )
    assert result["status"] == "ok"
    assert result["total_words"] == 5
    assert result["line_word_counts"] == [2, 3]


def test_generate_summary_deepseek_falls_back_model_on_tool_protocol_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback from reasoner to chat when tool protocol is rejected."""
    seen_models: list[str] = []
    seen_max_tokens: list[int] = []
    stream_events: list[dict[str, object]] = []

    def fake_build_client(**kwargs: object) -> object:
        return kwargs["base_url"]

    def fake_request(
        **kwargs: object,
    ) -> tuple[deepseek_summary.PodcastScript, bool, int, dict[str, int]]:
        settings = cast(deepseek_summary.DeepSeekRequestSettings, kwargs["settings"])
        seen_models.append(settings.model)
        seen_max_tokens.append(settings.max_completion_tokens)
        if settings.model == "deepseek-reasoner":
            raise _api_status_error(400, "tool_choice is unsupported")
        return (
            deepseek_summary.PodcastScript(
                dialogue=[
                    deepseek_summary.DialogueLine(
                        speaker="Speaker 0",
                        text="hello world",
                        emo_text="Neutral",
                        emo_alpha=0.6,
                        source_segment_ids=["seg_00000"],
                    )
                ]
            ),
            False,
            2,
            {deepseek_summary.EVALUATE_SCRIPT_TOOL_NAME: 1},
        )

    monkeypatch.setattr(deepseek_summary, "_build_deepseek_client", fake_build_client)
    monkeypatch.setattr(deepseek_summary, "_request_deepseek_completion", fake_request)

    result = deepseek_summary.generate_summary_deepseek(
        transcript_text="[seg_00000|Speaker 0]: hello world",
        transcript_segments=_segments(),
        api_key="test-key",
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0"},
        source_word_count=2,
        settings=deepseek_summary.DeepSeekRequestSettings(
            model="deepseek-reasoner",
            max_completion_tokens=64000,
            request_timeout_seconds=30.0,
            http_retries=1,
            temperature=None,
            auto_beta=False,
            agent_tool_loop=True,
            agent_tool_mode="constraints_only",
            agent_max_tool_rounds=10,
            agent_allow_model_fallback=True,
        ),
        segment_count=1,
        retry_context=None,
        word_budget=2,
        target_minutes=None,
        avg_wpm=None,
        word_budget_tolerance=0.0,
        stream_event_callback=stream_events.append,
    )

    assert seen_models == ["deepseek-reasoner", "deepseek-chat"]
    assert seen_max_tokens == [64000, 8192]
    assert result.model == "deepseek-chat"
    fallback_messages = [
        str(event.get("text", ""))
        for event in stream_events
        if event.get("event") == "token" and event.get("phase") == "status"
    ]
    assert any("retrying with deepseek-chat" in message for message in fallback_messages)


def test_generate_summary_deepseek_does_not_fallback_model_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep reasoner-only model path unless fallback is explicitly enabled."""
    seen_models: list[str] = []

    def fake_build_client(**kwargs: object) -> object:
        return kwargs["base_url"]

    def fake_request(
        **kwargs: object,
    ) -> tuple[deepseek_summary.PodcastScript, bool, int, dict[str, int]]:
        settings = cast(deepseek_summary.DeepSeekRequestSettings, kwargs["settings"])
        seen_models.append(settings.model)
        raise _api_status_error(400, "tool_choice is unsupported")

    monkeypatch.setattr(deepseek_summary, "_build_deepseek_client", fake_build_client)
    monkeypatch.setattr(deepseek_summary, "_request_deepseek_completion", fake_request)

    with pytest.raises(deepseek_summary.GenerationAttemptError) as wrapped_error:
        deepseek_summary.generate_summary_deepseek(
            transcript_text="[seg_00000|Speaker 0]: hello world",
            transcript_segments=_segments(),
            api_key="test-key",
            allowed_speakers={"Speaker 0"},
            segment_speaker_map={"seg_00000": "Speaker 0"},
            source_word_count=2,
            settings=deepseek_summary.DeepSeekRequestSettings(
                model="deepseek-reasoner",
                max_completion_tokens=64000,
                request_timeout_seconds=30.0,
                http_retries=1,
                temperature=None,
                auto_beta=False,
                agent_tool_loop=True,
                agent_tool_mode="constraints_only",
                agent_max_tool_rounds=10,
            ),
            segment_count=1,
            retry_context=None,
            word_budget=2,
            target_minutes=None,
            avg_wpm=None,
            word_budget_tolerance=0.0,
            stream_event_callback=None,
        )

    assert seen_models == ["deepseek-reasoner"]
    assert isinstance(wrapped_error.value.cause, APIStatusError)


def test_resolve_summary_report_path_defaults_to_output_sidecar() -> None:
    """Build default sidecar path next to output when no explicit report path."""
    path = deepseek_summary._resolve_summary_report_path(  # noqa: SLF001
        output_path="summarized_script.json", report_path=None
    )
    assert path == "summarized_script.json.report.json"


def test_request_completion_reuses_existing_conversation_messages() -> None:
    """Keep one chat window by reusing persisted conversation messages."""

    class _FakeCompletions:
        def __init__(self) -> None:
            self.seen_messages: list[dict[str, object]] | None = None

        def create(self, **kwargs: object) -> object:
            self.seen_messages = cast(list[dict[str, object]], kwargs["messages"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=json.dumps(_sample_payload())),
                    )
                ],
                usage=SimpleNamespace(total_tokens=128),
            )

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-chat",
        max_completion_tokens=8192,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=0.2,
        auto_beta=False,
        agent_tool_loop=False,
        agent_tool_mode="off",
        agent_max_tool_rounds=1,
        agent_read_max_lines=10,
    )
    conversation_state = deepseek_summary.ConversationState(
        messages=[
            {"role": "system", "content": "persisted system"},
            {"role": "user", "content": "persisted context"},
        ]
    )

    parsed_script, _, _, _ = deepseek_summary._request_deepseek_completion(  # noqa: SLF001
        client=cast(object, fake_client),
        settings=settings,
        transcript_text="[seg_00000|Speaker 0]: ignored for existing state",
        transcript_segments=_segments(),
        transcript_manifest="segment_count=2",
        system_prompt="test prompt",
        retry_context=None,
        endpoint_mode="stable",
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0"},
        word_budget=None,
        word_budget_tolerance=0.0,
        source_word_count=20,
        conversation_state=conversation_state,
    )

    assert parsed_script.dialogue[0].text == "hello world"
    assert completions.seen_messages is not None
    assert completions.seen_messages[0]["content"] == "persisted system"
    assert completions.seen_messages[1]["content"] == "persisted context"
    assert any(
        message.get("role") == "assistant" for message in conversation_state.messages
    )


def test_request_completion_reasoning_persistence_toggle() -> None:
    """Persist reasoning_content only when explicitly enabled in settings."""

    class _FakeCompletions:
        def create(self, **kwargs: object) -> object:
            return [
                _stream_chunk(reasoning="thinking"),
                _stream_chunk(
                    finish_reason="stop",
                    content=json.dumps(_sample_payload()),
                ),
            ]

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    base_state = deepseek_summary.ConversationState(
        messages=[
            {"role": "system", "content": "persisted system"},
            {"role": "user", "content": "persisted context"},
        ]
    )
    settings_default = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-reasoner",
        max_completion_tokens=64000,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=None,
        auto_beta=False,
        agent_tool_loop=False,
        agent_tool_mode="off",
    )

    deepseek_summary._request_deepseek_completion(  # noqa: SLF001
        client=cast(object, fake_client),
        settings=settings_default,
        transcript_text="[seg_00000|Speaker 0]: ignored",
        transcript_segments=_segments(),
        transcript_manifest="segment_count=2",
        system_prompt="test prompt",
        retry_context=None,
        endpoint_mode="stable",
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0"},
        word_budget=None,
        word_budget_tolerance=0.0,
        source_word_count=20,
        conversation_state=base_state,
    )
    assert not any("reasoning_content" in message for message in base_state.messages)

    second_state = deepseek_summary.ConversationState(
        messages=[
            {"role": "system", "content": "persisted system"},
            {"role": "user", "content": "persisted context"},
        ]
    )
    settings_persist = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-reasoner",
        max_completion_tokens=64000,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=None,
        auto_beta=False,
        agent_tool_loop=False,
        agent_tool_mode="off",
        agent_persist_reasoning_content=True,
    )
    deepseek_summary._request_deepseek_completion(  # noqa: SLF001
        client=cast(object, fake_client),
        settings=settings_persist,
        transcript_text="[seg_00000|Speaker 0]: ignored",
        transcript_segments=_segments(),
        transcript_manifest="segment_count=2",
        system_prompt="test prompt",
        retry_context=None,
        endpoint_mode="stable",
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0"},
        word_budget=None,
        word_budget_tolerance=0.0,
        source_word_count=20,
        conversation_state=second_state,
    )
    assert any("reasoning_content" in message for message in second_state.messages)


def test_request_completion_rolls_over_context_when_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summarize and reset conversation when context budget reaches rollover threshold."""

    class _FakeCompletions:
        def __init__(self) -> None:
            self.seen_messages: list[dict[str, object]] | None = None

        def create(self, **kwargs: object) -> object:
            self.seen_messages = cast(list[dict[str, object]], kwargs["messages"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=json.dumps(_sample_payload())),
                    )
                ],
                usage=SimpleNamespace(total_tokens=24),
            )

    def fake_context_summary(**kwargs: object) -> str:
        del kwargs
        return "condensed context"

    monkeypatch.setattr(
        deepseek_summary,
        "_summarize_conversation_context_via_deepseek",
        fake_context_summary,
    )

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-chat",
        max_completion_tokens=100,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=0.2,
        auto_beta=False,
        agent_tool_loop=False,
        agent_tool_mode="off",
        agent_max_tool_rounds=1,
        agent_read_max_lines=10,
    )
    conversation_state = deepseek_summary.ConversationState(
        messages=[
            {"role": "system", "content": "persisted system"},
            {"role": "user", "content": "large historic context"},
        ],
        rollover_count=0,
        context_tokens_used=75,
        context_tokens_limit=100,
    )
    stream_events: list[dict[str, object]] = []

    parsed_script, _, _, _ = deepseek_summary._request_deepseek_completion(  # noqa: SLF001
        client=cast(object, fake_client),
        settings=settings,
        transcript_text="[seg_00000|Speaker 0]: ignored",
        transcript_segments=_segments(),
        transcript_manifest="segment_count=2",
        system_prompt="test prompt",
        retry_context=None,
        endpoint_mode="stable",
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0"},
        word_budget=None,
        word_budget_tolerance=0.0,
        source_word_count=20,
        stream_event_callback=stream_events.append,
        conversation_state=conversation_state,
    )

    assert parsed_script.dialogue[0].text == "hello world"
    assert completions.seen_messages is not None
    assert str(completions.seen_messages[1]["content"]).startswith(
        "Summarized context from previous conversation window:"
    )
    assert conversation_state.rollover_count == 1
    assert any(
        event.get("event") == "context_usage" and event.get("rollover_triggered") is True
        for event in stream_events
    )
