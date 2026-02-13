"""Resilience tests for DeepSeek summarizer JSON handling and endpoint fallback."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from openai import APIStatusError

from audio2script_and_summarizer import summarizer_deepseek as deepseek_summary


def _sample_payload() -> dict[str, list[dict[str, object]]]:
    """Build a minimal valid podcast payload for tests."""
    return {
        "dialogue": [
            {
                "speaker": "Speaker 0",
                "text": "Hello everyone.",
                "emo_text": "Warm",
                "emo_alpha": 0.6,
                "source_segment_ids": ["seg_00000"],
            }
        ]
    }


def _sample_script() -> deepseek_summary.PodcastScript:
    """Build a minimal validated PodcastScript payload for tests."""
    return deepseek_summary.PodcastScript(
        dialogue=[
            deepseek_summary.DialogueLine(
                speaker="Speaker 0",
                text="Hello everyone.",
                emo_text="Warm",
                emo_alpha=0.6,
                source_segment_ids=["seg_00000"],
            )
        ]
    )


def _segments() -> list[deepseek_summary.TranscriptSegment]:
    """Build minimal transcript segments for read-tool capable flows."""
    return [
        deepseek_summary.TranscriptSegment(
            segment_id="seg_00000",
            speaker="Speaker 0",
            text="Hello everyone.",
            start_time=0.0,
            end_time=1.0,
        )
    ]


def _api_status_error(status_code: int, message: str) -> APIStatusError:
    """Create an APIStatusError instance with a deterministic request/response."""
    request = httpx.Request("POST", "https://api.deepseek.com/beta/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": message})
    return APIStatusError(message=message, response=response, body={"error": message})


def test_decode_podcast_script_with_fallback_strips_markdown_fence() -> None:
    """Recover valid JSON object from fenced content."""
    fenced_content = f"```json\n{json.dumps(_sample_payload())}\n```"

    parsed_script, used_repair = deepseek_summary._decode_podcast_script_with_fallback(  # noqa: SLF001
        fenced_content
    )

    assert parsed_script.dialogue[0].speaker == "Speaker 0"
    assert used_repair is True


def test_decode_podcast_script_with_fallback_keeps_direct_json() -> None:
    """Parse direct JSON without invoking fallback repair."""
    direct_content = json.dumps(_sample_payload())

    parsed_script, used_repair = deepseek_summary._decode_podcast_script_with_fallback(  # noqa: SLF001
        direct_content
    )

    assert parsed_script.dialogue[0].source_segment_ids == ["seg_00000"]
    assert used_repair is False


def test_looks_like_beta_endpoint_error_uses_status_code() -> None:
    """Treat 404 status errors as beta endpoint incompatibility."""
    status_error = _api_status_error(status_code=404, message="Not found")
    assert deepseek_summary._looks_like_beta_endpoint_error(status_error) is True  # noqa: SLF001


def test_generate_summary_deepseek_falls_back_to_stable_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback to stable endpoint after beta incompatibility."""
    endpoint_calls: list[tuple[str, int]] = []

    def fake_build_client(**kwargs: object) -> object:
        return kwargs["base_url"]

    def fake_request(
        **kwargs: object,
    ) -> tuple[deepseek_summary.PodcastScript, bool, int, dict[str, int]]:
        endpoint_mode = cast(deepseek_summary.EndpointMode, kwargs["endpoint_mode"])
        settings = cast(deepseek_summary.DeepSeekRequestSettings, kwargs["settings"])
        endpoint_calls.append((endpoint_mode, settings.max_completion_tokens))
        if endpoint_mode == "beta":
            raise _api_status_error(
                status_code=404, message="beta endpoint unavailable"
            )
        return _sample_script(), False, 0, {}

    monkeypatch.setattr(deepseek_summary, "_build_deepseek_client", fake_build_client)
    monkeypatch.setattr(deepseek_summary, "_request_deepseek_completion", fake_request)

    result = deepseek_summary.generate_summary_deepseek(
        transcript_text="[seg_00000|Speaker 0]: Hello everyone.",
        transcript_segments=_segments(),
        api_key="test-key",
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0"},
        source_word_count=2,
        settings=deepseek_summary.DeepSeekRequestSettings(
            model="deepseek-chat",
            max_completion_tokens=8192,
            request_timeout_seconds=30.0,
            http_retries=1,
            temperature=0.2,
            auto_beta=True,
        ),
        segment_count=1,
        retry_context=None,
        word_budget=None,
        target_minutes=None,
        avg_wpm=None,
        word_budget_tolerance=0.05,
    )

    assert result.endpoint_mode == "stable"
    assert endpoint_calls[0] == ("beta", 8192)
    assert endpoint_calls[1] == ("stable", 8192)


def test_generate_summary_deepseek_wraps_endpoint_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise endpoint-attributed wrapper when generation fails."""

    def fake_build_client(**kwargs: object) -> object:
        return kwargs["base_url"]

    def fake_request(
        **kwargs: object,
    ) -> tuple[deepseek_summary.PodcastScript, bool, int, dict[str, int]]:
        raise ValueError("bad completion")

    monkeypatch.setattr(deepseek_summary, "_build_deepseek_client", fake_build_client)
    monkeypatch.setattr(deepseek_summary, "_request_deepseek_completion", fake_request)

    with pytest.raises(deepseek_summary.GenerationAttemptError) as wrapped_error:
        deepseek_summary.generate_summary_deepseek(
            transcript_text="[seg_00000|Speaker 0]: Hello everyone.",
            transcript_segments=_segments(),
            api_key="test-key",
            allowed_speakers={"Speaker 0"},
            segment_speaker_map={"seg_00000": "Speaker 0"},
            source_word_count=2,
            settings=deepseek_summary.DeepSeekRequestSettings(
                model="deepseek-chat",
                max_completion_tokens=4096,
                request_timeout_seconds=30.0,
                http_retries=1,
                temperature=0.2,
                auto_beta=False,
            ),
            segment_count=1,
            retry_context=None,
            word_budget=None,
            target_minutes=None,
            avg_wpm=None,
            word_budget_tolerance=0.05,
        )

    assert wrapped_error.value.endpoint_mode == "stable"
    assert isinstance(wrapped_error.value.cause, ValueError)


def test_build_retry_instruction_includes_previous_attempt_context() -> None:
    """Include endpoint and error context in corrective retry instruction."""
    continuation = deepseek_summary.RetryContinuationState(
        read_ranges=[(0, 20), (21, 40)],
        max_read_index=40,
        write_tool_succeeded=False,
        latest_constraints_status="fail",
        last_validation_issues=["No staged JSON available."],
        staged_output_present=False,
        staged_output_valid_json=False,
    )
    context = deepseek_summary.RetryContext(
        attempt_index=2,
        endpoint_mode="stable",
        error_type="truncated_json",
        error_digest="Unterminated string near char 1024",
        continuation=continuation,
    )

    instruction = deepseek_summary._build_retry_instruction(context)  # noqa: SLF001

    assert "Previous attempt: 2" in instruction
    assert "Previous endpoint: stable" in instruction
    assert "Failure type: truncated_json" in instruction
    assert "Resume from prior progress" in instruction
    assert "max_read_index=40" in instruction


def test_default_model_constant_uses_reasoner() -> None:
    """Default model should be reasoner for larger output headroom."""
    assert deepseek_summary.DEEPSEEK_MODEL == "deepseek-reasoner"
    assert deepseek_summary.DEFAULT_MAX_COMPLETION_TOKENS == 64000


def test_should_persist_stream_event_filters_high_volume_tokens() -> None:
    """Persist non-token and status-token events while dropping token spam."""
    assert deepseek_summary._should_persist_stream_event(  # noqa: SLF001
        {"event": "start"}
    )
    assert deepseek_summary._should_persist_stream_event(  # noqa: SLF001
        {"event": "token", "phase": "status", "text": "ready"}
    )
    assert not deepseek_summary._should_persist_stream_event(  # noqa: SLF001
        {"event": "token", "phase": "reasoning", "text": "x"}
    )
    assert not deepseek_summary._should_persist_stream_event(  # noqa: SLF001
        {"event": "token", "phase": "tool_result", "text": "y"}
    )


def test_request_completion_omits_temperature_for_reasoner() -> None:
    """Reasoner requests should omit temperature to match API constraints."""

    class _FakeCompletions:
        def __init__(self) -> None:
            self.last_kwargs: dict[str, object] = {}

        def create(self, **kwargs: object) -> object:
            self.last_kwargs = kwargs
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(reasoning_content="Thinking..."),
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            delta=SimpleNamespace(content=json.dumps(_sample_payload())),
                        )
                    ]
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
        auto_beta=True,
    )

    parsed_script, used_repair, tool_rounds, tool_call_counts = deepseek_summary._request_deepseek_completion(  # noqa: SLF001
        client=cast(object, fake_client),
        settings=settings,
        transcript_text="[seg_00000|Speaker 0]: Hello everyone.",
        transcript_segments=_segments(),
        transcript_manifest="segment_count=1",
        system_prompt="test prompt",
        retry_context=None,
        endpoint_mode="stable",
        allowed_speakers={"Speaker 0"},
        segment_speaker_map={"seg_00000": "Speaker 0"},
        word_budget=None,
        word_budget_tolerance=0.05,
        source_word_count=2,
    )

    assert parsed_script.dialogue[0].speaker == "Speaker 0"
    assert used_repair is False
    assert tool_rounds == 0
    assert tool_call_counts == {}
    assert "temperature" not in completions.last_kwargs
    assert completions.last_kwargs["stream"] is True


def test_deepseek_chat_log_writer_creates_timestamped_run_dir(
    tmp_path: Path,
) -> None:
    """Create a run folder with timestamp naming and expected log files."""
    writer = deepseek_summary.DeepSeekChatLogWriter.create(tmp_path)
    try:
        assert writer.run_directory.parent == tmp_path.resolve()
        assert re.fullmatch(r"\d{8}T\d{6}Z(?:_\d+)?", writer.run_directory.name)
        assert writer.log_path.exists()
        assert (writer.run_directory / deepseek_summary.DEEPSEEK_CHAT_LOG_META_FILE).exists()
    finally:
        writer.close()


def test_request_completion_writes_per_call_trace_logs(
    tmp_path: Path,
) -> None:
    """Write streamed flush and final message events to one run log file."""

    class _FakeCompletions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(reasoning_content="Thinking..."),
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            delta=SimpleNamespace(content=json.dumps(_sample_payload())),
                        )
                    ],
                    usage=SimpleNamespace(total_tokens=64),
                ),
            ]

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    settings = deepseek_summary.DeepSeekRequestSettings(
        model="deepseek-reasoner",
        max_completion_tokens=64000,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=None,
        auto_beta=False,
    )
    writer = deepseek_summary.DeepSeekChatLogWriter.create(tmp_path)
    try:
        parsed_script, _, _, _ = deepseek_summary._request_deepseek_completion(  # noqa: SLF001
            client=cast(object, fake_client),
            settings=settings,
            transcript_text="[seg_00000|Speaker 0]: Hello everyone.",
            transcript_segments=_segments(),
            transcript_manifest="segment_count=1",
            system_prompt="test prompt",
            retry_context=None,
            endpoint_mode="stable",
            allowed_speakers={"Speaker 0"},
            segment_speaker_map={"seg_00000": "Speaker 0"},
            word_budget=None,
            word_budget_tolerance=0.05,
            source_word_count=2,
            chat_log_writer=writer,
        )
    finally:
        writer.close()

    assert parsed_script.dialogue[0].speaker == "Speaker 0"
    all_records = [
        json.loads(line)
        for line in writer.log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    call_records = [
        record
        for record in all_records
        if record.get("call_id") == "call_0001"
    ]
    call_records = [
        record
        for record in call_records
        if record.get("event") != "run_done"
    ]
    assert any(record.get("event") == "call_start" for record in call_records)
    assert any(
        record.get("event") == "stream_flush" and record.get("phase") == "reasoning"
        for record in call_records
    )
    assert any(
        record.get("event") == "stream_flush" and record.get("phase") == "answer"
        for record in call_records
    )
    assert not any(record.get("event") == "token" for record in call_records)
    assert any(
        record.get("event") == "message" and record.get("phase") == "answer"
        for record in call_records
    )
    assert any(record.get("event") == "call_done" for record in call_records)
