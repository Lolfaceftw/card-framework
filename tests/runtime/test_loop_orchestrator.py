import asyncio
from pathlib import Path
from typing import Awaitable, Callable

import pytest

from card_framework.runtime.loop_orchestrator import Orchestrator


class _FakeAgentClient:
    """Injectable fake client exposing the production ``send_task`` API."""

    def __init__(
        self,
        send_task: Callable[..., Awaitable[str]],
    ) -> None:
        self.send_task = send_task


def test_run_summarizer_once_returns_agent_output() -> None:
    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del task_data, timeout, max_retries
        assert port == 9010
        assert metadata is not None
        assert metadata.get("stage") == "summarizer_single_pass"
        return "<summary>ok</summary>"

    orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        agent_client=_FakeAgentClient(_fake_send_task),
    )

    result = asyncio.run(
        orchestrator.run_summarizer_once(
            target_seconds=300,
            duration_tolerance_ratio=0.05,
            full_transcript_text="",
        )
    )
    assert result == "<summary>ok</summary>"


def test_run_critic_once_parses_valid_json() -> None:
    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del task_data, timeout, max_retries
        assert port == 9011
        assert metadata is not None
        assert metadata.get("stage") == "critic_single_pass"
        return '{"status":"pass","word_count":72,"estimated_seconds":300.0,"feedback":"ok"}'

    orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        agent_client=_FakeAgentClient(_fake_send_task),
    )

    verdict = asyncio.run(
        orchestrator.run_critic_once(
            draft="<summary>ok</summary>",
            target_seconds=300,
            duration_tolerance_ratio=0.05,
            full_transcript_text="",
        )
    )
    assert verdict.status == "pass"
    assert verdict.word_count == 72


def test_run_critic_once_normalizes_status() -> None:
    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del task_data, timeout, max_retries
        assert port == 9011
        assert metadata is not None
        assert metadata.get("stage") == "critic_single_pass"
        return '{"status":" PASS ","word_count":72,"estimated_seconds":300.0,"feedback":"ok"}'

    orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        agent_client=_FakeAgentClient(_fake_send_task),
    )

    verdict = asyncio.run(
        orchestrator.run_critic_once(
            draft="<summary>ok</summary>",
            target_seconds=300,
            duration_tolerance_ratio=0.05,
            full_transcript_text="",
        )
    )
    assert verdict.status == "pass"
    assert verdict.word_count == 72


def test_run_critic_once_raises_for_invalid_json() -> None:
    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del port, task_data, timeout, max_retries, metadata
        return "not-json"

    orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        agent_client=_FakeAgentClient(_fake_send_task),
    )

    with pytest.raises(RuntimeError):
        asyncio.run(
            orchestrator.run_critic_once(
                draft="draft",
                target_seconds=120,
                duration_tolerance_ratio=0.05,
                full_transcript_text="",
            )
        )


def test_run_loop_sends_empty_loop_context_on_first_pass() -> None:
    summarizer_loop_contexts: list[str] = []

    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del timeout, max_retries, metadata
        if port == 9010:
            summarizer_loop_contexts.append(str(getattr(task_data, "loop_context", "")))
            return "<summary>draft</summary>"
        if port == 9011:
            return '{"status":"pass","word_count":72,"estimated_seconds":300.0,"feedback":"ok"}'
        raise AssertionError(f"Unexpected port: {port}")

    orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        agent_client=_FakeAgentClient(_fake_send_task),
    )

    result = asyncio.run(
        orchestrator.run_loop(
            target_seconds=300,
            duration_tolerance_ratio=0.05,
            max_iterations=2,
            full_transcript_text="",
        )
    )
    assert result == "<summary>draft</summary>"
    assert len(summarizer_loop_contexts) == 1
    assert summarizer_loop_contexts[0].strip() == ""


def test_run_loop_sends_non_empty_loop_context_on_retry_after_fail() -> None:
    summarizer_loop_contexts: list[str] = []
    critic_calls = 0

    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        nonlocal critic_calls
        del timeout, max_retries, metadata
        if port == 9010:
            summarizer_loop_contexts.append(str(getattr(task_data, "loop_context", "")))
            return "<summary>draft</summary>"
        if port == 9011:
            critic_calls += 1
            if critic_calls == 1:
                return (
                    '{"status":"fail","word_count":58,'
                    '"estimated_seconds":220.0,'
                    '"feedback":"[] Fix chronology\\n[] Fix chronology\\n[] Expand coverage"}'
                )
            return '{"status":"pass","word_count":74,"estimated_seconds":300.0,"feedback":"ok"}'
        raise AssertionError(f"Unexpected port: {port}")

    orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        agent_client=_FakeAgentClient(_fake_send_task),
    )

    result = asyncio.run(
        orchestrator.run_loop(
            target_seconds=300,
            duration_tolerance_ratio=0.05,
            max_iterations=3,
            full_transcript_text="",
        )
    )
    assert result == "<summary>draft</summary>"
    assert len(summarizer_loop_contexts) == 2
    assert summarizer_loop_contexts[0].strip() == ""
    assert summarizer_loop_contexts[1].strip() != ""


def test_run_loop_restores_persisted_repeated_remedy_history_across_runs(
    tmp_path: Path,
) -> None:
    """Reload persisted remedy history so the next retry warns against repeats."""
    artifact_path = tmp_path / "loop_memory.json"
    loop_memory_context = {
        "transcript_sha256": "abc123",
        "target_seconds": "300",
        "duration_tolerance_ratio": "0.050000",
    }

    async def _first_send_task(
        port,
        task_data,
        timeout=120.0,
        max_retries=3,
        metadata=None,
    ):
        del task_data, timeout, max_retries, metadata
        if port == 9010:
            return "<summary>draft</summary>"
        if port == 9011:
            return (
                '{"status":"fail","word_count":58,'
                '"estimated_seconds":220.0,'
                '"feedback":"[] Expand missing middle section coverage."}'
            )
        raise AssertionError(f"Unexpected port: {port}")

    first_orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        agent_client=_FakeAgentClient(_first_send_task),
    )

    first_result = asyncio.run(
        first_orchestrator.run_loop(
            target_seconds=300,
            duration_tolerance_ratio=0.05,
            max_iterations=1,
            full_transcript_text="",
            loop_memory_artifact_path=artifact_path,
            loop_memory_context=loop_memory_context,
        )
    )
    assert first_result is None
    assert artifact_path.exists()

    summarizer_loop_contexts: list[str] = []
    critic_calls = 0

    async def _second_send_task(
        port,
        task_data,
        timeout=120.0,
        max_retries=3,
        metadata=None,
    ):
        nonlocal critic_calls
        del timeout, max_retries, metadata
        if port == 9010:
            summarizer_loop_contexts.append(str(getattr(task_data, "loop_context", "")))
            return "<summary>draft</summary>"
        if port == 9011:
            critic_calls += 1
            if critic_calls == 1:
                return (
                    '{"status":"fail","word_count":58,'
                    '"estimated_seconds":220.0,'
                    '"feedback":"[] Include missing middle section coverage."}'
                )
            return (
                '{"status":"pass","word_count":74,'
                '"estimated_seconds":300.0,'
                '"feedback":"ok"}'
            )
        raise AssertionError(f"Unexpected port: {port}")

    second_orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        agent_client=_FakeAgentClient(_second_send_task),
    )

    second_result = asyncio.run(
        second_orchestrator.run_loop(
            target_seconds=300,
            duration_tolerance_ratio=0.05,
            max_iterations=2,
            full_transcript_text="",
            loop_memory_artifact_path=artifact_path,
            loop_memory_context=loop_memory_context,
        )
    )

    assert second_result == "<summary>draft</summary>"
    assert len(summarizer_loop_contexts) == 2
    assert summarizer_loop_contexts[0].strip() == ""
    assert "repeated remedy alert" in summarizer_loop_contexts[1].lower()


def test_run_summarizer_once_uses_full_transcript_timeout_floor() -> None:
    observed_timeout: dict[str, float] = {}

    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del task_data, max_retries, metadata
        assert port == 9010
        observed_timeout["value"] = float(timeout)
        return "<summary>ok</summary>"

    orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        timeouts={"summarizer": 180},
        agent_client=_FakeAgentClient(_fake_send_task),
    )

    asyncio.run(
        orchestrator.run_summarizer_once(
            target_seconds=300,
            duration_tolerance_ratio=0.05,
            full_transcript_text="full transcript present",
        )
    )
    assert observed_timeout["value"] == 1800.0


def test_run_loop_uses_live_draft_summarizer_timeout_floor() -> None:
    observed_timeout: dict[str, float] = {}

    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del task_data, max_retries, metadata
        if port == 9010:
            observed_timeout["value"] = float(timeout)
            return "<summary>ok</summary>"
        if port == 9011:
            return '{"status":"pass","word_count":72,"estimated_seconds":300.0,"feedback":"ok"}'
        raise AssertionError(f"Unexpected port: {port}")

    orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        timeouts={"summarizer": 180},
        agent_client=_FakeAgentClient(_fake_send_task),
    )

    result = asyncio.run(
        orchestrator.run_loop(
            target_seconds=300,
            duration_tolerance_ratio=0.05,
            max_iterations=1,
            full_transcript_text="full transcript present",
            speaker_samples_manifest_path=Path("speaker_samples_manifest.json"),
            draft_audio_state_path=Path("live_draft.state.json"),
        )
    )

    assert result == "<summary>ok</summary>"
    assert observed_timeout["value"] == 1800.0


def test_run_critic_once_uses_full_transcript_timeout_floor() -> None:
    observed_timeout: dict[str, float] = {}

    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del task_data, max_retries, metadata
        assert port == 9011
        observed_timeout["value"] = float(timeout)
        return '{"status":"pass","word_count":72,"estimated_seconds":300.0,"feedback":"ok"}'

    orchestrator = Orchestrator(
        retrieval_port=9012,
        summarizer_port=9010,
        critic_port=9011,
        timeouts={"critic": 120},
        agent_client=_FakeAgentClient(_fake_send_task),
    )

    verdict = asyncio.run(
        orchestrator.run_critic_once(
            draft="<summary>ok</summary>",
            target_seconds=300,
            duration_tolerance_ratio=0.05,
            full_transcript_text="full transcript present",
        )
    )
    assert verdict.status == "pass"
    assert observed_timeout["value"] == 300.0

