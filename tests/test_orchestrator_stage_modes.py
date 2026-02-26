import asyncio

import pytest

from orchestrator import Orchestrator


def test_run_summarizer_once_returns_agent_output(monkeypatch) -> None:
    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del task_data, timeout, max_retries
        assert port == 9010
        assert metadata is not None
        assert metadata.get("stage") == "summarizer_single_pass"
        return "<summary>ok</summary>"

    monkeypatch.setattr("orchestrator.agent_client.send_task", _fake_send_task)
    orchestrator = Orchestrator(retrieval_port=9012, summarizer_port=9010, critic_port=9011)

    result = asyncio.run(
        orchestrator.run_summarizer_once(
            min_words=50,
            max_words=80,
            full_transcript_text="",
        )
    )
    assert result == "<summary>ok</summary>"


def test_run_critic_once_parses_valid_json(monkeypatch) -> None:
    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del task_data, timeout, max_retries
        assert port == 9011
        assert metadata is not None
        assert metadata.get("stage") == "critic_single_pass"
        return '{"status":"pass","word_count":72,"feedback":"ok"}'

    monkeypatch.setattr("orchestrator.agent_client.send_task", _fake_send_task)
    orchestrator = Orchestrator(retrieval_port=9012, summarizer_port=9010, critic_port=9011)

    verdict = asyncio.run(
        orchestrator.run_critic_once(
            draft="<summary>ok</summary>",
            min_words=70,
            max_words=78,
            full_transcript_text="",
        )
    )
    assert verdict.status == "pass"
    assert verdict.word_count == 72


def test_run_critic_once_normalizes_status(monkeypatch) -> None:
    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del task_data, timeout, max_retries
        assert port == 9011
        assert metadata is not None
        assert metadata.get("stage") == "critic_single_pass"
        return '{"status":" PASS ","word_count":72,"feedback":"ok"}'

    monkeypatch.setattr("orchestrator.agent_client.send_task", _fake_send_task)
    orchestrator = Orchestrator(retrieval_port=9012, summarizer_port=9010, critic_port=9011)

    verdict = asyncio.run(
        orchestrator.run_critic_once(
            draft="<summary>ok</summary>",
            min_words=70,
            max_words=78,
            full_transcript_text="",
        )
    )
    assert verdict.status == "pass"
    assert verdict.word_count == 72


def test_run_critic_once_raises_for_invalid_json(monkeypatch) -> None:
    async def _fake_send_task(port, task_data, timeout=120.0, max_retries=3, metadata=None):
        del port, task_data, timeout, max_retries, metadata
        return "not-json"

    monkeypatch.setattr("orchestrator.agent_client.send_task", _fake_send_task)
    orchestrator = Orchestrator(retrieval_port=9012, summarizer_port=9010, critic_port=9011)

    with pytest.raises(RuntimeError):
        asyncio.run(
            orchestrator.run_critic_once(
                draft="draft",
                min_words=10,
                max_words=20,
                full_transcript_text="",
            )
        )
