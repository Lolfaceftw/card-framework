"""Runner helper coverage for the SPoRC benchmark."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from card_framework.benchmark.summarizer_critic.sporc.runner import (
    _ArtifactCaptureLLMProvider,
    _ArtifactCaptureResponseCallback,
    _canonical_speaker_label,
    _claim_next_shared_sample_id,
    _build_preflight_failed_result,
    _classify_sample_failure,
    _copy_sample_transcript,
    _prepare_provider_config,
    _compute_composite_score,
    _execute_sample,
    _evaluate_sample_preflight,
    _initialize_shared_sample_queue,
    _initialize_shared_sample_queue_from_manifest,
    _load_shared_sample,
    _normalize_max_iterations,
    _mark_shared_sample_completed,
    _open_shared_sample_queue_connection,
    _prepare_benchmark_audio_cfg,
    _resolve_override_target_seconds,
    _resolve_sample_target_seconds,
    _resolve_sample_shard_args,
    _resolve_orchestrator_timeouts,
    _SampleArtifactRecorder,
    _select_samples,
    _select_sample_shard,
    _summarize_results,
    _write_checkpoint_payload,
    build_execute_parser,
    execute_benchmark,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    PreparedBenchmarkSample,
    SampleBenchmarkResult,
    SampleMetricBundle,
)
from card_framework.benchmark.summarizer_critic.sporc.judge import SPoRCJudgeResult
from card_framework.benchmark.types import EmbeddingProfile, ProviderProfile
from card_framework.shared.llm_provider import LLMProvider, Message


def _sample(
    sample_id: str,
    *,
    transcript_path: str | None = None,
) -> PreparedBenchmarkSample:
    return PreparedBenchmarkSample(
        sample_id=sample_id,
        dataset="sporc_sample",
        episode_key=sample_id,
        transcript_path=transcript_path or f"{sample_id}.json",
        reference_summary="reference",
        target_seconds=30,
    )


def _write_transcript(
    path: Path,
    *,
    text: str,
    start_time: int = 0,
    end_time: int = 60_000,
) -> None:
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "text": text,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                ],
                "metadata": {"dataset": "sporc"},
            }
        ),
        encoding="utf-8",
    )


def _write_prepared_manifest_with_sidecar(
    manifest_path: Path,
    *,
    samples: list[PreparedBenchmarkSample],
) -> Path:
    samples_path = manifest_path.parent / "samples.jsonl.gz"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(samples_path, "wt", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample)))
            handle.write("\n")
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(manifest_path.parent / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(manifest_path.parent / "speakerTurnData.jsonl.gz"),
                "sample_count": len(samples),
                "samples_path": str(samples_path),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_empty_transcript(path: Path) -> None:
    path.write_text(
        json.dumps({"segments": [], "metadata": {"dataset": "sporc"}}),
        encoding="utf-8",
    )


def _write_transcript_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_result(sample_id: str) -> SampleBenchmarkResult:
    return SampleBenchmarkResult(
        sample_id=sample_id,
        dataset="sporc_sample",
        episode_key=sample_id,
        status="completed",
        target_seconds=30,
        candidate_seconds=29.5,
        converged=True,
        iterations_run=2,
        final_status="pass",
        duration_seconds=3.0,
        retrieval_events=1,
        tool_invocations=4,
        metrics=SampleMetricBundle(
            factualness=0.8,
            naturalness=0.7,
            speaker_grammar_similarity=0.6,
            composite=0.7,
        ),
    )


def _failed_result(sample_id: str) -> SampleBenchmarkResult:
    return SampleBenchmarkResult(
        sample_id=sample_id,
        dataset="sporc_sample",
        episode_key=sample_id,
        status="failed",
        target_seconds=30,
        candidate_seconds=0.0,
        converged=False,
        iterations_run=0,
        final_status="execution_error",
        duration_seconds=1.0,
        retrieval_events=0,
        tool_invocations=0,
        metrics=SampleMetricBundle(0.0, 0.0, 0.0, 0.0),
        error_message="boom",
    )


class _CallbackAwareProvider(LLMProvider):
    """Test double that emits deterministic streamed reasoning and content."""

    def __init__(
        self,
        *,
        agent_name: str,
        response_text: str = "",
        response_message: Message | None = None,
        reasoning_tokens: list[str] | None = None,
        content_tokens: list[str] | None = None,
    ) -> None:
        self._agent_name = agent_name
        self._response_text = response_text
        self._response_message = response_message
        self._reasoning_tokens = reasoning_tokens or []
        self._content_tokens = content_tokens or []
        self._callback = None

    def set_response_callback(self, callback) -> None:  # type: ignore[no-untyped-def]
        self._callback = callback

    def _emit_stream(self) -> None:
        if self._callback is None:
            return
        self._callback.on_start(self._agent_name)
        for token in self._reasoning_tokens:
            self._callback.on_thought_token(token)
        for token in self._content_tokens:
            self._callback.on_content_token(token)
        self._callback.on_complete()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        del system_prompt, user_prompt, max_tokens
        self._emit_stream()
        return self._response_text

    def chat(
        self,
        messages,
        tools=None,
        tool_choice=None,
        max_tokens: int | None = None,
    ) -> Message:
        del messages, tools, tool_choice, max_tokens
        self._emit_stream()
        if self._response_message is None:
            raise AssertionError("response_message must be set for chat()")
        return self._response_message


class _StubEventCapture:
    """Minimal event-capture stub for `_execute_sample` coverage."""

    def __init__(self) -> None:
        self._counts = {"retrieval_stats": 2, "tool_invocation": 4}

    def count(self, name: str) -> int:
        return int(self._counts.get(name, 0))


class _StubOrchestrator:
    """Minimal orchestrator stub for `_execute_sample` coverage."""

    def __init__(self) -> None:
        self.index_calls = 0
        self.clear_calls = 0
        self.indexed_sample_ids: list[str | None] = []

    async def index_transcript(
        self,
        transcript_payload: dict[str, object],
        *,
        sample_id: str | None = None,
    ) -> None:
        del transcript_payload
        self.index_calls += 1
        self.indexed_sample_ids.append(sample_id)

    async def clear_transcript_index(self) -> None:
        self.clear_calls += 1

    async def run_loop_with_diagnostics(
        self,
        *,
        target_seconds: int,
        duration_tolerance_ratio: float,
        max_iterations: int | None,
        full_transcript_text: str,
        speaker_samples_manifest_path: Path | None,
        draft_audio_state_path: Path | None,
    ) -> dict[str, object]:
        del (
            target_seconds,
            duration_tolerance_ratio,
            max_iterations,
            full_transcript_text,
            speaker_samples_manifest_path,
            draft_audio_state_path,
        )
        return {
            "draft": "plain summary output",
            "final_estimated_seconds": 27.5,
            "converged": True,
            "iterations_run": 2,
            "final_status": "pass",
            "final_feedback": "critic accepted the draft",
        }


class _FailingStubOrchestrator(_StubOrchestrator):
    """Orchestrator stub that raises after indexing to exercise cleanup paths."""

    async def run_loop_with_diagnostics(
        self,
        *,
        target_seconds: int,
        duration_tolerance_ratio: float,
        max_iterations: int | None,
        full_transcript_text: str,
        speaker_samples_manifest_path: Path | None,
        draft_audio_state_path: Path | None,
    ) -> dict[str, object]:
        del (
            target_seconds,
            duration_tolerance_ratio,
            max_iterations,
            full_transcript_text,
            speaker_samples_manifest_path,
            draft_audio_state_path,
        )
        raise RuntimeError("retrieval cleanup retry")


def test_sample_artifact_recorder_writes_hierarchical_llm_traces(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    sample = _sample("sample-a")
    recorder = _SampleArtifactRecorder(run_root=run_root)
    recorder.activate_sample(sample=sample, target_seconds=60)

    summarizer_inner = _CallbackAwareProvider(
        agent_name="Summarizer",
        response_message=Message(
            role="assistant",
            content="plain summary output",
            reasoning_content="summary reasoning",
        ),
        reasoning_tokens=["summary ", "reasoning"],
        content_tokens=["plain ", "summary ", "output"],
    )
    summarizer_inner.set_response_callback(
        _ArtifactCaptureResponseCallback(recorder=recorder, role="summarizer")
    )
    summarizer = _ArtifactCaptureLLMProvider(
        inner_provider=summarizer_inner,
        recorder=recorder,
        role="summarizer",
    )
    summarizer.chat(
        messages=[
            {"role": "system", "content": "You are the summarizer."},
            {"role": "user", "content": "Summarize this transcript."},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "save_draft",
                    "description": "Save the current draft.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        max_tokens=256,
    )

    critic_inner = _CallbackAwareProvider(
        agent_name="Critic",
        response_message=Message(
            role="assistant",
            content='{"verdict":"pass"}',
            reasoning_content="critic reasoning",
        ),
        reasoning_tokens=["critic ", "reasoning"],
        content_tokens=['{"verdict":"pass"}'],
    )
    critic_inner.set_response_callback(
        _ArtifactCaptureResponseCallback(recorder=recorder, role="critic")
    )
    critic = _ArtifactCaptureLLMProvider(
        inner_provider=critic_inner,
        recorder=recorder,
        role="critic",
    )
    critic.chat(
        messages=[
            {"role": "system", "content": "You are the critic."},
            {"role": "user", "content": "Evaluate the draft."},
        ],
        max_tokens=128,
    )

    judge_inner = _CallbackAwareProvider(
        agent_name="Judge",
        response_text=(
            '{"Factualness": 0.8, "Naturalness": 0.7, '
            '"Speaker Grammar Similarity": 0.6}'
        ),
        reasoning_tokens=["judge ", "reasoning"],
        content_tokens=['{"Factualness":0.8}'],
    )
    judge_inner.set_response_callback(
        _ArtifactCaptureResponseCallback(recorder=recorder, role="judge")
    )
    judge = _ArtifactCaptureLLMProvider(
        inner_provider=judge_inner,
        recorder=recorder,
        role="judge",
    )
    judge.generate(
        system_prompt="You are the judge.",
        user_prompt="Score the summary.",
        max_tokens=64,
    )

    result = _completed_result(sample.sample_id)
    artifact_paths = recorder.finalize_sample(
        result=result,
        diagnostics={
            "draft": "plain summary output",
            "final_feedback": "critic accepted the draft",
        },
        candidate_summary="plain summary output",
    )
    recorder.clear_active_sample()

    assert artifact_paths is not None
    sample_root = run_root / "samples" / sample.sample_id
    assert (sample_root / "final" / "summary.xml").read_text(encoding="utf-8") == (
        "plain summary output"
    )
    assert (
        sample_root / "final" / "critic_feedback.txt"
    ).read_text(encoding="utf-8") == "critic accepted the draft"

    summarizer_trace = json.loads(
        (sample_root / "llm" / "summarizer" / "call_001.json").read_text(
            encoding="utf-8"
        )
    )
    critic_trace = json.loads(
        (sample_root / "llm" / "critic" / "call_001.json").read_text(
            encoding="utf-8"
        )
    )
    judge_trace = json.loads(
        (sample_root / "llm" / "judge" / "call_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert summarizer_trace["streamed_reasoning_content"] == "summary reasoning"
    assert summarizer_trace["streamed_content"] == "plain summary output"
    assert critic_trace["streamed_reasoning_content"] == "critic reasoning"
    assert judge_trace["streamed_reasoning_content"] == "judge reasoning"
    assert judge_trace["response_text"].startswith('{"Factualness": 0.8')

    manifest_payload = json.loads(
        (sample_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert Path(manifest_payload["final"]["sample_result"]).name == "sample_result.json"
    assert (
        Path(manifest_payload["llm_traces"]["summarizer"]["calls"][0]).name
        == "call_001.json"
    )

    sample_result_payload = json.loads(
        (sample_root / "final" / "sample_result.json").read_text(encoding="utf-8")
    )
    assert sample_result_payload["summarizer_trace_dir"] == str(
        (sample_root / "llm" / "summarizer").resolve()
    )
    assert sample_result_payload["final_summary_path"] == str(
        (sample_root / "final" / "summary.xml").resolve()
    )


def test_execute_sample_persists_artifact_paths_in_result(tmp_path: Path) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(
        transcript_path,
        text="hello there from the podcast transcript",
    )
    run_root = tmp_path / "run"
    sample = _sample("sample-a", transcript_path=str(transcript_path))
    recorder = _SampleArtifactRecorder(run_root=run_root)
    judge = SimpleNamespace(
        evaluate=lambda **kwargs: SPoRCJudgeResult(
            status="ok",
            scores={
                "Factualness": 0.8,
                "Naturalness": 0.7,
                "Speaker Grammar Similarity": 0.6,
            },
        )
    )

    result = asyncio.run(
        _execute_sample(
            sample=sample,
            target_seconds=30,
            preloaded_transcript=None,
            orchestrator=_StubOrchestrator(),
            event_capture=_StubEventCapture(),
            retrieval_enabled=False,
            duration_tolerance_ratio=0.25,
            max_iterations=3,
            judge=judge,
            calibration=SimpleNamespace(),
            run_root=run_root,
            artifact_recorder=recorder,
            audio_cfg={},
            transcript_only=True,
            voice_clone_orchestrator=None,
            force_download_audio=False,
        )
    )

    assert result.status == "completed"
    assert result.sample_artifact_dir == str(
        (run_root / "samples" / sample.sample_id).resolve()
    )
    assert result.final_summary_path == str(
        (run_root / "samples" / sample.sample_id / "final" / "summary.xml").resolve()
    )
    assert result.final_critic_feedback_path == str(
        (
            run_root
            / "samples"
            / sample.sample_id
            / "final"
            / "critic_feedback.txt"
        ).resolve()
    )
    assert Path(result.sample_result_artifact_path or "").exists()
    assert Path(result.artifact_manifest_path or "").exists()
    assert Path(result.judge_trace_dir or "").exists()

    sample_result_payload = json.loads(
        Path(result.sample_result_artifact_path or "").read_text(encoding="utf-8")
    )
    assert sample_result_payload["sample_artifact_dir"] == result.sample_artifact_dir
    assert sample_result_payload["final_summary_path"] == result.final_summary_path
    assert Path(result.final_summary_path or "").read_text(encoding="utf-8") == (
        "plain summary output"
    )
    assert Path(result.final_critic_feedback_path or "").read_text(
        encoding="utf-8"
    ) == "critic accepted the draft"


def test_execute_sample_clears_retrieval_namespace_after_success(tmp_path: Path) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(transcript_path, text="apple banana cherry")
    run_root = tmp_path / "run"
    sample = _sample("sample-a", transcript_path=str(transcript_path))
    recorder = _SampleArtifactRecorder(run_root=run_root)
    judge = SimpleNamespace(
        evaluate=lambda **kwargs: SPoRCJudgeResult(
            status="ok",
            scores={
                "Factualness": 0.8,
                "Naturalness": 0.7,
                "Speaker Grammar Similarity": 0.6,
            },
        )
    )
    orchestrator = _StubOrchestrator()

    result = asyncio.run(
        _execute_sample(
            sample=sample,
            target_seconds=30,
            preloaded_transcript=None,
            orchestrator=orchestrator,
            event_capture=_StubEventCapture(),
            retrieval_enabled=True,
            duration_tolerance_ratio=0.25,
            max_iterations=3,
            judge=judge,
            calibration=SimpleNamespace(),
            run_root=run_root,
            artifact_recorder=recorder,
            audio_cfg={},
            transcript_only=True,
            voice_clone_orchestrator=None,
            force_download_audio=False,
        )
    )

    assert result.status == "completed"
    assert orchestrator.index_calls == 1
    assert orchestrator.indexed_sample_ids == ["sample-a"]
    assert orchestrator.clear_calls == 1


def test_execute_sample_clears_retrieval_namespace_after_failure(tmp_path: Path) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(transcript_path, text="apple banana cherry")
    run_root = tmp_path / "run"
    sample = _sample("sample-a", transcript_path=str(transcript_path))
    recorder = _SampleArtifactRecorder(run_root=run_root)
    orchestrator = _FailingStubOrchestrator()

    result = asyncio.run(
        _execute_sample(
            sample=sample,
            target_seconds=30,
            preloaded_transcript=None,
            orchestrator=orchestrator,
            event_capture=_StubEventCapture(),
            retrieval_enabled=True,
            duration_tolerance_ratio=0.25,
            max_iterations=3,
            judge=SimpleNamespace(),
            calibration=SimpleNamespace(),
            run_root=run_root,
            artifact_recorder=recorder,
            audio_cfg={},
            transcript_only=True,
            voice_clone_orchestrator=None,
            force_download_audio=False,
        )
    )

    assert result.status == "failed"
    assert orchestrator.index_calls == 1
    assert orchestrator.indexed_sample_ids == ["sample-a"]
    assert orchestrator.clear_calls == 1


def test_copy_sample_transcript_keeps_raw_speakers_and_marks_uncertainty(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "noisy.transcript.json"
    original_payload = {
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "text": "Welcome back to the show.",
                "start_time": 0,
                "end_time": 4000,
            },
            {
                "speaker": "SPEAKER_09",
                "text": "yeah",
                "start_time": 4000,
                "end_time": 4300,
                "overlap_speakers": ["SPEAKER_00"],
            },
            {
                "speaker": "SPEAKER_00",
                "text": "Today we are discussing the launch.",
                "start_time": 4300,
                "end_time": 9000,
            },
            {
                "speaker": "SPEAKER_42",
                "text": "...",
                "start_time": 9000,
                "end_time": 9300,
            },
            {
                "speaker": "SPEAKER_77",
                "text": "absolutely",
                "start_time": 9300,
                "end_time": 9800,
                "overlap_speakers": ["SPEAKER_01"],
            },
            {
                "speaker": "SPEAKER_01",
                "text": "Thanks for having me on the podcast.",
                "start_time": 9800,
                "end_time": 15000,
            },
        ],
        "metadata": {
            "dataset": "sporc",
            "speaker_alias_map": {
                "SPEAKER_00": "SPEAKER_00",
                "speaker_00": "SPEAKER_00",
                "Alex Host": "SPEAKER_00",
                "SPEAKER_01": "SPEAKER_01",
                "speaker_01": "SPEAKER_01",
                "Blair Guest": "SPEAKER_01",
            },
            "named_speaker_labels": ["SPEAKER_00", "SPEAKER_01"],
            "main_speaker_labels": ["SPEAKER_00", "SPEAKER_01"],
            "speaker_notes": ["mainEpSpeakers: SPEAKER_00, SPEAKER_01"],
        },
    }
    _write_transcript_payload(transcript_path, original_payload)
    sample = _sample(
        "sample-noisy",
        transcript_path=str(transcript_path),
    )

    copied_transcript, working_path = _copy_sample_transcript(
        sample=sample,
        run_root=tmp_path / "run",
        normalize_for_transcript_only=True,
    )

    copied_payload = json.loads(working_path.read_text(encoding="utf-8"))
    copied_speakers = [segment["speaker"] for segment in copied_payload["segments"]]
    assert copied_speakers == [
        "SPEAKER_00",
        "SPEAKER_09",
        "SPEAKER_00",
        "SPEAKER_77",
        "SPEAKER_01",
    ]
    assert copied_payload["segments"][1]["raw_speakers"] == ["SPEAKER_09", "SPEAKER_00"]
    assert copied_payload["segments"][1]["speaker_candidates"] == [
        "SPEAKER_09",
        "SPEAKER_00",
    ]
    assert copied_payload["segments"][1]["speaker_uncertain"] is True
    assert copied_payload["segments"][3]["raw_speakers"] == ["SPEAKER_77", "SPEAKER_01"]
    assert copied_payload["segments"][3]["speaker_candidates"] == [
        "SPEAKER_77",
        "SPEAKER_01",
    ]
    assert copied_payload["segments"][3]["speaker_uncertain"] is True
    assert copied_payload["segments"][4]["speaker_uncertain"] is False
    assert copied_payload["metadata"]["speaker_alias_map"] == {
        "SPEAKER_00": "SPEAKER_00",
        "Alex Host": "SPEAKER_00",
        "SPEAKER_01": "SPEAKER_01",
        "Blair Guest": "SPEAKER_01",
    }
    assert copied_payload["metadata"]["speaker_normalization"]["raw_segment_count"] == 6
    assert copied_payload["metadata"]["speaker_normalization"][
        "speaker_assignment_mode"
    ] == "raw_with_uncertainty"
    assert copied_payload["metadata"]["speaker_normalization"][
        "speaker_reassignment_enabled"
    ] is False
    assert (
        copied_payload["metadata"]["speaker_normalization"][
            "normalized_distinct_speaker_count"
        ]
        == 4
    )
    assert (
        copied_payload["metadata"]["speaker_normalization"]["uncertain_segment_count"]
        == 2
    )
    assert copied_transcript.metadata["speaker_normalization"]["mode"] == (
        "transcript_only_benchmark"
    )
    assert json.loads(transcript_path.read_text(encoding="utf-8")) == original_payload


def test_copy_sample_transcript_preserves_stable_main_speakers(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "stable.transcript.json"
    _write_transcript_payload(
        transcript_path,
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Happy Father's Day to everyone listening.",
                    "start_time": 0,
                    "end_time": 5000,
                },
                {
                    "speaker": "SPEAKER_02",
                    "text": "Absolutely and thanks again for joining us today.",
                    "start_time": 5000,
                    "end_time": 10000,
                },
                {
                    "speaker": "SPEAKER_00",
                    "text": "Let's get into the details now.",
                    "start_time": 10000,
                    "end_time": 14000,
                },
            ],
            "metadata": {
                "dataset": "sporc",
                "speaker_alias_map": {
                    "SPEAKER_00": "SPEAKER_00",
                    "Host": "SPEAKER_00",
                    "SPEAKER_02": "SPEAKER_02",
                    "Guest": "SPEAKER_02",
                },
                "named_speaker_labels": ["SPEAKER_00", "SPEAKER_02"],
                "main_speaker_labels": ["SPEAKER_00", "SPEAKER_02"],
            },
        },
    )
    sample = _sample(
        "sample-stable",
        transcript_path=str(transcript_path),
    )

    _, working_path = _copy_sample_transcript(
        sample=sample,
        run_root=tmp_path / "run",
        normalize_for_transcript_only=True,
    )

    copied_payload = json.loads(working_path.read_text(encoding="utf-8"))
    assert [segment["speaker"] for segment in copied_payload["segments"]] == [
        "SPEAKER_00",
        "SPEAKER_02",
        "SPEAKER_00",
    ]
    assert copied_payload["segments"][0]["speaker_candidates"] == ["SPEAKER_00"]
    assert copied_payload["segments"][0]["speaker_uncertain"] is False
    assert copied_payload["metadata"]["named_speaker_labels"] == [
        "SPEAKER_00",
        "SPEAKER_02",
    ]
    assert copied_payload["metadata"]["main_speaker_labels"] == [
        "SPEAKER_00",
        "SPEAKER_02",
    ]


def test_canonical_speaker_label_is_case_insensitive_without_duplicate_alias_keys() -> None:
    alias_map = {
        "SPEAKER_00": "SPEAKER_00",
        "Alex Host": "SPEAKER_00",
    }

    assert _canonical_speaker_label("speaker_00", alias_map) == "SPEAKER_00"
    assert _canonical_speaker_label("alex host", alias_map) == "SPEAKER_00"


def _benchmark_base_cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "orchestrator": {"duration_tolerance_ratio": 0.25},
            "audio": {
                "work_dir": str(tmp_path / "audio"),
                "speaker_samples": {"enabled": False},
                "voice_clone": {
                    "enabled": False,
                    "emo_presets": {"neutral": "steady neutral delivery"},
                    "live_drafting": {"enabled": False},
                },
            },
            "logging": {"enabled": False},
            "agents": {
                "summarizer": {"max_tool_turns": 1, "loop_guardrails": {}},
                "critic": {"max_tool_turns": 1, "reasoning_loop_guard": {}},
            },
        }
    )


def _benchmark_provider_profile() -> ProviderProfile:
    return ProviderProfile(
        provider_id="vllm_default",
        description="test",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )


def _disabled_embedding_profile() -> EmbeddingProfile:
    return EmbeddingProfile(
        embedding_id="disabled",
        description="disabled",
        embedding_config={
            "_target_": "card_framework.providers.noop.NoOpEmbeddingProvider"
        },
    )


def test_select_samples_defaults_to_all_when_zero() -> None:
    samples = (_sample("a"), _sample("b"), _sample("c"))
    assert [
        sample.sample_id
        for sample in _select_samples(
            samples,
            max_samples=0,
            sample_seed=None,
        )
    ] == ["a", "b", "c"]


def test_select_samples_uses_seed_independent_of_manifest_order() -> None:
    forward = (_sample("a"), _sample("b"), _sample("c"), _sample("d"))
    reverse = tuple(reversed(forward))

    forward_ids = [
        sample.sample_id
        for sample in _select_samples(
            forward,
            max_samples=3,
            sample_seed=23,
        )
    ]
    reverse_ids = [
        sample.sample_id
        for sample in _select_samples(
            reverse,
            max_samples=3,
            sample_seed=23,
        )
    ]

    assert forward_ids == reverse_ids


def test_select_sample_shard_uses_deterministic_modulo_partition() -> None:
    selected_samples = [_sample("a"), _sample("b"), _sample("c"), _sample("d"), _sample("e")]

    assert [
        sample.sample_id
        for sample in _select_sample_shard(
            selected_samples,
            shard_index=1,
            shard_count=3,
        )
    ] == ["a", "d"]
    assert [
        sample.sample_id
        for sample in _select_sample_shard(
            selected_samples,
            shard_index=2,
            shard_count=3,
        )
    ] == ["b", "e"]
    assert [
        sample.sample_id
        for sample in _select_sample_shard(
            selected_samples,
            shard_index=3,
            shard_count=3,
        )
    ] == ["c"]


def test_resolve_sample_shard_args_rejects_out_of_range_index() -> None:
    args = argparse.Namespace(sample_shard_index=4, sample_shard_count=3)

    with pytest.raises(Exception, match="sample-shard-index"):
        _resolve_sample_shard_args(args)


def test_shared_sample_queue_claims_each_sample_once(tmp_path: Path) -> None:
    queue_root = tmp_path / "shared_queue"
    first_connection = _open_shared_sample_queue_connection(queue_root)
    second_connection = _open_shared_sample_queue_connection(queue_root)
    try:
        _initialize_shared_sample_queue(
            first_connection,
            sample_order=["sample-a", "sample-b", "sample-c"],
            target_sample_count=3,
        )
        _initialize_shared_sample_queue(
            second_connection,
            sample_order=["sample-a", "sample-b", "sample-c"],
            target_sample_count=3,
        )

        assert (
            _claim_next_shared_sample_id(
                first_connection,
                worker_label="worker-1",
                worker_pid=101,
            )
            == "sample-a"
        )
        assert (
            _claim_next_shared_sample_id(
                second_connection,
                worker_label="worker-2",
                worker_pid=202,
            )
            == "sample-b"
        )

        _mark_shared_sample_completed(
            first_connection,
            sample_id="sample-a",
            final_status="pass",
            counts_toward_target=True,
        )
        _mark_shared_sample_completed(
            second_connection,
            sample_id="sample-b",
            final_status="pass",
            counts_toward_target=True,
        )

        assert (
            _claim_next_shared_sample_id(
                first_connection,
                worker_label="worker-1",
                worker_pid=101,
            )
            == "sample-c"
        )
        _mark_shared_sample_completed(
            first_connection,
            sample_id="sample-c",
            final_status="pass",
            counts_toward_target=True,
        )
        assert (
            _claim_next_shared_sample_id(
                second_connection,
                worker_label="worker-2",
                worker_pid=202,
            )
            is None
        )
    finally:
        first_connection.close()
        second_connection.close()


def test_shared_sample_queue_rejects_mismatched_sample_order(tmp_path: Path) -> None:
    connection = _open_shared_sample_queue_connection(tmp_path / "shared_queue")
    try:
        _initialize_shared_sample_queue(
            connection,
            sample_order=["sample-a", "sample-b"],
            target_sample_count=2,
        )

        with pytest.raises(Exception, match="sample order|retained target"):
            _initialize_shared_sample_queue(
                connection,
                sample_order=["sample-b", "sample-a"],
                target_sample_count=2,
            )
    finally:
        connection.close()


def test_shared_sample_queue_replacement_exclusion_frees_one_retained_slot(
    tmp_path: Path,
) -> None:
    connection = _open_shared_sample_queue_connection(tmp_path / "shared_queue")
    try:
        _initialize_shared_sample_queue(
            connection,
            sample_order=["sample-a", "sample-b", "sample-c"],
            target_sample_count=2,
        )

        assert (
            _claim_next_shared_sample_id(
                connection,
                worker_label="worker-1",
                worker_pid=101,
            )
            == "sample-a"
        )
        assert (
            _claim_next_shared_sample_id(
                connection,
                worker_label="worker-2",
                worker_pid=202,
            )
            == "sample-b"
        )
        assert (
            _claim_next_shared_sample_id(
                connection,
                worker_label="worker-3",
                worker_pid=303,
            )
            is None
        )

        _mark_shared_sample_completed(
            connection,
            sample_id="sample-a",
            final_status="connection_unavailable",
            counts_toward_target=False,
            replacement_reason="connection_unavailable",
        )

        assert (
            _claim_next_shared_sample_id(
                connection,
                worker_label="worker-3",
                worker_pid=303,
            )
            == "sample-c"
        )
    finally:
        connection.close()


def test_shared_sample_queue_manifest_catalog_loads_only_claimed_sample(
    tmp_path: Path,
) -> None:
    manifest_path = _write_prepared_manifest_with_sidecar(
        tmp_path / "prepared_full" / "manifest.json",
        samples=[
            _sample("sample-a", transcript_path=str(tmp_path / "a.json")),
            _sample("sample-b", transcript_path=str(tmp_path / "b.json")),
            _sample("sample-c", transcript_path=str(tmp_path / "c.json")),
        ],
    )
    connection = _open_shared_sample_queue_connection(tmp_path / "shared_queue")
    try:
        sample_order, target_sample_count = _initialize_shared_sample_queue_from_manifest(
            connection,
            manifest_path=manifest_path,
            requested_target_sample_count=2,
            sample_seed=None,
        )

        assert sample_order == ["sample-a", "sample-b", "sample-c"]
        assert target_sample_count == 2

        claimed_sample_id = _claim_next_shared_sample_id(
            connection,
            worker_label="worker-1",
            worker_pid=101,
        )

        assert claimed_sample_id == "sample-a"
        claimed_sample = _load_shared_sample(
            connection,
            sample_id=claimed_sample_id,
        )
        assert claimed_sample.sample_id == "sample-a"
        assert claimed_sample.transcript_path == str(tmp_path / "a.json")
    finally:
        connection.close()


def test_normalize_max_iterations_maps_non_positive_values_to_none() -> None:
    assert _normalize_max_iterations(None) is None
    assert _normalize_max_iterations(0) is None
    assert _normalize_max_iterations(-1) is None
    assert _normalize_max_iterations(4) == 4


def test_resolve_orchestrator_timeouts_defaults_to_no_timeout() -> None:
    args = argparse.Namespace(
        summarizer_timeout_seconds=None,
        critic_timeout_seconds=None,
        retrieval_timeout_seconds=None,
        index_timeout_seconds=None,
    )
    assert _resolve_orchestrator_timeouts(args) == {
        "summarizer": None,
        "critic": None,
        "retrieval": None,
        "index": None,
    }


def test_resolve_orchestrator_timeouts_accepts_positive_overrides() -> None:
    args = argparse.Namespace(
        summarizer_timeout_seconds=25.0,
        critic_timeout_seconds=35.0,
        retrieval_timeout_seconds=45.0,
        index_timeout_seconds=55.0,
    )
    assert _resolve_orchestrator_timeouts(args) == {
        "summarizer": 25.0,
        "critic": 35.0,
        "retrieval": 45.0,
        "index": 55.0,
    }


def test_classify_sample_failure_maps_timeout_messages_to_stage_timeout() -> None:
    final_status, error_message = _classify_sample_failure(
        TimeoutError("summarizer request timed out after 900s")
    )

    assert final_status == "stage_timeout"
    assert "timed out" in error_message


@pytest.mark.parametrize(
    ("message", "expected_fragment"),
    [
        (
            "[WinError 10061] No connection could be made because the target machine actively refused it",
            "actively refused",
        ),
        (
            "[WinError 10054] An existing connection was forcibly closed by the remote host",
            "forcibly closed by the remote host",
        ),
        (
            "httpx.ReadError: [Errno 104] Connection reset by peer",
            "Connection reset by peer",
        ),
        (
            "Connection aborted.",
            "Connection aborted",
        ),
    ],
)
def test_classify_sample_failure_maps_connection_reset_and_refused_messages_to_connection_unavailable(
    message: str,
    expected_fragment: str,
) -> None:
    final_status, error_message = _classify_sample_failure(RuntimeError(message))

    assert final_status == "connection_unavailable"
    assert expected_fragment in error_message


def test_compute_composite_score_uses_only_three_judge_dimensions() -> None:
    assert (
        _compute_composite_score(
            factualness=0.9,
            naturalness=0.6,
            speaker_grammar_similarity=0.3,
        )
        == 0.6
    )


def test_resolve_override_target_seconds_uses_duration_preset() -> None:
    args = argparse.Namespace(duration_preset="5m", target_seconds_override=None)

    assert _resolve_override_target_seconds(args) == 300


def test_resolve_sample_target_seconds_prefers_override() -> None:
    sample = _sample("a")
    args = argparse.Namespace(duration_preset="1m", target_seconds_override=None)

    assert _resolve_sample_target_seconds(sample, args) == 60


def test_evaluate_sample_preflight_skips_incompatible_long_target(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(
        transcript_path,
        text=" ".join(["word"] * 124),
    )

    _, stats, reason = _evaluate_sample_preflight(
        sample=_sample("a", transcript_path=str(transcript_path)),
        target_seconds=300,
        reference_wpm=150.0,
    )

    assert reason is not None
    assert "exceeds transcript capacity" in reason
    assert stats["segment_count"] == 1
    assert stats["transcript_word_count"] == 124
    assert stats["max_feasible_summary_seconds"] == 60.0


def test_evaluate_sample_preflight_allows_compatible_short_target(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(
        transcript_path,
        text=" ".join(["word"] * 124),
    )

    _, stats, reason = _evaluate_sample_preflight(
        sample=_sample("a", transcript_path=str(transcript_path)),
        target_seconds=30,
        reference_wpm=150.0,
    )

    assert reason is None
    assert stats["max_feasible_summary_seconds"] == 60.0


def test_summarize_results_excludes_skipped_samples_from_aggregates() -> None:
    skipped = _build_preflight_failed_result(
        sample=_sample("skip"),
        target_seconds=300,
        reason="too long",
        stats={"max_feasible_summary_seconds": 60.0},
    )

    summary = _summarize_results(
        [skipped, _completed_result("ok"), _failed_result("bad")],
        target_sample_count=2,
    )

    assert summary.target_sample_count == 2
    assert summary.sample_count == 2
    assert summary.attempted_candidate_count == 3
    assert summary.replacement_count == 1
    assert summary.executed_count == 2
    assert summary.skipped_count == 0
    assert summary.success_count == 1
    assert summary.failure_count == 1
    assert summary.mean_composite == 0.35


def test_prepare_benchmark_audio_cfg_enables_faithful_live_draft_audio(
    tmp_path,
) -> None:
    audio_cfg = {
        "work_dir": "artifacts/audio_stage",
        "speaker_samples": {"enabled": False},
        "voice_clone": {"enabled": False, "live_drafting": {"enabled": False}},
    }

    prepared = _prepare_benchmark_audio_cfg(audio_cfg, run_root=tmp_path)

    assert prepared["speaker_samples"]["enabled"] is True
    assert prepared["voice_clone"]["enabled"] is True
    assert prepared["voice_clone"]["use_fp16"] is True
    assert prepared["voice_clone"]["live_drafting"]["enabled"] is True
    assert prepared["work_dir"] == str((tmp_path / "audio_pipeline").resolve())


def test_prepare_provider_config_applies_benchmark_vllm_timeout() -> None:
    profile = ProviderProfile(
        provider_id="vllm_default",
        description="test",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )

    prepared = _prepare_provider_config(profile)

    assert prepared["request_timeout_seconds"] == 60.0


def test_prepare_provider_config_applies_transcript_only_critic_overrides() -> None:
    profile = ProviderProfile(
        provider_id="vllm_default",
        description="test",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )

    prepared = _prepare_provider_config(
        profile,
        role="critic",
        transcript_only=True,
    )

    assert prepared["request_timeout_seconds"] == 300.0
    assert prepared["enable_thinking"] is True
    assert prepared["thinking_extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_execute_benchmark_short_circuits_when_all_samples_fail_preflight(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(
        transcript_path,
        text=" ".join(["word"] * 124),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_sample",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "sample",
                "episode_data_path": str(tmp_path / "episode.jsonl.gz"),
                "turn_data_path": str(tmp_path / "turn.jsonl.gz"),
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "sporc_00001_example",
                        "dataset": "sporc_sample",
                        "episode_key": "example.mp3",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "reference",
                        "target_seconds": 12,
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--target-seconds-override",
            "300",
            "--max-samples",
            "1",
        ]
    )

    def _unexpected_instantiate(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("provider startup should not happen for all-skipped runs")

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.hydra.utils.instantiate",
        _unexpected_instantiate,
    )

    report = execute_benchmark(args)

    assert report.status == "failed"
    assert report.summary.target_sample_count == 1
    assert report.summary.sample_count == 0
    assert report.summary.attempted_candidate_count == 1
    assert report.summary.replacement_count == 1
    assert report.summary.executed_count == 0
    assert report.summary.skipped_count == 0
    assert report.samples[0].status == "skipped"
    assert report.samples[0].final_status == "preflight_failed"
    assert report.samples[0].counts_toward_target is False


def test_build_execute_parser_accepts_sweep_child_progress_flags() -> None:
    parser = argparse.ArgumentParser()
    build_execute_parser(parser)

    args = parser.parse_args(
        [
            "--retrieval-port",
            "5001",
            "--summarizer-port",
            "5002",
            "--critic-port",
            "5003",
            "--disable-stream-output",
            "--emit-progress-events",
            "--emit-progress-heartbeats",
            "--progress-heartbeat-seconds",
            "45",
            "--checkpoint-path",
            "checkpoint.json",
            "--resume-from-checkpoint",
            "--resume-failure-reason",
            "sweep_watchdog_stalled",
            "--sample-shard-index",
            "2",
            "--sample-shard-count",
            "3",
        ]
    )

    assert args.retrieval_port == 5001
    assert args.summarizer_port == 5002
    assert args.critic_port == 5003
    assert args.disable_stream_output is True
    assert args.emit_progress_events is True
    assert args.emit_progress_heartbeats is True
    assert args.progress_heartbeat_seconds == 45.0
    assert args.checkpoint_path == "checkpoint.json"
    assert args.resume_from_checkpoint is True
    assert args.resume_failure_reason == "sweep_watchdog_stalled"
    assert args.sample_shard_index == 2
    assert args.sample_shard_count == 3


def test_execute_benchmark_treats_empty_shard_as_completed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcript_a = tmp_path / "a.transcript.json"
    transcript_b = tmp_path / "b.transcript.json"
    _write_transcript(transcript_a, text=" ".join(["word"] * 124))
    _write_transcript(transcript_b, text=" ".join(["word"] * 124))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_sample",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "sample",
                "episode_data_path": str(tmp_path / "episode.jsonl.gz"),
                "turn_data_path": str(tmp_path / "turn.jsonl.gz"),
                "sample_count": 2,
                "samples": [
                    {
                        "sample_id": "a",
                        "dataset": "sporc_sample",
                        "episode_key": "a.mp3",
                        "transcript_path": str(transcript_a),
                        "reference_summary": "reference a",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                    {
                        "sample_id": "b",
                        "dataset": "sporc_sample",
                        "episode_key": "b.mp3",
                        "transcript_path": str(transcript_b),
                        "reference_summary": "reference b",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    parser = argparse.ArgumentParser()
    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--transcript-only",
            "--max-samples",
            "2",
            "--sample-shard-index",
            "3",
            "--sample-shard-count",
            "3",
        ]
    )

    provider_profile = _benchmark_provider_profile()
    embedding_profile = _disabled_embedding_profile()

    async def _unexpected_start_runtime(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise AssertionError("runtime startup should not happen for empty shards")

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._load_base_config",
        lambda path: _benchmark_base_cfg(tmp_path),
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.load_provider_profiles",
        lambda path: [provider_profile],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._find_provider_profile",
        lambda profiles, provider_id: provider_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.missing_required_env",
        lambda profile: [],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._resolve_embedding_profile",
        lambda base_cfg, embedding_mode: embedding_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._start_managed_runtime",
        _unexpected_start_runtime,
    )

    report = execute_benchmark(args)

    assert report.status == "completed"
    assert report.summary.target_sample_count == 0
    assert report.summary.sample_count == 0
    assert report.sample_shard_index == 3
    assert report.sample_shard_count == 3


def test_execute_benchmark_resumes_checkpoint_and_marks_orphaned_sample_failed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcript_a = tmp_path / "a.transcript.json"
    transcript_b = tmp_path / "b.transcript.json"
    transcript_c = tmp_path / "c.transcript.json"
    _write_transcript(transcript_a, text=" ".join(["word"] * 124))
    _write_transcript(transcript_b, text=" ".join(["word"] * 124))
    _write_empty_transcript(transcript_c)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_sample",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "sample",
                "episode_data_path": str(tmp_path / "episode.jsonl.gz"),
                "turn_data_path": str(tmp_path / "turn.jsonl.gz"),
                "sample_count": 3,
                "samples": [
                    {
                        "sample_id": "a",
                        "dataset": "sporc_sample",
                        "episode_key": "a.mp3",
                        "transcript_path": str(transcript_a),
                        "reference_summary": "reference a",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                    {
                        "sample_id": "b",
                        "dataset": "sporc_sample",
                        "episode_key": "b.mp3",
                        "transcript_path": str(transcript_b),
                        "reference_summary": "reference b",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                    {
                        "sample_id": "c",
                        "dataset": "sporc_sample",
                        "episode_key": "c.mp3",
                        "transcript_path": str(transcript_c),
                        "reference_summary": "reference c",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "checkpoint_version": 1,
                "run_id": "resume-run",
                "dataset_manifest_path": str(manifest_path.resolve()),
                "dataset_variant": "sample",
                "duration_preset": "",
                "override_target_seconds": None,
                "transcript_only": True,
                "sample_seed": None,
                "sample_order": ["a", "b", "c"],
                "completed_results": [asdict(_completed_result("a"))],
                "active_sample": {
                    "sample_index": 2,
                    "sample_count": 3,
                    "sample_id": "b",
                    "target_seconds": 30,
                    "stage": "summarizer",
                    "iteration": 2,
                    "elapsed_sample_seconds": 42.5,
                    "last_activity_age_seconds": 601.0,
                },
                "resume_count": 0,
                "recovered_stall_count": 0,
                "updated_at_utc": "2026-03-15T04:30:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    parser = argparse.ArgumentParser()
    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--transcript-only",
            "--max-samples",
            "3",
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume-from-checkpoint",
        ]
    )
    provider_profile = _benchmark_provider_profile()
    embedding_profile = _disabled_embedding_profile()

    async def _unexpected_start_runtime(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise AssertionError("runtime startup should not happen on all-terminal resume")

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._load_base_config",
        lambda path: _benchmark_base_cfg(tmp_path),
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.load_provider_profiles",
        lambda path: [provider_profile],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._find_provider_profile",
        lambda profiles, provider_id: provider_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.missing_required_env",
        lambda profile: [],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._resolve_embedding_profile",
        lambda base_cfg, embedding_mode: embedding_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._start_managed_runtime",
        _unexpected_start_runtime,
    )

    report = execute_benchmark(args)

    assert report.run_id == "resume-run"
    assert report.status == "partial"
    assert report.resumed_from_checkpoint is True
    assert report.resume_count == 1
    assert [sample.sample_id for sample in report.samples] == ["a", "b", "c"]
    assert report.samples[1].status == "failed"
    assert report.samples[1].final_status == "resume_interrupted_sample"
    assert report.samples[1].duration_seconds == 42.5
    assert report.samples[2].status == "skipped"
    assert report.samples[2].final_status == "preflight_failed"

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["resume_count"] == 1
    assert checkpoint_payload["active_sample"] is None
    assert len(checkpoint_payload["completed_results"]) == 3


def test_write_checkpoint_payload_retries_transient_permission_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    payload = {"checkpoint_version": 1, "run_id": "retry-run"}
    original_replace = Path.replace
    replace_attempts = {"count": 0}

    def _flaky_replace(self: Path, target: Path) -> Path:
        if Path(target) == checkpoint_path and replace_attempts["count"] < 2:
            replace_attempts["count"] += 1
            raise PermissionError("[WinError 32] The process cannot access the file")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _flaky_replace)

    _write_checkpoint_payload(checkpoint_path, payload)

    persisted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert persisted["run_id"] == "retry-run"
    assert replace_attempts["count"] == 2


def test_write_checkpoint_payload_serializes_concurrent_writers(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    failures: list[Exception] = []
    start_event = threading.Event()

    def _writer(run_id: str) -> None:
        start_event.wait()
        try:
            _write_checkpoint_payload(
                checkpoint_path,
                {"checkpoint_version": 1, "run_id": run_id},
            )
        except Exception as exc:  # pragma: no cover - failure surfaced below
            failures.append(exc)

    threads = [
        threading.Thread(target=_writer, args=("run-a",)),
        threading.Thread(target=_writer, args=("run-b",)),
    ]
    for thread in threads:
        thread.start()
    start_event.set()
    for thread in threads:
        thread.join()

    assert failures == []
    persisted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert persisted["run_id"] in {"run-a", "run-b"}


def test_execute_benchmark_continues_after_failed_sample_and_rebuilds_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcript_a = tmp_path / "a.transcript.json"
    transcript_b = tmp_path / "b.transcript.json"
    _write_transcript(transcript_a, text=" ".join(["word"] * 124))
    _write_transcript(transcript_b, text=" ".join(["word"] * 124))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_sample",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "sample",
                "episode_data_path": str(tmp_path / "episode.jsonl.gz"),
                "turn_data_path": str(tmp_path / "turn.jsonl.gz"),
                "sample_count": 2,
                "samples": [
                    {
                        "sample_id": "a",
                        "dataset": "sporc_sample",
                        "episode_key": "a.mp3",
                        "transcript_path": str(transcript_a),
                        "reference_summary": "reference a",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                    {
                        "sample_id": "b",
                        "dataset": "sporc_sample",
                        "episode_key": "b.mp3",
                        "transcript_path": str(transcript_b),
                        "reference_summary": "reference b",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    parser = argparse.ArgumentParser()
    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--transcript-only",
            "--max-samples",
            "2",
        ]
    )

    base_cfg = _benchmark_base_cfg(tmp_path)
    provider_profile = _benchmark_provider_profile()
    embedding_profile = _disabled_embedding_profile()
    start_calls: list[bool] = []
    stop_calls: list[str] = []

    async def _fake_start_runtime(**kwargs):  # type: ignore[no-untyped-def]
        start_calls.append(bool(kwargs["force_dynamic_ports"]))
        return SimpleNamespace(
            orchestrator=object(),
            judge=object(),
            event_capture=object(),
            artifact_recorder=_SampleArtifactRecorder(run_root=kwargs["run_root"]),
            retrieval_enabled=False,
            shared_agent_client=object(),
            servers=[],
            retrieval_port=5001,
            summarizer_port=5002,
            critic_port=5003,
        )

    async def _fake_stop_runtime(runtime):  # type: ignore[no-untyped-def]
        if runtime is not None:
            stop_calls.append("stopped")

    async def _fake_execute_sample(*, sample, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        if sample.sample_id == "a":
            failed = _failed_result("a")
            failed.final_status = "stage_timeout"
            failed.error_message = "summarizer request timed out after 900s"
            return failed
        return _completed_result(sample.sample_id)

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._load_base_config",
        lambda path: base_cfg,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.load_provider_profiles",
        lambda path: [provider_profile],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._find_provider_profile",
        lambda profiles, provider_id: provider_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.missing_required_env",
        lambda profile: [],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._resolve_embedding_profile",
        lambda base_cfg, embedding_mode: embedding_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._start_managed_runtime",
        _fake_start_runtime,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._stop_managed_runtime",
        _fake_stop_runtime,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._execute_sample",
        _fake_execute_sample,
    )

    report = execute_benchmark(args)

    assert report.status == "partial"
    assert [sample.sample_id for sample in report.samples] == ["a", "b"]
    assert report.samples[0].final_status == "stage_timeout"
    assert report.samples[1].status == "completed"
    assert start_calls == [False, True]
    assert len(stop_calls) == 2


def test_build_execute_parser_accepts_sample_seed() -> None:
    parser = argparse.ArgumentParser()
    build_execute_parser(parser)

    args = parser.parse_args(["--sample-seed", "29"])

    assert args.sample_seed == 29


def test_build_execute_parser_accepts_judge_only_sources() -> None:
    parser = argparse.ArgumentParser()
    build_execute_parser(parser)

    args = parser.parse_args(
        [
            "--judge-only-source",
            "run-a",
            "--judge-only-source",
            "run-b",
        ]
    )

    assert args.judge_only_source == ["run-a", "run-b"]


def test_execute_benchmark_rejudges_persisted_artifacts_from_sweep_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(transcript_path, text="hello from the transcript benchmark")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_sample",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "sample",
                "episode_data_path": str(tmp_path / "episode.jsonl.gz"),
                "turn_data_path": str(tmp_path / "turn.jsonl.gz"),
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "sample-a",
                        "dataset": "sporc_sample",
                        "episode_key": "sample-a.mp3",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "reference",
                        "target_seconds": 30,
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source_root = tmp_path / "source-runs"
    sweep_root = tmp_path / "source-sweep"
    (sweep_root / "logs").mkdir(parents=True)

    def _write_source_report(
        *,
        run_id: str,
        duration_preset: str,
        target_seconds: int,
    ) -> Path:
        run_root = source_root / run_id
        summary_dir = run_root / "samples" / "sample-a" / "final"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "summary.xml"
        summary_path.write_text(
            '<SPEAKER_00 emo_preset="neutral">hello</SPEAKER_00>',
            encoding="utf-8",
        )
        report_path = run_root / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "partial",
                    "dataset_manifest_path": str(manifest_path),
                    "sporc_dataset_id": "blitt/SPoRC",
                    "dataset_variant": "sample",
                    "summarizer_provider": "vllm_default",
                    "critic_provider": "vllm_default",
                    "judge_provider": "vllm_default",
                    "embedding_mode": "disabled",
                    "commands_executed": ["python -m card_framework ..."],
                    "git_commit": "deadbeef",
                    "git_branch": "test",
                    "report_path": str(report_path),
                    "notes_path": str(run_root / "notes.md"),
                    "results_path": str(run_root / "results.md"),
                    "samples": [
                        {
                            "sample_id": "sample-a",
                            "dataset": "sporc_sample",
                            "episode_key": "sample-a.mp3",
                            "status": "completed",
                            "target_seconds": target_seconds,
                            "candidate_seconds": float(target_seconds),
                            "converged": True,
                            "iterations_run": 1,
                            "final_status": "pass",
                            "duration_seconds": 12.0,
                            "retrieval_events": 1,
                            "tool_invocations": 2,
                            "metrics": {
                                "factualness": 0.1,
                                "naturalness": 0.2,
                                "speaker_grammar_similarity": 0.3,
                                "composite": 0.2,
                            },
                            "counts_toward_target": True,
                            "judge_status": "ok",
                            "judge_scores": {
                                "Factualness": 0.1,
                                "Naturalness": 0.2,
                                "Speaker Grammar Similarity": 0.3,
                            },
                            "final_summary_path": str(summary_path),
                        }
                    ],
                    "summary": {
                        "target_sample_count": 1,
                        "sample_count": 1,
                        "attempted_candidate_count": 1,
                        "replacement_count": 0,
                        "executed_count": 1,
                        "skipped_count": 0,
                        "success_count": 1,
                        "failure_count": 0,
                        "mean_factualness": 0.1,
                        "mean_naturalness": 0.2,
                        "mean_speaker_grammar_similarity": 0.3,
                        "mean_composite": 0.2,
                    },
                    "sample_seed": 42,
                    "duration_tolerance_ratio": 0.05,
                    "transcript_only": True,
                    "duration_preset": duration_preset,
                    "override_target_seconds": target_seconds,
                    "checkpoint_path": None,
                    "resumed_from_checkpoint": False,
                    "resume_count": 0,
                    "sample_shard_index": 1,
                    "sample_shard_count": 1,
                }
            ),
            encoding="utf-8",
        )
        return report_path

    report_30s = _write_source_report(
        run_id="source-30s",
        duration_preset="30s",
        target_seconds=30,
    )
    report_1m = _write_source_report(
        run_id="source-1m",
        duration_preset="1m",
        target_seconds=60,
    )
    (sweep_root / "report.json").write_text(
        json.dumps({"sweep_id": "sweep-1"}),
        encoding="utf-8",
    )
    (sweep_root / "logs" / "30s.log").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sporc_progress_event": "run_complete",
                        "report_path": str(report_30s),
                    }
                ),
                json.dumps(
                    {
                        "sporc_progress_event": "run_complete",
                        "report_path": str(report_1m),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parser = argparse.ArgumentParser()
    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--judge-provider",
            "deepseek_judge",
            "--judge-only-source",
            str(sweep_root),
            "--max-samples",
            "10",
            "--disable-stream-output",
        ]
    )

    provider_profile = ProviderProfile(
        provider_id="deepseek_judge",
        description="judge",
        llm_config={"_target_": "tests.fake.Provider"},
    )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._load_base_config",
        lambda path: _benchmark_base_cfg(tmp_path),
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.load_provider_profiles",
        lambda path: [provider_profile],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._find_provider_profile",
        lambda profiles, provider_id: provider_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.missing_required_env",
        lambda profile: [],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.hydra.utils.instantiate",
        lambda cfg: _CallbackAwareProvider(
            agent_name="judge",
            response_text=json.dumps(
                {
                    "Factualness": 0.75,
                    "Naturalness": 0.5,
                    "Speaker Grammar Similarity": 0.25,
                }
            ),
            reasoning_tokens=["judge-thought"],
            content_tokens=["judge-content"],
        ),
    )

    report = execute_benchmark(args)

    assert report.status == "completed"
    assert report.summary.sample_count == 2
    assert report.summary.target_sample_count == 2
    assert report.duration_preset == "mixed"
    assert report.judge_provider == "deepseek_judge"
    assert report.summarizer_provider == "vllm_default"
    assert report.critic_provider == "vllm_default"
    assert report.sample_seed == 42
    assert [sample.target_seconds for sample in report.samples] == [30, 60]
    assert report.samples[0].sample_id.startswith("sample-a__30s__")
    assert report.samples[1].sample_id.startswith("sample-a__1m__")
    assert all(sample.judge_scores is not None for sample in report.samples)
    assert [sample.metrics.composite for sample in report.samples] == [0.5, 0.5]
    assert all(sample.sample_artifact_dir is not None for sample in report.samples)
    assert all(sample.judge_trace_dir is not None for sample in report.samples)
    assert all(Path(sample.final_summary_path).exists() for sample in report.samples)
    assert all(Path(sample.sample_result_artifact_path).exists() for sample in report.samples)


def test_execute_benchmark_backfills_sidecar_candidates_until_target_or_exhaustion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(
        transcript_path,
        text=" ".join(["word"] * 124),
    )
    samples_path = tmp_path / "samples.jsonl.gz"
    with gzip.open(samples_path, "wt", encoding="utf-8") as handle:
        for sample_id in ("sporc_00001_example", "sporc_00002_unused"):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "dataset": "sporc_full",
                        "episode_key": f"{sample_id}.mp3",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "reference",
                        "target_seconds": 12,
                        "metadata": {},
                    }
                )
            )
            handle.write("\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episode.jsonl.gz"),
                "turn_data_path": str(tmp_path / "turn.jsonl.gz"),
                "sample_count": 2,
                "samples_path": str(samples_path),
            }
        ),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--target-seconds-override",
            "300",
            "--max-samples",
            "1",
        ]
    )

    def _unexpected_instantiate(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("provider startup should not happen for all-skipped runs")

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.hydra.utils.instantiate",
        _unexpected_instantiate,
    )

    report = execute_benchmark(args)

    assert report.summary.target_sample_count == 1
    assert report.summary.sample_count == 0
    assert report.summary.attempted_candidate_count == 2
    assert report.summary.replacement_count == 2
    assert [sample.sample_id for sample in report.samples] == [
        "sporc_00001_example",
        "sporc_00002_unused",
    ]
    assert all(sample.status == "skipped" for sample in report.samples)
    assert all(sample.counts_toward_target is False for sample in report.samples)


def test_prepare_provider_config_applies_judge_specific_vllm_overrides() -> None:
    profile = ProviderProfile(
        provider_id="vllm_default",
        description="test",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )

    prepared = _prepare_provider_config(profile, role="judge")

    assert prepared["enable_thinking"] is True
    assert prepared["thinking_extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
