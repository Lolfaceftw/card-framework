from __future__ import annotations

import asyncio
from pathlib import Path

from card_framework.audio_pipeline.summary_turn_emotion_linker import (
    SummaryTurnEmotionLinker,
    SummaryTurnEmotionLinkingConfig,
)
from card_framework.audio_pipeline.voice_clone_orchestrator import VoiceCloneRunResult
from card_framework.orchestration.stage_orchestrator import StageOrchestrator
from card_framework.orchestration.transcript import Transcript
from card_framework.runtime.pipeline_plan import PipelineStagePlan


class _FakeLoopOrchestrator:
    async def run_loop_with_diagnostics(self, **kwargs):
        del kwargs
        return {
            "converged": True,
            "draft": "<SPEAKER_00>Hello from stage two</SPEAKER_00>",
        }


class _FakeSessionStore:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.completed: list[str] = []

    def mark_stage_started(self, *, session_id: str, stage_name: str) -> None:
        del session_id
        self.started.append(stage_name)

    def mark_stage_completed(self, *, session_id: str, stage_name: str, detail=None) -> None:
        del session_id, detail
        self.completed.append(stage_name)


class _FakeVoiceCloneOrchestrator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir


class _CapturingStageThreeOrchestrator(StageOrchestrator):
    __slots__ = ("captured",)

    def __init__(self, *args, captured: dict[str, str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "captured", captured)

    def _run_voice_clone_stage(self, *, summary_xml: str, transcript: Transcript):
        del transcript
        self.captured["summary_xml"] = summary_xml
        return _dummy_voice_clone_result(self.project_root)

    def _run_interjector_stage(self, *, summary_xml: str, **kwargs):
        del summary_xml, kwargs
        return None


class _CapturingLiveDraftStageOrchestrator(StageOrchestrator):
    __slots__ = ("captured",)

    def __init__(self, *args, captured: dict[str, str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "captured", captured)

    def _prepare_transcript_for_live_draft_voice_clone(
        self,
        *,
        transcript: Transcript,
    ) -> tuple[Transcript, Path]:
        return transcript, Path(transcript.metadata["speaker_samples_manifest_path"])

    def _run_live_batch_voice_clone_stage(
        self,
        *,
        summary_xml: str,
        transcript: Transcript,
    ):
        del transcript
        self.captured["summary_xml"] = summary_xml
        return _dummy_voice_clone_result(self.project_root)

    def _run_interjector_stage(self, *, summary_xml: str, **kwargs):
        del summary_xml, kwargs
        return None


def _dummy_voice_clone_result(tmp_path: Path) -> VoiceCloneRunResult:
    output_dir = tmp_path / "voice_clone"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return VoiceCloneRunResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        generated_at_utc="2026-03-23T00:00:00+00:00",
        artifacts=(),
        merged_output_audio_path=None,
    )


def test_run_skips_stage_three_when_voice_cloning_is_disabled(tmp_path: Path) -> None:
    session_store = _FakeSessionStore()
    orchestrator = StageOrchestrator(
        orchestrator=_FakeLoopOrchestrator(),
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        project_root=tmp_path,
        target_seconds=60,
        duration_tolerance_ratio=0.1,
        max_iterations=1,
        session_store=session_store,
        session_id="session-1",
    )

    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "hello world",
                }
            ],
            "metadata": {},
        }
    )

    asyncio.run(orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert session_store.started == ["stage-2"]
    assert session_store.completed == ["stage-2"]
    assert (tmp_path / "summary.xml").exists()


def test_run_links_summary_turn_emotions_before_stage_three(tmp_path: Path) -> None:
    session_store = _FakeSessionStore()
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "hello from stage two",
                    "emotion_vector": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.08],
                    "emotion_vector_confidence": 0.9,
                }
            ],
            "metadata": {
                "speaker_samples_manifest_path": str(tmp_path / "speaker_samples.json")
            },
        }
    )
    (tmp_path / "speaker_samples.json").write_text(
        '{"samples":[{"speaker":"SPEAKER_00","path":"sample.wav","duration_ms":1000}]}',
        encoding="utf-8",
    )
    captured: dict[str, str] = {}
    orchestrator = _CapturingStageThreeOrchestrator(
        orchestrator=_FakeLoopOrchestrator(),
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        project_root=tmp_path,
        target_seconds=60,
        duration_tolerance_ratio=0.1,
        max_iterations=1,
        voice_clone_orchestrator=_FakeVoiceCloneOrchestrator(tmp_path / "voice_clone"),
        summary_turn_emotion_linker=SummaryTurnEmotionLinker(
            SummaryTurnEmotionLinkingConfig(
                top_k=1,
                min_support_score=0.0,
                apply_indextts_bias=False,
                force_live_draft_rerender=False,
            )
        ),
        session_store=session_store,
        session_id="session-1",
        captured=captured,
    )

    asyncio.run(orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert 'emotion_vector_source="summary_turn_linking"' in captured["summary_xml"]
    assert 'source_segment_indices="0"' in captured["summary_xml"]


def test_live_draft_rerenders_final_summary_when_linking_is_enabled(tmp_path: Path) -> None:
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "hello from stage two",
                    "emotion_vector": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.08],
                }
            ],
            "metadata": {
                "speaker_samples_manifest_path": str(tmp_path / "speaker_samples.json")
            },
        }
    )
    (tmp_path / "speaker_samples.json").write_text(
        '{"samples":[{"speaker":"SPEAKER_00","path":"sample.wav","duration_ms":1000}]}',
        encoding="utf-8",
    )
    captured: dict[str, str] = {}
    orchestrator = _CapturingLiveDraftStageOrchestrator(
        orchestrator=_FakeLoopOrchestrator(),
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        project_root=tmp_path,
        target_seconds=60,
        duration_tolerance_ratio=0.1,
        max_iterations=1,
        voice_clone_orchestrator=_FakeVoiceCloneOrchestrator(tmp_path / "voice_clone"),
        live_draft_audio_enabled=True,
        summary_turn_emotion_linker=SummaryTurnEmotionLinker(
            SummaryTurnEmotionLinkingConfig(
                top_k=1,
                min_support_score=0.0,
                apply_indextts_bias=False,
                force_live_draft_rerender=True,
            )
        ),
        captured=captured,
    )

    asyncio.run(orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert 'emotion_vector_source="summary_turn_linking"' in captured["summary_xml"]


def test_stage_three_only_links_summary_turn_emotions(tmp_path: Path) -> None:
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "hello from stage two",
                    "emotion_vector": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.08],
                }
            ],
            "metadata": {
                "speaker_samples_manifest_path": str(tmp_path / "speaker_samples.json")
            },
        }
    )
    final_summary_path = tmp_path / "existing_summary.xml"
    final_summary_path.write_text(
        "<SPEAKER_00>Hello from stage two</SPEAKER_00>",
        encoding="utf-8",
    )
    (tmp_path / "speaker_samples.json").write_text(
        '{"samples":[{"speaker":"SPEAKER_00","path":"sample.wav","duration_ms":1000}]}',
        encoding="utf-8",
    )
    captured: dict[str, str] = {}
    orchestrator = _CapturingStageThreeOrchestrator(
        orchestrator=_FakeLoopOrchestrator(),
        stage_plan=PipelineStagePlan(
            start_stage="stage-3",
            final_summary_path=final_summary_path,
        ),
        project_root=tmp_path,
        target_seconds=60,
        duration_tolerance_ratio=0.1,
        max_iterations=1,
        voice_clone_orchestrator=_FakeVoiceCloneOrchestrator(tmp_path / "voice_clone"),
        summary_turn_emotion_linker=SummaryTurnEmotionLinker(
            SummaryTurnEmotionLinkingConfig(
                top_k=1,
                min_support_score=0.0,
                apply_indextts_bias=False,
            )
        ),
        captured=captured,
    )

    asyncio.run(orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert 'emotion_vector_source="summary_turn_linking"' in captured["summary_xml"]
