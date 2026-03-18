from __future__ import annotations

import asyncio
from pathlib import Path

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
