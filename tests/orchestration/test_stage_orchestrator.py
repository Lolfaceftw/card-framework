"""Tests for stage-level orchestration flow control."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestration.stage_orchestrator import StageOrchestrator
from orchestration.transcript import Transcript
from pipeline_plan import PipelineStagePlan


class _FakeOrchestrator:
    """Capture stage-orchestrator calls without external dependencies."""

    def __init__(self) -> None:
        self.indexed_transcript: Transcript | None = None
        self.loop_kwargs: dict[str, Any] | None = None

    async def index_transcript(self, transcript: Transcript) -> int:
        self.indexed_transcript = transcript
        return 1

    async def run_loop(self, **kwargs: Any) -> str:
        self.loop_kwargs = kwargs
        return "<SPEAKER_00>loop</SPEAKER_00>"


@dataclass(slots=True)
class _FakeVoiceCloneResult:
    output_dir: Path
    artifacts: tuple[str, ...]


class _FakeVoiceCloneOrchestrator:
    """Capture voice-clone stage invocation without external dependencies."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.summary_xml: str | None = None
        self.manifest_path: Path | None = None

    def run(
        self,
        *,
        summary_xml: str,
        speaker_samples_manifest_path: Path,
        progress_callback=None,
    ) -> _FakeVoiceCloneResult:
        del progress_callback
        self.summary_xml = summary_xml
        self.manifest_path = speaker_samples_manifest_path
        return _FakeVoiceCloneResult(output_dir=self.output_dir, artifacts=("turn.wav",))


class _FakeGpuHeartbeat:
    """Capture heartbeat lifecycle interactions during voice-clone stage."""

    def __init__(self) -> None:
        self.started_with_pid: int | None = None
        self.stop_calls = 0

    def start(self, *, pipeline_root_pid: int) -> None:
        self.started_with_pid = pipeline_root_pid

    def stop(self) -> None:
        self.stop_calls += 1


def _build_stage_orchestrator(
    *,
    stage_plan: PipelineStagePlan,
    fake_orchestrator: _FakeOrchestrator,
    project_root: Path,
    fake_voice_clone_orchestrator: _FakeVoiceCloneOrchestrator | None = None,
    fake_gpu_heartbeat: _FakeGpuHeartbeat | None = None,
) -> StageOrchestrator:
    """Create a stage-orchestrator with stable defaults for tests."""
    return StageOrchestrator(
        orchestrator=fake_orchestrator,  # type: ignore[arg-type]
        stage_plan=stage_plan,
        project_root=project_root,
        min_words=10,
        max_words=30,
        max_iterations=3,
        voice_clone_orchestrator=fake_voice_clone_orchestrator,  # type: ignore[arg-type]
        voice_clone_gpu_heartbeat=fake_gpu_heartbeat,  # type: ignore[arg-type]
    )


def test_run_stage_two_uses_full_transcript_when_retrieval_disabled(
    tmp_path: Path,
) -> None:
    """Build full transcript text for stage-2 when retrieval is disabled."""
    fake_orchestrator = _FakeOrchestrator()
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
    )
    transcript = {"segments": [{"speaker": "SPEAKER_00", "text": "hello world"}]}

    asyncio.run(stage_orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert fake_orchestrator.indexed_transcript is None
    assert fake_orchestrator.loop_kwargs is not None
    assert (
        fake_orchestrator.loop_kwargs["full_transcript_text"]
        == "[SPEAKER_00]: hello world\n"
    )
    assert (tmp_path / "summary.xml").read_text(encoding="utf-8") == (
        "<SPEAKER_00>loop</SPEAKER_00>\n"
    )


def test_run_stage_two_indexes_when_retrieval_enabled(tmp_path: Path) -> None:
    """Index transcript before loop when retrieval tools are available."""
    fake_orchestrator = _FakeOrchestrator()
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
    )
    transcript = Transcript.from_mapping(
        {"segments": [{"speaker": "SPEAKER_01", "text": "segment"}]}
    )

    asyncio.run(stage_orchestrator.run(transcript=transcript, retrieval_enabled=True))

    assert fake_orchestrator.indexed_transcript == transcript
    assert fake_orchestrator.loop_kwargs is not None
    assert fake_orchestrator.loop_kwargs["full_transcript_text"] == ""


def test_run_stage_three_reads_existing_summary_and_triggers_voice_clone(
    tmp_path: Path,
) -> None:
    """Use provided summary XML directly as stage-3 voice-clone input."""
    fake_orchestrator = _FakeOrchestrator()
    fake_voice_clone = _FakeVoiceCloneOrchestrator(output_dir=tmp_path / "voice_clone")
    final_summary_path = tmp_path / "summary.xml"
    final_summary_path.write_text("<SPEAKER_00>hello</SPEAKER_00>", encoding="utf-8")
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(
            start_stage="stage-3",
            final_summary_path=final_summary_path,
        ),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
        fake_voice_clone_orchestrator=fake_voice_clone,
    )

    asyncio.run(
        stage_orchestrator.run(
            transcript={
                "segments": [],
                "metadata": {
                    "speaker_samples_manifest_path": "speaker_samples/manifest.json"
                },
            },
            retrieval_enabled=False,
        )
    )

    assert fake_orchestrator.loop_kwargs is None
    assert (tmp_path / "summary.xml").read_text(encoding="utf-8") == (
        "<SPEAKER_00>hello</SPEAKER_00>\n"
    )
    assert fake_voice_clone.summary_xml == "<SPEAKER_00>hello</SPEAKER_00>"
    assert fake_voice_clone.manifest_path == (
        tmp_path / "speaker_samples" / "manifest.json"
    ).resolve()


def test_run_accepts_legacy_raw_transcript_dict(tmp_path: Path) -> None:
    """Accept raw transcript dictionaries at run ingress for compatibility."""
    fake_orchestrator = _FakeOrchestrator()
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
    )

    asyncio.run(
        stage_orchestrator.run(
            transcript={"segments": [{"speaker": "SPEAKER_00", "text": "legacy"}]},
            retrieval_enabled=True,
        )
    )

    assert fake_orchestrator.indexed_transcript is not None
    assert len(fake_orchestrator.indexed_transcript.segments) == 1
    assert fake_orchestrator.indexed_transcript.segments[0].text == "legacy"


def test_run_stage_two_triggers_voice_clone(tmp_path: Path) -> None:
    """Run stage-3 voice clone after stage-2 when orchestrator is configured."""
    fake_orchestrator = _FakeOrchestrator()
    fake_voice_clone = _FakeVoiceCloneOrchestrator(output_dir=tmp_path / "voice_clone")
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
        fake_voice_clone_orchestrator=fake_voice_clone,
    )
    transcript = {
        "segments": [{"speaker": "SPEAKER_00", "text": "hello world"}],
        "metadata": {"speaker_samples_manifest_path": "speaker_samples/manifest.json"},
    }

    asyncio.run(stage_orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert fake_voice_clone.summary_xml == "<SPEAKER_00>loop</SPEAKER_00>"
    assert fake_voice_clone.manifest_path == (
        tmp_path / "speaker_samples" / "manifest.json"
    ).resolve()


def test_run_stage_two_starts_and_stops_gpu_heartbeat(tmp_path: Path) -> None:
    """Start and stop GPU heartbeat around stage-3 voice-clone execution."""
    fake_orchestrator = _FakeOrchestrator()
    fake_voice_clone = _FakeVoiceCloneOrchestrator(output_dir=tmp_path / "voice_clone")
    fake_gpu_heartbeat = _FakeGpuHeartbeat()
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
        fake_voice_clone_orchestrator=fake_voice_clone,
        fake_gpu_heartbeat=fake_gpu_heartbeat,
    )
    transcript = {
        "segments": [{"speaker": "SPEAKER_00", "text": "hello world"}],
        "metadata": {"speaker_samples_manifest_path": "speaker_samples/manifest.json"},
    }

    asyncio.run(stage_orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert fake_gpu_heartbeat.started_with_pid is not None
    assert fake_gpu_heartbeat.stop_calls == 1
