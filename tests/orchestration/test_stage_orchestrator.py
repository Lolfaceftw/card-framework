"""Tests for stage-level orchestration flow control."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestration.stage_orchestrator import StageOrchestrator
from orchestration.transcript import Transcript
from pipeline_plan import PipelineStagePlan


@dataclass(slots=True)
class _FakeVerdict:
    """Minimal critic verdict used by stage-orchestrator tests."""

    status: str
    word_count: int
    feedback: str


class _FakeOrchestrator:
    """Capture stage-orchestrator calls without external dependencies."""

    def __init__(self) -> None:
        self.indexed_transcript: Transcript | None = None
        self.summarizer_kwargs: dict[str, Any] | None = None
        self.loop_kwargs: dict[str, Any] | None = None
        self.critic_kwargs: dict[str, Any] | None = None

    async def index_transcript(self, transcript: Transcript) -> int:
        self.indexed_transcript = transcript
        return 1

    async def run_summarizer_once(self, **kwargs: Any) -> str:
        self.summarizer_kwargs = kwargs
        return "<SPEAKER_00>summary</SPEAKER_00>"

    async def run_loop(self, **kwargs: Any) -> str:
        self.loop_kwargs = kwargs
        return "<SPEAKER_00>loop</SPEAKER_00>"

    async def run_critic_once(self, **kwargs: Any) -> _FakeVerdict:
        self.critic_kwargs = kwargs
        return _FakeVerdict(status="pass", word_count=42, feedback="ok")


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
    ) -> _FakeVoiceCloneResult:
        self.summary_xml = summary_xml
        self.manifest_path = speaker_samples_manifest_path
        return _FakeVoiceCloneResult(output_dir=self.output_dir, artifacts=("turn.wav",))


def _build_stage_orchestrator(
    *,
    stage_plan: PipelineStagePlan,
    fake_orchestrator: _FakeOrchestrator,
    project_root: Path,
    fake_voice_clone_orchestrator: _FakeVoiceCloneOrchestrator | None = None,
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
    )


def test_run_summarizer_stage_uses_full_transcript_when_retrieval_disabled(
    tmp_path: Path,
) -> None:
    """Build full transcript text for summarizer stage when retrieval is disabled."""
    fake_orchestrator = _FakeOrchestrator()
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="transcript", stop_stage="summarizer"),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
    )
    transcript = {"segments": [{"speaker": "SPEAKER_00", "text": "hello world"}]}

    asyncio.run(stage_orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert fake_orchestrator.indexed_transcript is None
    assert fake_orchestrator.summarizer_kwargs is not None
    assert (
        fake_orchestrator.summarizer_kwargs["full_transcript_text"]
        == "[SPEAKER_00]: hello world\n"
    )


def test_run_full_loop_indexes_when_retrieval_enabled(tmp_path: Path) -> None:
    """Index transcript before loop when retrieval tools are available."""
    fake_orchestrator = _FakeOrchestrator()
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="transcript", stop_stage="critic"),
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


def test_run_draft_stage_reads_existing_draft(tmp_path: Path) -> None:
    """Forward draft content to critic-only stage in draft start mode."""
    fake_orchestrator = _FakeOrchestrator()
    draft_path = tmp_path / "draft.xml"
    draft_path.write_text("<SPEAKER_00>draft</SPEAKER_00>", encoding="utf-8")
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(
            start_stage="draft",
            stop_stage="critic",
            draft_path=draft_path,
        ),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
    )

    asyncio.run(
        stage_orchestrator.run(
            transcript={"segments": []},
            retrieval_enabled=False,
        )
    )

    assert fake_orchestrator.critic_kwargs is not None
    assert fake_orchestrator.critic_kwargs["draft"] == "<SPEAKER_00>draft</SPEAKER_00>"


def test_run_accepts_legacy_raw_transcript_dict(tmp_path: Path) -> None:
    """Accept raw transcript dictionaries at run ingress for compatibility."""
    fake_orchestrator = _FakeOrchestrator()
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="transcript", stop_stage="critic"),
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


def test_run_summarizer_stage_triggers_voice_clone(tmp_path: Path) -> None:
    """Run voice clone stage after summarizer when voice-clone orchestrator is configured."""
    fake_orchestrator = _FakeOrchestrator()
    fake_voice_clone = _FakeVoiceCloneOrchestrator(output_dir=tmp_path / "voice_clone")
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="transcript", stop_stage="summarizer"),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
        fake_voice_clone_orchestrator=fake_voice_clone,
    )
    transcript = {
        "segments": [{"speaker": "SPEAKER_00", "text": "hello world"}],
        "metadata": {"speaker_samples_manifest_path": "speaker_samples/manifest.json"},
    }

    asyncio.run(stage_orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert fake_voice_clone.summary_xml == "<SPEAKER_00>summary</SPEAKER_00>"
    assert fake_voice_clone.manifest_path == (
        tmp_path / "speaker_samples" / "manifest.json"
    ).resolve()
