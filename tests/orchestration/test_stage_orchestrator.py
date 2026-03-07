"""Tests for stage-level orchestration flow control."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from audio_pipeline.eta import LinearStageEtaStrategy, StageSpeedProfile
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
    manifest_path: Path
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
        return _FakeVoiceCloneResult(
            output_dir=self.output_dir,
            manifest_path=self.output_dir / "manifest.json",
            artifacts=("turn.wav",),
        )


@dataclass(slots=True)
class _FakeInterjectorResult:
    output_dir: Path
    artifacts: tuple[str, ...]


class _FakeInterjectorOrchestrator:
    """Capture Stage-4 interjector invocations without external dependencies."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.summary_xml: str | None = None
        self.voice_clone_manifest_path: Path | None = None

    def run(
        self,
        *,
        summary_xml: str,
        voice_clone_manifest_path: Path,
        language: str = "en",
    ) -> _FakeInterjectorResult:
        del language
        self.summary_xml = summary_xml
        self.voice_clone_manifest_path = voice_clone_manifest_path
        return _FakeInterjectorResult(
            output_dir=self.output_dir,
            artifacts=("overlay.wav",),
        )


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
    fake_interjector_orchestrator: _FakeInterjectorOrchestrator | None = None,
    fake_gpu_heartbeat: _FakeGpuHeartbeat | None = None,
) -> StageOrchestrator:
    """Create a stage-orchestrator with stable defaults for tests."""
    return StageOrchestrator(
        orchestrator=fake_orchestrator,  # type: ignore[arg-type]
        stage_plan=stage_plan,
        project_root=project_root,
        target_seconds=60,
        duration_tolerance_ratio=0.05,
        max_iterations=3,
        voice_clone_orchestrator=fake_voice_clone_orchestrator,  # type: ignore[arg-type]
        interjector_orchestrator=fake_interjector_orchestrator,  # type: ignore[arg-type]
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


def test_run_stage_two_passes_scoped_loop_memory_context(tmp_path: Path) -> None:
    """Forward a transcript-scoped loop-memory artifact path into the loop call."""
    fake_orchestrator = _FakeOrchestrator()
    artifact_path = tmp_path / "loop_memory.json"
    stage_orchestrator = StageOrchestrator(
        orchestrator=fake_orchestrator,  # type: ignore[arg-type]
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        project_root=tmp_path,
        target_seconds=60,
        duration_tolerance_ratio=0.05,
        max_iterations=3,
        loop_memory_artifact_path=artifact_path,
    )

    asyncio.run(
        stage_orchestrator.run(
            transcript={"segments": [{"speaker": "SPEAKER_00", "text": "hello"}]},
            retrieval_enabled=False,
        )
    )

    assert fake_orchestrator.loop_kwargs is not None
    assert fake_orchestrator.loop_kwargs["loop_memory_artifact_path"] == artifact_path
    loop_memory_context = fake_orchestrator.loop_kwargs["loop_memory_context"]
    assert loop_memory_context["target_seconds"] == "60"
    assert loop_memory_context["duration_tolerance_ratio"] == "0.050000"
    assert loop_memory_context["transcript_sha256"]


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


def test_run_stage_two_voice_clone_hides_first_run_eta_and_persists_learning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Keep first-run voice-clone ETA silent while still persisting learned throughput."""
    fake_orchestrator = _FakeOrchestrator()
    fake_voice_clone = _FakeVoiceCloneOrchestrator(output_dir=tmp_path / "voice_clone")
    messages: list[str] = []

    def _capture(event_type: str, *args: Any, **kwargs: Any) -> None:
        del event_type, kwargs
        if args:
            messages.append(str(args[-1]))

    monkeypatch.setattr("orchestration.stage_orchestrator.event_bus.publish", _capture)

    profile_path = tmp_path / "eta_profile.json"
    stage_orchestrator = StageOrchestrator(
        orchestrator=fake_orchestrator,  # type: ignore[arg-type]
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        project_root=tmp_path,
        target_seconds=60,
        duration_tolerance_ratio=0.05,
        max_iterations=3,
        voice_clone_orchestrator=fake_voice_clone,  # type: ignore[arg-type]
        eta_strategy=LinearStageEtaStrategy(
            separation=StageSpeedProfile(cpu=1.0, cuda=1.0),
            transcription=StageSpeedProfile(cpu=1.0, cuda=1.0),
            diarization=StageSpeedProfile(cpu=1.0, cuda=1.0),
        ),
        eta_profile_path=profile_path,
        eta_profile_context={"device": "cpu"},
        eta_update_interval_seconds=0.0,
    )
    transcript = {
        "segments": [{"speaker": "SPEAKER_00", "text": "hello world"}],
        "metadata": {"speaker_samples_manifest_path": "speaker_samples/manifest.json"},
    }

    asyncio.run(stage_orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert not any("estimated time left" in message for message in messages)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert payload["unit_stages"]["voice_clone"]["samples"] == 1


def test_run_stage_two_triggers_interjector_after_voice_clone(tmp_path: Path) -> None:
    """Run Stage-4 interjection after voice clone when configured."""
    fake_orchestrator = _FakeOrchestrator()
    fake_voice_clone = _FakeVoiceCloneOrchestrator(output_dir=tmp_path / "voice_clone")
    fake_interjector = _FakeInterjectorOrchestrator(output_dir=tmp_path / "interjector")
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
        fake_voice_clone_orchestrator=fake_voice_clone,
        fake_interjector_orchestrator=fake_interjector,
    )
    transcript = {
        "segments": [{"speaker": "SPEAKER_00", "text": "hello world"}],
        "metadata": {"speaker_samples_manifest_path": "speaker_samples/manifest.json"},
    }

    asyncio.run(stage_orchestrator.run(transcript=transcript, retrieval_enabled=False))

    assert fake_voice_clone.summary_xml == "<SPEAKER_00>loop</SPEAKER_00>"
    assert fake_interjector.summary_xml == "<SPEAKER_00>loop</SPEAKER_00>"
    assert fake_interjector.voice_clone_manifest_path == (
        tmp_path / "voice_clone" / "manifest.json"
    )


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


def test_run_stage_two_prepares_deferred_speaker_samples_for_voice_clone(
    tmp_path: Path,
) -> None:
    """Generate speaker samples on demand when manifest creation is deferred."""
    fake_orchestrator = _FakeOrchestrator()
    fake_voice_clone = _FakeVoiceCloneOrchestrator(output_dir=tmp_path / "voice_clone")
    prepared_calls: list[str] = []
    manifest_path = tmp_path / "speaker_samples" / "manifest.json"

    def _prepare_speaker_samples(transcript: Transcript) -> Transcript:
        prepared_calls.append("called")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text('{"samples": []}', encoding="utf-8")
        return transcript.with_metadata(
            {
                **transcript.metadata,
                "speaker_samples_manifest_path": str(manifest_path),
            }
        )

    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(start_stage="stage-2"),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
        fake_voice_clone_orchestrator=fake_voice_clone,
    )
    stage_orchestrator = StageOrchestrator(
        orchestrator=stage_orchestrator.orchestrator,
        stage_plan=stage_orchestrator.stage_plan,
        project_root=stage_orchestrator.project_root,
        target_seconds=stage_orchestrator.target_seconds,
        duration_tolerance_ratio=stage_orchestrator.duration_tolerance_ratio,
        max_iterations=stage_orchestrator.max_iterations,
        voice_clone_orchestrator=stage_orchestrator.voice_clone_orchestrator,
        speaker_sample_preparer=_prepare_speaker_samples,
    )

    asyncio.run(
        stage_orchestrator.run(
            transcript={"segments": [{"speaker": "SPEAKER_00", "text": "hello world"}]},
            retrieval_enabled=False,
        )
    )

    assert prepared_calls == ["called"]
    assert fake_voice_clone.manifest_path == manifest_path.resolve()


def test_run_stage_four_uses_existing_summary_and_manifest(tmp_path: Path) -> None:
    """Run direct Stage-4 interjection from existing summary and stage-3 manifest."""
    fake_orchestrator = _FakeOrchestrator()
    fake_interjector = _FakeInterjectorOrchestrator(output_dir=tmp_path / "interjector")
    final_summary_path = tmp_path / "summary.xml"
    voice_clone_manifest_path = tmp_path / "voice_clone" / "manifest.json"
    final_summary_path.write_text("<SPEAKER_00>hello</SPEAKER_00>", encoding="utf-8")
    voice_clone_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    voice_clone_manifest_path.write_text('{"artifacts": []}', encoding="utf-8")
    stage_orchestrator = _build_stage_orchestrator(
        stage_plan=PipelineStagePlan(
            start_stage="stage-4",
            final_summary_path=final_summary_path,
            voice_clone_manifest_path=voice_clone_manifest_path,
        ),
        fake_orchestrator=fake_orchestrator,
        project_root=tmp_path,
        fake_interjector_orchestrator=fake_interjector,
    )

    asyncio.run(
        stage_orchestrator.run(
            transcript={"segments": [], "metadata": {}},
            retrieval_enabled=False,
        )
    )

    assert fake_orchestrator.loop_kwargs is None
    assert fake_interjector.summary_xml == "<SPEAKER_00>hello</SPEAKER_00>"
    assert fake_interjector.voice_clone_manifest_path == voice_clone_manifest_path
