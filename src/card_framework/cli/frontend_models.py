"""Shared contracts for the setup-and-run Textual frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

InteractiveMenuMode: TypeAlias = Literal[
    "stage1_single",
    "stage1_batch",
    "stage1_batch_duration_sweep",
    "stage2_transcript",
    "stage3_summary",
    "stage4_interjector",
    "resume_session",
    "setup_only",
]
FrontendAction: TypeAlias = Literal["pipeline", "benchmark", "cancel"]
FrontendUiMode: TypeAlias = Literal["textual", "legacy"]
BenchmarkKind: TypeAlias = Literal[
    "summary_matrix",
    "summarizer_benchmark",
    "qa_benchmark",
    "qmsum_benchmark",
    "sporc_benchmark",
    "sporc_duration_sweep",
    "diarization_benchmark",
]


@dataclass(slots=True, frozen=True)
class InteractiveMenuSelection:
    """Represent one operator launch plan selected in the interactive frontend."""

    mode: InteractiveMenuMode
    audio_paths: tuple[str, ...] = ()
    duration_presets: tuple[str, ...] = ()
    transcript_path: str | None = None
    voiceclone_from_summary: str | None = None
    voice_clone_manifest_path: str | None = None
    resume_session: str | None = None
    output_root: str | None = None
    setup_only: bool = False
    overrides: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class BenchmarkLaunchRequest:
    """Describe one benchmark command that the frontend should launch."""

    kind: BenchmarkKind
    command: tuple[str, ...]
    description: str


@dataclass(slots=True, frozen=True)
class FrontendResult:
    """Return value for the Textual frontend."""

    action: FrontendAction
    pipeline_selection: InteractiveMenuSelection | None = None
    benchmark_request: BenchmarkLaunchRequest | None = None
    ui_mode: FrontendUiMode = "legacy"
