"""Incremental live voice-clone session used by the stage-2 drafting loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from card_framework.agents.utils import count_words
from card_framework.audio_pipeline.errors import NonRetryableAudioStageError
from card_framework.audio_pipeline.runtime import probe_audio_duration_ms
from card_framework.audio_pipeline.voice_clone_contracts import VoiceCloneArtifact, VoiceCloneProvider
from card_framework.audio_pipeline.voice_clone_orchestrator import (
    VoiceCloneOrchestrator,
    VoiceCloneRunResult,
    _resolve_emo_text,
    _write_json_atomic,
    load_speaker_sample_references,
    merge_audio_artifacts_to_wav,
)


@dataclass(slots=True, frozen=True)
class LiveDraftVoiceCloneTurn:
    """Persist one live-drafted turn and its synthesized audio metadata."""

    turn_id: str
    speaker: str
    text: str
    emo_preset: str
    output_audio_path: Path
    duration_ms: int
    word_count: int
    actual_wpm: float

    @classmethod
    def from_payload(
        cls,
        *,
        payload: Mapping[str, Any],
    ) -> "LiveDraftVoiceCloneTurn":
        """Build one turn record from persisted JSON payload."""
        duration_ms = int(payload.get("duration_ms", 0))
        if duration_ms <= 0:
            raise ValueError("duration_ms must be > 0")
        return cls(
            turn_id=str(payload.get("turn_id", "")).strip(),
            speaker=str(payload.get("speaker", "")).strip(),
            text=str(payload.get("text", "")),
            emo_preset=str(payload.get("emo_preset", "")).strip(),
            output_audio_path=Path(str(payload.get("output_audio_path", ""))).resolve(),
            duration_ms=duration_ms,
            word_count=int(payload.get("word_count", 0)),
            actual_wpm=float(payload.get("actual_wpm", 0.0)),
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize one turn record into JSON-compatible data."""
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "text": self.text,
            "emo_preset": self.emo_preset,
            "output_audio_path": str(self.output_audio_path),
            "duration_ms": self.duration_ms,
            "word_count": self.word_count,
            "actual_wpm": round(self.actual_wpm, 6),
        }


@dataclass(slots=True, frozen=True)
class LiveDraftVoiceCloneState:
    """Persisted live-draft audio state used across critic retries."""

    state_path: Path
    generated_at_utc: str
    speaker_samples_manifest_path: Path
    current_snapshot: tuple[dict[str, Any], ...]
    turn_audio: dict[str, LiveDraftVoiceCloneTurn]

    @classmethod
    def from_payload(
        cls,
        *,
        state_path: Path,
        payload: Mapping[str, Any],
    ) -> "LiveDraftVoiceCloneState":
        """Build persisted state from JSON payload."""
        manifest_path = Path(
            str(payload.get("speaker_samples_manifest_path", ""))
        ).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(
                "Live draft state references a missing speaker-sample manifest: "
                f"{manifest_path}"
            )
        raw_snapshot = payload.get("current_snapshot", [])
        snapshot = tuple(
            {
                "line": int(item.get("line", index)),
                "turn_id": str(item.get("turn_id", "")).strip(),
                "speaker_id": str(item.get("speaker_id", "")).strip(),
                "content": str(item.get("content", "")),
                "emo_preset": str(item.get("emo_preset", "")).strip(),
            }
            for index, item in enumerate(raw_snapshot, start=1)
            if isinstance(item, Mapping)
        )
        raw_turn_audio = payload.get("turn_audio", {})
        turn_audio = {
            str(turn_id): LiveDraftVoiceCloneTurn.from_payload(payload=item)
            for turn_id, item in dict(raw_turn_audio).items()
            if isinstance(item, Mapping)
        }
        return cls(
            state_path=state_path.resolve(),
            generated_at_utc=str(payload.get("generated_at_utc", "")).strip(),
            speaker_samples_manifest_path=manifest_path,
            current_snapshot=snapshot,
            turn_audio=turn_audio,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize the state into a JSON-compatible payload."""
        return {
            "generated_at_utc": self.generated_at_utc,
            "speaker_samples_manifest_path": str(self.speaker_samples_manifest_path),
            "current_snapshot": [dict(item) for item in self.current_snapshot],
            "turn_audio": {
                turn_id: record.to_payload()
                for turn_id, record in self.turn_audio.items()
            },
        }


def load_live_draft_voice_clone_state(state_path: Path) -> LiveDraftVoiceCloneState:
    """Load one persisted live-draft state artifact from disk."""
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return LiveDraftVoiceCloneState.from_payload(
        state_path=state_path,
        payload=payload,
    )


@dataclass(slots=True)
class LiveDraftVoiceCloneSession:
    """Manage incremental per-turn voice cloning for the live drafting loop."""

    provider: VoiceCloneProvider
    state_path: Path
    output_dir: Path
    speaker_samples_manifest_path: Path
    emo_preset_catalog: dict[str, str] | None = None
    manifest_filename: str = "manifest.json"
    merge_segments: bool = True
    merged_output_filename: str = "voice_cloned.wav"
    merge_audio_codec: str = "pcm_s24le"
    merge_timeout_seconds: int = 300
    _state: LiveDraftVoiceCloneState | None = field(default=None, init=False, repr=False)
    _sample_refs: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_orchestrator(
        cls,
        *,
        orchestrator: VoiceCloneOrchestrator,
        state_path: Path,
        speaker_samples_manifest_path: Path,
    ) -> "LiveDraftVoiceCloneSession":
        """Build a live session from the existing batch voice-clone orchestrator."""
        return cls(
            provider=orchestrator.provider,
            state_path=state_path,
            output_dir=orchestrator.output_dir,
            speaker_samples_manifest_path=speaker_samples_manifest_path.resolve(),
            emo_preset_catalog=(
                dict(orchestrator.emo_preset_catalog)
                if orchestrator.emo_preset_catalog
                else None
            ),
            manifest_filename=orchestrator.manifest_filename,
            merge_segments=orchestrator.merge_segments,
            merged_output_filename=orchestrator.merged_output_filename,
            merge_audio_codec=orchestrator.merge_audio_codec,
            merge_timeout_seconds=orchestrator.merge_timeout_seconds,
        )

    def clear(self) -> None:
        """Remove any previously persisted live-draft state and session files."""
        existing_state = self._load_state_if_available()
        tracked_paths = []
        if existing_state is not None:
            tracked_paths.extend(
                record.output_audio_path for record in existing_state.turn_audio.values()
            )
        for path in tracked_paths:
            self._delete_path(path)
        self._delete_path(self.state_path)
        self._delete_path(self._session_turn_dir(), is_dir=True)
        self._state = None

    def restore_snapshot_for_draft(self, draft_xml: str) -> list[dict[str, Any]] | None:
        """Return the persisted registry snapshot when it matches the supplied draft XML."""
        state = self._load_state_if_available()
        if state is None:
            return None
        if state.speaker_samples_manifest_path != self.speaker_samples_manifest_path.resolve():
            return None
        if not _snapshot_matches_summary_xml(state.current_snapshot, draft_xml):
            return None
        for item in state.current_snapshot:
            record = state.turn_audio.get(str(item.get("turn_id", "")).strip())
            if record is None or not record.output_audio_path.exists():
                return None
        return [dict(item) for item in state.current_snapshot]

    def bootstrap_from_snapshot(self, snapshot: Sequence[Mapping[str, Any]]) -> None:
        """Populate audio state for an existing snapshot when no reusable state exists."""
        self.clear()
        for item in snapshot:
            turn_id = str(item.get("turn_id", "")).strip()
            if not turn_id:
                raise NonRetryableAudioStageError(
                    "Cannot bootstrap live draft audio without stable turn_id values."
                )
            record = self.render_turn(
                turn_id=turn_id,
                speaker=str(item.get("speaker_id", "")),
                text=str(item.get("content", "")),
                emo_preset=str(item.get("emo_preset", "")),
            )
            self._state = self._replace_turn_record(record)
        self.sync_snapshot(snapshot)

    def render_turn(
        self,
        *,
        turn_id: str,
        speaker: str,
        text: str,
        emo_preset: str,
    ) -> LiveDraftVoiceCloneTurn:
        """Synthesize or replace one turn-level audio artifact and return its metadata."""
        reference = self._speaker_sample_refs().get(speaker)
        if reference is None:
            raise NonRetryableAudioStageError(
                "Missing speaker sample for live draft speaker "
                f"'{speaker}' in {self.speaker_samples_manifest_path}."
            )
        output_audio_path = self._turn_output_path(turn_id=turn_id)
        temp_output_path = output_audio_path.with_name(f"{output_audio_path.stem}.tmp.wav")
        emo_text = _resolve_emo_text(
            emo_preset=emo_preset,
            emo_preset_catalog=self.emo_preset_catalog,
        )
        rendered_path = self.provider.synthesize(
            reference_audio_path=reference.path,
            text=text,
            output_audio_path=temp_output_path,
            emo_text=emo_text,
        )
        duration_ms = probe_audio_duration_ms(rendered_path)
        if duration_ms is None or duration_ms <= 0:
            self._delete_path(rendered_path)
            raise NonRetryableAudioStageError(
                "Unable to determine live draft duration for synthesized output "
                f"{rendered_path}."
            )
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_path.replace(output_audio_path)
        word_count = count_words(text)
        actual_wpm = (
            round((word_count / (duration_ms / 1000.0)) * 60.0, 6)
            if word_count > 0 and duration_ms > 0
            else 0.0
        )
        record = LiveDraftVoiceCloneTurn(
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            emo_preset=emo_preset.strip() or "neutral",
            output_audio_path=output_audio_path.resolve(),
            duration_ms=int(duration_ms),
            word_count=word_count,
            actual_wpm=actual_wpm,
        )
        self._state = self._replace_turn_record(record)
        return record

    def remove_turn(self, turn_id: str) -> None:
        """Delete one tracked turn artifact and remove it from persisted state."""
        state = self._get_state()
        record = state.turn_audio.get(turn_id)
        if record is None:
            return
        self._delete_path(record.output_audio_path)
        updated_turn_audio = dict(state.turn_audio)
        updated_turn_audio.pop(turn_id, None)
        self._state = LiveDraftVoiceCloneState(
            state_path=self.state_path.resolve(),
            generated_at_utc=_utc_now_iso(),
            speaker_samples_manifest_path=self.speaker_samples_manifest_path.resolve(),
            current_snapshot=state.current_snapshot,
            turn_audio=updated_turn_audio,
        )

    def sync_snapshot(self, snapshot: Sequence[Mapping[str, Any]]) -> None:
        """Persist the ordered registry snapshot that defines the current draft."""
        state = self._get_state()
        normalized_snapshot: list[dict[str, Any]] = []
        for index, item in enumerate(snapshot, start=1):
            turn_id = str(item.get("turn_id", "")).strip()
            if not turn_id:
                raise NonRetryableAudioStageError(
                    "Live draft snapshot entry is missing turn_id."
                )
            if turn_id not in state.turn_audio:
                raise NonRetryableAudioStageError(
                    "Live draft snapshot references a turn without synthesized audio: "
                    f"{turn_id}"
                )
            normalized_snapshot.append(
                {
                    "line": index,
                    "turn_id": turn_id,
                    "speaker_id": str(item.get("speaker_id", "")),
                    "content": str(item.get("content", "")),
                    "emo_preset": str(item.get("emo_preset", "")).strip() or "neutral",
                }
            )
        self._state = LiveDraftVoiceCloneState(
            state_path=self.state_path.resolve(),
            generated_at_utc=_utc_now_iso(),
            speaker_samples_manifest_path=self.speaker_samples_manifest_path.resolve(),
            current_snapshot=tuple(normalized_snapshot),
            turn_audio=dict(state.turn_audio),
        )
        self._persist_state()

    def get_duration_breakdown(
        self,
        *,
        snapshot: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return actual rendered duration totals for the current live draft."""
        active_snapshot = list(snapshot) if snapshot is not None else self.snapshot()
        state = self._get_state()
        messages = []
        per_speaker: dict[str, float] = {}
        total_seconds = 0.0
        for index, item in enumerate(active_snapshot, start=1):
            turn_id = str(item.get("turn_id", "")).strip()
            record = state.turn_audio.get(turn_id)
            if record is None or not record.output_audio_path.exists():
                raise NonRetryableAudioStageError(
                    "Live draft state is missing audio for turn "
                    f"{turn_id or index}."
                )
            actual_seconds = round(record.duration_ms / 1000.0, 3)
            total_seconds += actual_seconds
            messages.append(
                {
                    "line": index,
                    "turn_id": turn_id,
                    "speaker_id": record.speaker,
                    "emo_preset": record.emo_preset,
                    "word_count": record.word_count,
                    "estimated_seconds": actual_seconds,
                    "actual_seconds": actual_seconds,
                    "actual_wpm": round(record.actual_wpm, 6),
                    "duration_ms": record.duration_ms,
                }
            )
            per_speaker[record.speaker] = round(
                per_speaker.get(record.speaker, 0.0) + actual_seconds,
                3,
            )
        rounded_total = round(total_seconds, 3)
        return {
            "duration_source": "actual_audio",
            "total_estimated_seconds": rounded_total,
            "total_actual_seconds": rounded_total,
            "messages": messages,
            "per_speaker_estimated_seconds": per_speaker,
            "per_speaker_actual_seconds": per_speaker,
        }

    def finalize(self) -> VoiceCloneRunResult:
        """Write the final stage-3 manifest and optional merged WAV from live draft state."""
        snapshot = self.snapshot()
        if not snapshot:
            raise NonRetryableAudioStageError(
                "Live draft session cannot finalize an empty snapshot."
            )
        state = self._get_state()
        sample_refs = self._speaker_sample_refs()
        artifacts: list[VoiceCloneArtifact] = []
        manifest_artifacts: list[dict[str, Any]] = []
        for index, item in enumerate(snapshot, start=1):
            turn_id = str(item["turn_id"])
            record = state.turn_audio[turn_id]
            reference = sample_refs.get(record.speaker)
            if reference is None:
                raise NonRetryableAudioStageError(
                    "Missing speaker sample during live draft finalization for "
                    f"speaker '{record.speaker}'."
                )
            artifacts.append(
                VoiceCloneArtifact(
                    turn_index=index,
                    speaker=record.speaker,
                    text=record.text,
                    reference_audio_path=reference.path,
                    output_audio_path=record.output_audio_path,
                    emo_preset=record.emo_preset,
                )
            )
            manifest_artifacts.append(
                {
                    "turn_index": index,
                    "turn_id": record.turn_id,
                    "speaker": record.speaker,
                    "text": record.text,
                    "emo_preset": record.emo_preset,
                    "reference_audio_path": str(reference.path),
                    "output_audio_path": str(record.output_audio_path),
                    "duration_ms": record.duration_ms,
                    "word_count": record.word_count,
                    "actual_wpm": round(record.actual_wpm, 6),
                }
            )
        merged_output_audio_path: Path | None = None
        if self.merge_segments:
            merged_output_audio_path = self.output_dir / self.merged_output_filename
            merge_audio_artifacts_to_wav(
                artifact_paths=[artifact.output_audio_path for artifact in artifacts],
                output_path=merged_output_audio_path,
                audio_codec=self.merge_audio_codec,
                timeout_seconds=self.merge_timeout_seconds,
            )
        generated_at_utc = _utc_now_iso()
        manifest_path = self.output_dir / self.manifest_filename
        _write_json_atomic(
            {
                "generated_at_utc": generated_at_utc,
                "speaker_samples_manifest_path": str(self.speaker_samples_manifest_path),
                "artifact_count": len(artifacts),
                "merge_segments": self.merge_segments,
                "merged_output_audio_path": (
                    str(merged_output_audio_path)
                    if merged_output_audio_path is not None
                    else None
                ),
                "artifacts": manifest_artifacts,
            },
            manifest_path,
        )
        return VoiceCloneRunResult(
            output_dir=self.output_dir,
            manifest_path=manifest_path,
            generated_at_utc=generated_at_utc,
            artifacts=tuple(artifacts),
            merged_output_audio_path=merged_output_audio_path,
        )

    def snapshot(self) -> list[dict[str, Any]]:
        """Return the currently persisted ordered snapshot."""
        return [dict(item) for item in self._get_state().current_snapshot]

    def _speaker_sample_refs(self) -> dict[str, Any]:
        """Return cached speaker-sample references keyed by speaker ID."""
        if self._sample_refs is None:
            self._sample_refs = load_speaker_sample_references(
                self.speaker_samples_manifest_path
            )
        return self._sample_refs

    def _load_state_if_available(self) -> LiveDraftVoiceCloneState | None:
        """Load persisted state when available and valid, otherwise return None."""
        if self._state is not None:
            return self._state
        if not self.state_path.exists():
            return None
        try:
            self._state = load_live_draft_voice_clone_state(self.state_path)
        except Exception:
            self._state = None
        return self._state

    def _get_state(self) -> LiveDraftVoiceCloneState:
        """Return the current state, initializing an empty one when needed."""
        state = self._load_state_if_available()
        if state is not None:
            return state
        self._state = LiveDraftVoiceCloneState(
            state_path=self.state_path.resolve(),
            generated_at_utc=_utc_now_iso(),
            speaker_samples_manifest_path=self.speaker_samples_manifest_path.resolve(),
            current_snapshot=(),
            turn_audio={},
        )
        return self._state

    def _replace_turn_record(
        self,
        record: LiveDraftVoiceCloneTurn,
    ) -> LiveDraftVoiceCloneState:
        """Return updated state after replacing one turn-level record."""
        state = self._get_state()
        updated_turn_audio = dict(state.turn_audio)
        updated_turn_audio[record.turn_id] = record
        return LiveDraftVoiceCloneState(
            state_path=self.state_path.resolve(),
            generated_at_utc=_utc_now_iso(),
            speaker_samples_manifest_path=self.speaker_samples_manifest_path.resolve(),
            current_snapshot=state.current_snapshot,
            turn_audio=updated_turn_audio,
        )

    def _persist_state(self) -> None:
        """Write the current state atomically to the configured sidecar path."""
        state = self._get_state()
        _write_json_atomic(state.to_payload(), self.state_path)

    def _session_turn_dir(self) -> Path:
        """Return the directory used for incremental turn-level live artifacts."""
        return self.output_dir / "live_draft_turns"

    def _turn_output_path(self, *, turn_id: str) -> Path:
        """Return deterministic output path for one stable turn ID."""
        return self._session_turn_dir() / f"{turn_id}.wav"

    @staticmethod
    def _delete_path(path: Path, *, is_dir: bool = False) -> None:
        """Remove one path on a best-effort basis."""
        try:
            if is_dir:
                if path.exists():
                    shutil.rmtree(path)
                return
            if path.exists():
                path.unlink()
        except OSError:
            pass


def _snapshot_matches_summary_xml(
    snapshot: Sequence[Mapping[str, Any]],
    summary_xml: str,
) -> bool:
    """Return whether a persisted snapshot matches the supplied summary XML content."""
    from card_framework.shared.summary_xml import parse_summary_xml

    try:
        turns = parse_summary_xml(summary_xml)
    except ValueError:
        return False
    if len(snapshot) != len(turns):
        return False
    for item, turn in zip(snapshot, turns, strict=True):
        speaker_id = str(item.get("speaker_id", "")).strip()
        content = str(item.get("content", ""))
        emo_preset = str(item.get("emo_preset", "")).strip() or "neutral"
        if (
            speaker_id != turn.speaker
            or content != turn.text
            or emo_preset != turn.emo_preset
        ):
            return False
    return True


def _utc_now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

