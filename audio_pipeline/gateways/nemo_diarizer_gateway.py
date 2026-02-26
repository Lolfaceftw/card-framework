"""NeMo adapter for speaker diarization."""

from __future__ import annotations

import json
from pathlib import Path

from audio_pipeline.contracts import DiarizationTurn, SpeakerDiarizer
from audio_pipeline.errors import (
    DependencyMissingError,
    NonRetryableAudioStageError,
)
from audio_pipeline.runtime import ensure_module_available, probe_audio_duration_ms


class NemoSpeakerDiarizer(SpeakerDiarizer):
    """
    Run speaker diarization with NeMo, with optional single-speaker fallback.

    NeMo diarization setup varies between environments. This adapter attempts
    programmatic diarization first; when unavailable and fallback is enabled,
    a single-speaker timeline is returned so the pipeline remains runnable.
    """

    def __init__(
        self,
        *,
        msdd_model: str = "diar_msdd_telephonic",
        vad_model: str = "vad_multilingual_marblenet",
        speaker_embedding_model: str = "titanet_large",
        max_speakers: int | None = None,
        min_speakers: int | None = None,
        allow_single_speaker_fallback: bool = True,
    ) -> None:
        self.msdd_model = msdd_model
        self.vad_model = vad_model
        self.speaker_embedding_model = speaker_embedding_model
        self.max_speakers = max_speakers
        self.min_speakers = min_speakers
        self.allow_single_speaker_fallback = allow_single_speaker_fallback

    def diarize(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        device: str,
    ) -> list[DiarizationTurn]:
        """
        Run diarization and return speaker turns.

        Args:
            audio_path: Input audio path.
            output_dir: Workspace directory for NeMo artifacts.
            device: Runtime device (``cpu`` or ``cuda``).

        Returns:
            Ordered speaker turns.
        """
        if not audio_path.exists():
            raise NonRetryableAudioStageError(f"Audio file does not exist: {audio_path}")

        try:
            turns = self._run_nemo(audio_path=audio_path, output_dir=output_dir, device=device)
            if turns:
                return turns
            raise NonRetryableAudioStageError("NeMo diarization returned zero RTTM turns.")
        except Exception as exc:
            if not self.allow_single_speaker_fallback:
                if isinstance(exc, NonRetryableAudioStageError):
                    raise
                raise NonRetryableAudioStageError("NeMo diarization failed.") from exc
            fallback_end = max(1, self._probe_duration_ms(audio_path))
            return [
                DiarizationTurn(
                    speaker="SPEAKER_00",
                    start_time_ms=0,
                    end_time_ms=fallback_end,
                )
            ]

    def _run_nemo(
        self,
        *,
        audio_path: Path,
        output_dir: Path,
        device: str,
    ) -> list[DiarizationTurn]:
        """Execute NeMo clustering diarizer and parse RTTM output."""
        ensure_module_available("nemo")
        ensure_module_available("omegaconf")
        ensure_module_available("yaml")

        try:
            from nemo.collections.asr.models import ClusteringDiarizer
            from omegaconf import OmegaConf
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise DependencyMissingError(
                "Failed to import NeMo diarization dependencies."
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "diarization_manifest.json"
        manifest_payload = {
            "audio_filepath": str(audio_path),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        cfg = OmegaConf.create(
            {
                "num_workers": 0,
                "diarizer": {
                    "manifest_filepath": str(manifest_path),
                    "out_dir": str(output_dir),
                    "speaker_embeddings": {
                        "model_path": self.speaker_embedding_model,
                    },
                    "vad": {
                        "model_path": self.vad_model,
                    },
                    "msdd_model": {
                        "model_path": self.msdd_model,
                    },
                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers": False,
                        }
                    },
                },
            }
        )

        parameters = cfg.diarizer.clustering.parameters
        if self.max_speakers is not None:
            parameters.max_num_speakers = int(self.max_speakers)
        if self.min_speakers is not None:
            parameters.min_num_speakers = int(self.min_speakers)
        if device == "cpu":
            cfg.diarizer.device = "cpu"

        diarizer = ClusteringDiarizer(cfg=cfg)
        diarizer.diarize()

        rttm_path = self._find_rttm(output_dir=output_dir, audio_path=audio_path)
        turns = self._parse_rttm(rttm_path)
        return self._normalize_speaker_labels(turns)

    def _find_rttm(self, *, output_dir: Path, audio_path: Path) -> Path:
        """Resolve RTTM file produced by NeMo."""
        candidates = [
            output_dir / "pred_rttms" / f"{audio_path.stem}.rttm",
            output_dir / f"{audio_path.stem}.rttm",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        fallback = next(output_dir.rglob("*.rttm"), None)
        if fallback is not None:
            return fallback
        raise NonRetryableAudioStageError(
            f"NeMo diarization did not emit RTTM under '{output_dir}'."
        )

    def _parse_rttm(self, rttm_path: Path) -> list[DiarizationTurn]:
        """Parse RTTM into diarization turns."""
        turns: list[DiarizationTurn] = []
        for line in rttm_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start_seconds = float(parts[3])
            duration_seconds = float(parts[4])
            end_seconds = start_seconds + duration_seconds
            speaker = parts[7]
            turns.append(
                DiarizationTurn(
                    speaker=speaker,
                    start_time_ms=max(0, int(round(start_seconds * 1000))),
                    end_time_ms=max(0, int(round(end_seconds * 1000))),
                )
            )
        turns.sort(key=lambda turn: turn.start_time_ms)
        return turns

    def _normalize_speaker_labels(
        self, turns: list[DiarizationTurn]
    ) -> list[DiarizationTurn]:
        """Normalize arbitrary speaker labels into SPEAKER_XX."""
        mapping: dict[str, str] = {}
        normalized: list[DiarizationTurn] = []
        next_index = 0
        for turn in turns:
            if turn.speaker not in mapping:
                mapping[turn.speaker] = f"SPEAKER_{next_index:02d}"
                next_index += 1
            normalized.append(
                DiarizationTurn(
                    speaker=mapping[turn.speaker],
                    start_time_ms=turn.start_time_ms,
                    end_time_ms=turn.end_time_ms,
                )
            )
        return normalized

    def _probe_duration_ms(self, audio_path: Path) -> int:
        """Best-effort duration probe via ffprobe."""
        duration_ms = probe_audio_duration_ms(audio_path)
        if duration_ms is None:
            return 24 * 60 * 60 * 1000
        return max(1, duration_ms)
