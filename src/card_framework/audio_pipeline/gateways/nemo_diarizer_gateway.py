"""NeMo MSDD adapter for speaker diarization."""

from __future__ import annotations

import json
from pathlib import Path
import signal
import sys
from typing import Any

from card_framework.audio_pipeline.contracts import DiarizationTurn, SpeakerDiarizer
from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from card_framework.audio_pipeline.errors import DependencyMissingError, NonRetryableAudioStageError
from card_framework.audio_pipeline.gateways.diarization_common import (
    normalize_speaker_labels,
    parse_rttm_file,
    prepare_diarization_audio,
)
from card_framework.audio_pipeline.runtime import (
    ensure_module_available,
    probe_audio_duration_ms,
    release_torch_device_memory_with_reporting,
)
from card_framework.shared.events import event_bus
from card_framework.shared.perf import report_loading_duration


class NemoSpeakerDiarizer(SpeakerDiarizer):
    """
    Run speaker diarization with NeMo MSDD, with optional single-speaker fallback.

    This implementation mirrors the legacy Whisper + NeMo MSDD flow used in the
    stage-3 pipeline while preserving the adapter shape of ``audio_pipeline``.
    """

    def __init__(
        self,
        *,
        msdd_model: str = "diar_msdd_telephonic",
        vad_model: str = "vad_multilingual_marblenet",
        speaker_embedding_model: str = "titanet_large",
        max_speakers: int | None = 10,
        min_speakers: int | None = None,
        allow_single_speaker_fallback: bool = False,
        inference_batch_size: int = 8,
        cuda_long_audio_guard_enabled: bool = False,
        cuda_max_audio_duration_seconds: int | None = 3600,
        long_audio_fallback_device: str = "cpu",
        vad_onset: float = 0.8,
        vad_offset: float = 0.6,
        vad_pad_onset: float = 0.1,
        vad_pad_offset: float = 0.0,
        vad_min_duration_on: float = 0.2,
        vad_min_duration_off: float = 0.6,
        clustering_max_rp_threshold: float = 0.4,
        clustering_sparse_search_volume: int = 60,
        clustering_maj_vote_spk_count: bool = True,
    ) -> None:
        self.msdd_model = msdd_model
        self.vad_model = vad_model
        self.speaker_embedding_model = speaker_embedding_model
        if (
            min_speakers is not None
            and max_speakers is not None
            and int(min_speakers) > int(max_speakers)
        ):
            raise ValueError("min_speakers must be <= max_speakers.")
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be > 0.")
        if (
            cuda_max_audio_duration_seconds is not None
            and cuda_max_audio_duration_seconds <= 0
        ):
            raise ValueError(
                "cuda_max_audio_duration_seconds must be > 0 when configured."
            )
        normalized_fallback_device = long_audio_fallback_device.strip().lower()
        if normalized_fallback_device not in {"cpu", "cuda"}:
            raise ValueError(
                "long_audio_fallback_device must be one of: cpu, cuda."
            )
        self.max_speakers = max_speakers
        self.min_speakers = min_speakers
        self.allow_single_speaker_fallback = allow_single_speaker_fallback
        self.inference_batch_size = inference_batch_size
        self.cuda_long_audio_guard_enabled = cuda_long_audio_guard_enabled
        self.cuda_max_audio_duration_seconds = cuda_max_audio_duration_seconds
        self.long_audio_fallback_device = normalized_fallback_device
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset
        self.vad_pad_onset = vad_pad_onset
        self.vad_pad_offset = vad_pad_offset
        self.vad_min_duration_on = vad_min_duration_on
        self.vad_min_duration_off = vad_min_duration_off
        self.clustering_max_rp_threshold = clustering_max_rp_threshold
        self.clustering_sparse_search_volume = clustering_sparse_search_volume
        self.clustering_maj_vote_spk_count = clustering_maj_vote_spk_count

    def diarize(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        device: str,
        progress_callback: StageProgressCallback | None = None,
    ) -> list[DiarizationTurn]:
        """
        Run diarization and return speaker turns.

        Args:
            audio_path: Input audio path.
            output_dir: Workspace directory for NeMo artifacts.
            device: Runtime device (``cpu`` or ``cuda``).
            progress_callback: Optional callback for progress updates.

        Returns:
            Ordered speaker turns.
        """
        if not audio_path.exists():
            raise NonRetryableAudioStageError(f"Audio file does not exist: {audio_path}")
        if progress_callback is not None:
            try:
                progress_callback(
                    StageProgressUpdate(
                        completed_units=0,
                        total_units=1,
                        note="nemo diarization started",
                    )
                )
            except Exception:
                pass

        completed = False
        try:
            with report_loading_duration(
                f"NeMo diarization model '{self.msdd_model}'",
                reporter=self._report_system_message,
            ):
                turns = self._run_nemo_msdd(
                    audio_path=audio_path,
                    output_dir=output_dir,
                    device=device,
                )
            if turns:
                completed = True
                return turns
            raise NonRetryableAudioStageError("NeMo diarization returned zero RTTM turns.")
        except Exception as exc:
            if not self.allow_single_speaker_fallback:
                if isinstance(exc, NonRetryableAudioStageError):
                    raise
                raise NonRetryableAudioStageError("NeMo diarization failed.") from exc
            fallback_end = max(1, self._probe_duration_ms(audio_path))
            completed = True
            return [
                DiarizationTurn(
                    speaker="SPEAKER_00",
                    start_time_ms=0,
                    end_time_ms=fallback_end,
                )
            ]
        finally:
            if completed and progress_callback is not None:
                try:
                    progress_callback(
                        StageProgressUpdate(
                            completed_units=1,
                            total_units=1,
                            note="nemo diarization completed",
                        )
                    )
                except Exception:
                    pass

    def _run_nemo_msdd(
        self,
        *,
        audio_path: Path,
        output_dir: Path,
        device: str,
    ) -> list[DiarizationTurn]:
        """Execute NeMo MSDD diarizer and parse RTTM output."""
        ensure_module_available("nemo")
        ensure_module_available("omegaconf")
        ensure_module_available("yaml")

        try:
            # NeMo uses SIGKILL in places; patch on Windows for compatibility.
            if sys.platform == "win32" and not hasattr(signal, "SIGKILL"):
                setattr(signal, "SIGKILL", signal.SIGTERM)

            from nemo.collections.asr.models.msdd_models import NeuralDiarizer  # type: ignore[import-not-found]
            from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels  # type: ignore[import-not-found]
            from omegaconf import OmegaConf
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise DependencyMissingError(
                "Failed to import NeMo MSDD diarization dependencies."
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        diarization_audio_path = self._prepare_diarization_audio(
            audio_path=audio_path,
            output_dir=output_dir,
        )
        runtime_device = self._resolve_runtime_device(
            audio_path=diarization_audio_path,
            requested_device=device,
        )
        manifest_path = output_dir / "diarization_manifest.json"
        manifest_payload = self._build_manifest_payload(diarization_audio_path)
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

        cfg = self._build_msdd_config(
            manifest_path=manifest_path,
            output_dir=output_dir,
            omega_conf=OmegaConf,
            device=runtime_device,
        )

        diarizer = NeuralDiarizer(cfg=cfg).to(
            "cuda" if runtime_device == "cuda" else "cpu"
        )
        exact_speaker_count = self._exact_speaker_count()
        diarizer._initialize_configs(
            manifest_path=str(manifest_path),
            max_speakers=int(self.max_speakers or 8),
            num_speakers=exact_speaker_count,
            tmpdir=str(output_dir),
            batch_size=self.inference_batch_size,
            num_workers=0,
            verbose=False,
        )
        diarizer.clustering_embedding.clus_diar_model._diarizer_params.out_dir = str(output_dir)
        diarizer.clustering_embedding.clus_diar_model._diarizer_params.manifest_filepath = str(
            manifest_path
        )
        diarizer.msdd_model.cfg.test_ds.manifest_filepath = str(manifest_path)
        diarizer.diarize()

        rttm_path = self._find_rttm(output_dir=output_dir, audio_path=diarization_audio_path)
        try:
            raw_labels = rttm_to_labels(str(rttm_path))
        except Exception:
            return normalize_speaker_labels(parse_rttm_file(rttm_path))

        turns: list[DiarizationTurn] = []
        for raw_label in raw_labels:
            parts = raw_label.split()
            if len(parts) != 3:
                continue
            start_seconds = float(parts[0])
            end_seconds = float(parts[1])
            speaker_id = str(parts[2]).split("_")[-1]
            turns.append(
                DiarizationTurn(
                    speaker=f"SPEAKER_{int(speaker_id):02d}",
                    start_time_ms=max(0, int(round(start_seconds * 1000))),
                    end_time_ms=max(0, int(round(end_seconds * 1000))),
                )
            )
        turns.sort(key=lambda turn: turn.start_time_ms)
        if turns:
            return turns
        return normalize_speaker_labels(parse_rttm_file(rttm_path))

    def _resolve_runtime_device(
        self,
        *,
        audio_path: Path,
        requested_device: str,
    ) -> str:
        """Downgrade long-form NeMo CUDA runs before they reach the driver-reset path."""
        normalized_requested_device = requested_device.strip().lower()
        if normalized_requested_device != "cuda":
            return normalized_requested_device
        if not self.cuda_long_audio_guard_enabled:
            return normalized_requested_device
        max_audio_duration_seconds = self.cuda_max_audio_duration_seconds
        if max_audio_duration_seconds is None:
            return normalized_requested_device
        audio_duration_ms = probe_audio_duration_ms(audio_path)
        if audio_duration_ms is None:
            return normalized_requested_device
        if audio_duration_ms <= max_audio_duration_seconds * 1000:
            return normalized_requested_device
        fallback_device = self.long_audio_fallback_device
        if fallback_device == normalized_requested_device:
            return normalized_requested_device
        self._report_system_message(
            "Switching NeMo diarization to "
            f"{fallback_device.upper()} for {audio_duration_ms / 60000.0:.1f}-minute "
            "audio because it exceeds "
            "audio.diarization.nemo_cuda_max_audio_duration_seconds="
            f"{max_audio_duration_seconds}; avoiding the long-form CUDA watchdog path."
        )
        return fallback_device

    def _exact_speaker_count(self) -> int | None:
        """Return an exact speaker count only when both bounds match."""
        if self.min_speakers is None or self.max_speakers is None:
            return None
        if int(self.min_speakers) != int(self.max_speakers):
            return None
        return int(self.min_speakers)

    def _build_manifest_payload(self, diarization_audio_path: Path) -> dict[str, Any]:
        """Build the NeMo manifest payload for one diarization audio artifact."""
        return {
            "audio_filepath": str(diarization_audio_path),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": self._exact_speaker_count(),
            "rttm_filepath": None,
            "uem_filepath": None,
        }

    def _build_msdd_config(
        self,
        *,
        manifest_path: Path,
        output_dir: Path,
        omega_conf: Any,
        device: str,
    ):
        """Build NeMo MSDD config from the packaged telephonic template."""
        config_template_path = Path(__file__).with_name("nemo_msdd_infer_telephonic.yaml")
        try:
            cfg = omega_conf.load(str(config_template_path))
        except Exception as exc:
            raise NonRetryableAudioStageError(
                f"Failed to load NeMo MSDD config template at '{config_template_path}'."
            ) from exc

        cfg.num_workers = 0
        cfg.batch_size = self.inference_batch_size
        cfg.sample_rate = 16000
        cfg.verbose = False
        cfg.device = "cuda" if device == "cuda" else "cpu"
        cfg.diarizer.manifest_filepath = str(manifest_path)
        cfg.diarizer.out_dir = str(output_dir)
        cfg.diarizer.oracle_vad = False
        cfg.diarizer.ignore_overlap = False

        cfg.diarizer.vad.model_path = self.vad_model
        cfg.diarizer.vad.external_vad_manifest = None
        vad_parameters = cfg.diarizer.vad.parameters
        vad_parameters.onset = float(self.vad_onset)
        vad_parameters.offset = float(self.vad_offset)
        vad_parameters.pad_onset = float(self.vad_pad_onset)
        vad_parameters.pad_offset = float(self.vad_pad_offset)
        vad_parameters.min_duration_on = float(self.vad_min_duration_on)
        vad_parameters.min_duration_off = float(self.vad_min_duration_off)

        cfg.diarizer.speaker_embeddings.model_path = self.speaker_embedding_model
        cfg.diarizer.msdd_model.model_path = self.msdd_model

        clustering_parameters = cfg.diarizer.clustering.parameters
        clustering_parameters.oracle_num_speakers = self._exact_speaker_count() is not None
        clustering_parameters.max_rp_threshold = float(self.clustering_max_rp_threshold)
        clustering_parameters.sparse_search_volume = int(self.clustering_sparse_search_volume)
        clustering_parameters.maj_vote_spk_count = bool(self.clustering_maj_vote_spk_count)
        if self.max_speakers is not None:
            clustering_parameters.max_num_speakers = int(self.max_speakers)
        if self.min_speakers is not None and "min_num_speakers" in clustering_parameters:
            clustering_parameters.min_num_speakers = int(self.min_speakers)
        return cfg

    def _prepare_diarization_audio(self, *, audio_path: Path, output_dir: Path) -> Path:
        """Normalize diarization input to mono 16kHz WAV via ffmpeg."""
        return prepare_diarization_audio(audio_path=audio_path, output_dir=output_dir)

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

    def _probe_duration_ms(self, audio_path: Path) -> int:
        """Best-effort duration probe via ffprobe."""
        duration_ms = probe_audio_duration_ms(audio_path)
        if duration_ms is None:
            return 24 * 60 * 60 * 1000
        return max(1, duration_ms)

    def release_cached_resources(self, *, device: str) -> None:
        """Release any process-local CUDA allocator slack after diarization."""
        release_torch_device_memory_with_reporting(
            device=device,
            description=f"NeMo diarization model '{self.msdd_model}'",
        )

    def _report_system_message(self, message: str) -> None:
        """Publish one operator-facing loading message through the event bus."""
        event_bus.publish("system_message", message)

