"""IndexTTS-backed voice-cloning gateway."""

from __future__ import annotations

import atexit
from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

from card_framework.audio_pipeline.emotion.emotion_vectors import EmotionVector
from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from card_framework.audio_pipeline.errors import (
    DependencyMissingError,
    NonRetryableAudioStageError,
)
from card_framework.audio_pipeline.runtime import (
    release_torch_device_memory_with_reporting,
)
from card_framework.audio_pipeline.voice_clone_contracts import VoiceCloneProvider
from card_framework.shared.events import event_bus

_INDEXTTS_PROTOCOL_PREFIX = "__INDEXTTS_JSON__"
_INDEXTTS_READY_TIMEOUT_SECONDS = 600.0
_INDEXTTS_PROTOCOL_POLL_SECONDS = 0.2
_INDEXTTS_LOG_TAIL_LINE_LIMIT = 80
_INDEXTTS_CUDA_STARTUP_MIN_FREE_MEMORY_MB = 6000
_INDEXTTS_CUDA_VRAM_BUDGET_MB = 8192
_INDEXTTS_CUDA_STARTUP_EXPECTED_WORKER_DELTA_MB = 6600
_INDEXTTS_CUDA_STARTUP_WAIT_SECONDS = 90.0
_INDEXTTS_CUDA_STARTUP_POLL_SECONDS = 5.0
_INDEXTTS_CUDA_ADAPTIVE_RESERVE_FREE_MEMORY_MB = 1024
_INDEXTTS_CUDA_ADAPTIVE_LOW_HEADROOM_MB = 3072
_INDEXTTS_CUDA_ADAPTIVE_TIGHT_HEADROOM_MB = 2048
_INDEXTTS_CUDA_ADAPTIVE_BALANCED_TEXT_TOKENS = 80
_INDEXTTS_CUDA_ADAPTIVE_TIGHT_TEXT_TOKENS = 60
_INDEXTTS_CUDA_ADAPTIVE_BALANCED_MEL_TOKENS = 1200
_INDEXTTS_CUDA_ADAPTIVE_TIGHT_MEL_TOKENS = 900
_CONSOLE_SAFE_TRANSLATIONS = str.maketrans(
    {
        "\u2192": "->",
        "\u2014": "-",
        "\u2013": "-",
        "\u2026": "...",
    }
)


@dataclass(slots=True, frozen=True)
class _IndexTTSRuntimeKey:
    """Identify one shared IndexTTS runtime configuration."""

    execution_backend: str
    cfg_path: Path
    model_dir: Path
    device: str
    use_fp16: bool
    use_cuda_kernel: bool
    use_deepspeed: bool
    use_accel: bool
    use_torch_compile: bool
    runner_project_dir: Path | None
    uv_executable: str
    stream_subprocess_output: bool


@dataclass(slots=True, frozen=True)
class _CudaMemorySnapshot:
    """Describe one coarse CUDA memory snapshot from nvidia-smi."""

    free_mb: int
    total_mb: int
    used_mb: int


@dataclass(slots=True, frozen=True)
class _CudaStartupBudget:
    """Describe the adaptive CUDA startup budget for one telemetry snapshot."""

    effective_budget_mb: int
    expected_delta_mb: int
    projected_total_mb: int
    projected_free_after_startup_mb: int
    reserve_free_memory_mb: int


_SHARED_INPROCESS_MODELS: dict[_IndexTTSRuntimeKey, object] = {}
_SHARED_SUBPROCESS_WORKERS: dict[
    _IndexTTSRuntimeKey, "_PersistentIndexTTSSubprocessWorker"
] = {}
_SHARED_INPROCESS_EMO_VECTOR_CACHE: dict[
    _IndexTTSRuntimeKey,
    dict[str, tuple[float, ...]],
] = {}
_SHARED_RUNTIME_LOCK = threading.Lock()


class _PersistentIndexTTSSubprocessWorker:
    """Keep one nested IndexTTS2 subprocess warm for repeated syntheses."""

    def __init__(
        self,
        *,
        cfg_path: Path,
        model_dir: Path,
        device: str,
        use_fp16: bool,
        use_cuda_kernel: bool,
        use_deepspeed: bool,
        use_accel: bool,
        use_torch_compile: bool,
        runner_project_dir: Path,
        uv_executable: str,
        stream_output: bool,
    ) -> None:
        self._device = str(device).strip().lower() or "cpu"
        self._stream_output = stream_output
        self._response_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._log_tail: deque[str] = deque(maxlen=_INDEXTTS_LOG_TAIL_LINE_LIMIT)
        self._request_lock = threading.Lock()
        self._request_counter = 0
        self._closed = False

        runner_script = _resolve_persistent_runner_script_path()
        command = _build_persistent_worker_command(
            uv_executable=uv_executable,
            runner_project_dir=runner_project_dir,
            runner_script=runner_script,
            cfg_path=cfg_path,
            model_dir=model_dir,
            device=device,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed,
            use_accel=use_accel,
            use_torch_compile=use_torch_compile,
        )
        environment = _build_persistent_worker_env(
            source_root=_resolve_repo_source_root()
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="IndexTTSPersistentWorkerReader",
            daemon=True,
        )
        self._reader_thread.start()
        try:
            ready_payload = self._wait_for_protocol_message(
                expected_type="ready",
                timeout_seconds=_INDEXTTS_READY_TIMEOUT_SECONDS,
            )
            if not bool(ready_payload.get("ok", False)):
                detail = self._format_failure_detail(
                    str(ready_payload.get("error", "")).strip()
                )
                raise NonRetryableAudioStageError(
                    f"IndexTTS subprocess worker failed to initialize: {detail}"
                )
        except Exception:
            self.close()
            raise

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None,
        emotion_vector: EmotionVector | None,
        verbose: bool,
        max_text_tokens_per_segment: int,
        generation_kwargs: dict[str, Any],
    ) -> Path:
        """Run one synthesis request on the warm worker process."""
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        with self._request_lock:
            if self._closed:
                raise NonRetryableAudioStageError(
                    "IndexTTS subprocess worker is already closed."
                )
            if self._process.poll() is not None:
                detail = self._format_failure_detail("")
                raise NonRetryableAudioStageError(
                    "IndexTTS subprocess worker exited unexpectedly before synthesis: "
                    f"{detail}"
                )
            request_id = f"req-{self._request_counter}"
            self._request_counter += 1
            self._send_payload(
                {
                    "action": "synthesize",
                    "request_id": request_id,
                    "reference_audio_path": str(reference_audio_path),
                    "text": text,
                    "output_audio_path": str(output_audio_path),
                    "emo_text": emo_text,
                    "emotion_vector": (
                        emotion_vector.to_list() if emotion_vector is not None else None
                    ),
                    "verbose": verbose,
                    "max_text_tokens_per_segment": max_text_tokens_per_segment,
                    "generation_kwargs": generation_kwargs,
                }
            )
            response = self._wait_for_protocol_message(
                expected_type="result",
                request_id=request_id,
            )
        if not bool(response.get("ok", False)):
            detail = self._format_failure_detail(str(response.get("error", "")).strip())
            raise NonRetryableAudioStageError(
                f"IndexTTS subprocess synthesis failed: {detail}"
            )
        return output_audio_path

    @property
    def device(self) -> str:
        """Return the actual worker device."""
        return self._device

    def close(self) -> None:
        """Terminate the worker process and free its cached model state."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.poll() is None and process.stdin is not None:
            try:
                self._send_payload(
                    {
                        "action": "shutdown",
                        "request_id": "shutdown",
                    }
                )
            except Exception:
                pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        self._reader_thread.join(timeout=1.0)

    def is_alive(self) -> bool:
        """Return whether the worker process is still ready for requests."""
        return not self._closed and self._process.poll() is None

    def _send_payload(self, payload: dict[str, Any]) -> None:
        """Write one JSON request payload to the worker stdin."""
        stdin = self._process.stdin
        if stdin is None:
            raise NonRetryableAudioStageError(
                "IndexTTS subprocess worker stdin is unavailable."
            )
        try:
            stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stdin.flush()
        except OSError as exc:
            detail = self._format_failure_detail(str(exc))
            raise NonRetryableAudioStageError(
                f"Failed to communicate with IndexTTS subprocess worker: {detail}"
            ) from exc

    def _wait_for_protocol_message(
        self,
        *,
        expected_type: str,
        request_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Wait for one protocol message, surfacing worker exits as rich errors."""
        started_at = time.monotonic()
        while True:
            timeout = _INDEXTTS_PROTOCOL_POLL_SECONDS
            if timeout_seconds is not None:
                remaining = timeout_seconds - (time.monotonic() - started_at)
                if remaining <= 0:
                    detail = self._format_failure_detail("")
                    raise NonRetryableAudioStageError(
                        "Timed out waiting for IndexTTS subprocess worker response: "
                        f"{detail}"
                    )
                timeout = min(timeout, remaining)
            try:
                payload = self._response_queue.get(timeout=timeout)
            except queue.Empty:
                if self._process.poll() is not None:
                    detail = self._format_failure_detail("")
                    raise NonRetryableAudioStageError(
                        f"IndexTTS subprocess worker exited unexpectedly: {detail}"
                    )
                continue
            payload_type = str(payload.get("type", "")).strip()
            if payload_type == "process_exit":
                detail = self._format_failure_detail("")
                raise NonRetryableAudioStageError(
                    f"IndexTTS subprocess worker exited unexpectedly: {detail}"
                )
            if payload_type != expected_type:
                continue
            if (
                request_id is not None
                and str(payload.get("request_id", "")) != request_id
            ):
                continue
            return payload

    def _reader_loop(self) -> None:
        """Drain merged worker stdout and split protocol lines from normal logs."""
        stdout = self._process.stdout
        if stdout is None:
            self._response_queue.put(
                {
                    "type": "process_exit",
                    "returncode": self._process.poll(),
                }
            )
            return
        try:
            for raw_line in stdout:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                if line.startswith(_INDEXTTS_PROTOCOL_PREFIX):
                    payload_text = line.removeprefix(_INDEXTTS_PROTOCOL_PREFIX)
                    try:
                        payload = json.loads(payload_text)
                    except json.JSONDecodeError:
                        self._record_log_line(line)
                        continue
                    if isinstance(payload, dict):
                        self._response_queue.put(payload)
                        continue
                self._record_log_line(line)
        finally:
            self._response_queue.put(
                {
                    "type": "process_exit",
                    "returncode": self._process.poll(),
                }
            )

    def _record_log_line(self, line: str) -> None:
        """Capture worker logs and optionally stream them to the parent terminal."""
        normalized = line.strip()
        if not normalized:
            return
        self._log_tail.append(normalized)
        if self._stream_output:
            sanitized = normalized.translate(_CONSOLE_SAFE_TRANSLATIONS)
            encoding = getattr(sys.stdout, "encoding", None)
            if encoding:
                sanitized = sanitized.encode(encoding, errors="replace").decode(
                    encoding
                )
            print(
                sanitized,
                flush=True,
            )

    def _format_failure_detail(self, error_message: str) -> str:
        """Build a compact operator-facing detail string for worker failures."""
        normalized_error = error_message.strip()
        log_tail = "\n".join(self._log_tail).strip()
        streamed_suffix = "\nSee streamed subprocess output above for full context."
        if self._stream_output:
            if normalized_error and log_tail and normalized_error not in log_tail:
                return f"{normalized_error}\nRecent subprocess output:\n{log_tail}{streamed_suffix}"
            if log_tail:
                return f"Recent subprocess output:\n{log_tail}{streamed_suffix}"
            if normalized_error:
                return f"{normalized_error}{streamed_suffix}"
            return streamed_suffix.lstrip()
        if normalized_error and log_tail and normalized_error not in log_tail:
            return f"{normalized_error}\n{log_tail}"
        if normalized_error:
            return normalized_error
        if log_tail:
            return log_tail
        return "No subprocess output captured."


def _build_persistent_worker_command(
    *,
    uv_executable: str,
    runner_project_dir: Path,
    runner_script: Path,
    cfg_path: Path,
    model_dir: Path,
    device: str,
    use_fp16: bool,
    use_cuda_kernel: bool,
    use_deepspeed: bool,
    use_accel: bool,
    use_torch_compile: bool,
) -> list[str]:
    """Build the nested ``uv run`` command for the warm IndexTTS worker."""
    command = [
        uv_executable,
        "run",
        "--project",
        str(runner_project_dir),
        "--python",
        sys.executable,
        "python",
        str(runner_script),
        "--cfg-path",
        str(cfg_path),
        "--model-dir",
        str(model_dir),
        "--device",
        device,
    ]
    if use_fp16:
        command.append("--use-fp16")
    if use_cuda_kernel:
        command.append("--use-cuda-kernel")
    if use_deepspeed:
        command.append("--use-deepspeed")
    if use_accel:
        command.append("--use-accel")
    if use_torch_compile:
        command.append("--use-torch-compile")
    return command


def _build_persistent_worker_env(
    *,
    source_root: Path,
    existing_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Ensure nested workers can import the repo-local ``card_framework`` package."""
    environment = dict(existing_env or os.environ)
    environment.pop("VIRTUAL_ENV", None)
    source_root_text = str(source_root.resolve())
    current_entries = [
        entry for entry in environment.get("PYTHONPATH", "").split(os.pathsep) if entry
    ]
    current_entries = [entry for entry in current_entries if entry != source_root_text]
    environment["PYTHONPATH"] = os.pathsep.join([source_root_text, *current_entries])
    return environment


def _resolve_repo_source_root() -> Path:
    """Return the repository ``src`` directory for subprocess import wiring."""
    return Path(__file__).resolve().parents[3]


def _resolve_persistent_runner_script_path() -> Path:
    """Return the repository-local persistent worker runner script path."""
    runner_script = (
        Path(__file__).resolve().parent.parent
        / "runners"
        / "indextts_persistent_runner.py"
    )
    if not runner_script.exists():
        raise NonRetryableAudioStageError(
            f"IndexTTS persistent runner script does not exist: {runner_script}"
        )
    return runner_script


def _validate_model_artifact_paths(*, cfg_path: Path, model_dir: Path) -> None:
    """Validate IndexTTS configuration and checkpoint paths."""
    if not cfg_path.exists():
        raise NonRetryableAudioStageError(
            f"IndexTTS config file does not exist: {cfg_path}"
        )
    if not model_dir.exists() or not model_dir.is_dir():
        raise NonRetryableAudioStageError(
            f"IndexTTS model directory does not exist: {model_dir}"
        )


def _validate_runner_project_dir(*, runner_project_dir: Path | None) -> Path:
    """Validate the nested ``uv`` project directory used by subprocess backend."""
    if runner_project_dir is None:
        raise NonRetryableAudioStageError(
            "IndexTTS subprocess backend requires runner_project_dir."
        )
    resolved_project_dir = runner_project_dir.resolve()
    if not resolved_project_dir.exists() or not resolved_project_dir.is_dir():
        raise NonRetryableAudioStageError(
            f"IndexTTS runner project directory does not exist: {resolved_project_dir}"
        )
    return resolved_project_dir


def _normalize_emo_text(emo_text: str | None) -> str | None:
    """Normalize optional emotion-guidance text into a cacheable value."""
    if emo_text is None:
        return None
    normalized = emo_text.strip()
    return normalized or None


def _looks_retryable_subprocess_crash(error: Exception) -> bool:
    """Return whether one subprocess failure should retry on CPU once."""
    lowered = str(error).strip().lower()
    return (
        "worker exited unexpectedly" in lowered
        or "timed out waiting for indextts subprocess worker response" in lowered
    )


def _format_one_line_error(error: Exception) -> str:
    """Collapse one exception into a compact single-line detail."""
    parts = [line.strip() for line in str(error).splitlines() if line.strip()]
    if not parts:
        return error.__class__.__name__
    return " ".join(parts)


def _query_cuda_memory_snapshot() -> _CudaMemorySnapshot | None:
    """Return coarse free/used VRAM data from nvidia-smi when available."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            free_mb = int(parts[0])
            total_mb = int(parts[1])
            used_mb = int(parts[2])
        except ValueError:
            continue
        if total_mb <= 0 or free_mb < 0 or used_mb < 0:
            continue
        return _CudaMemorySnapshot(
            free_mb=free_mb,
            total_mb=total_mb,
            used_mb=used_mb,
        )
    return None


def _release_inprocess_model(
    model: object,
    *,
    device: str,
    description: str,
) -> None:
    """Best-effort release of one cached in-process IndexTTS runtime."""
    del model
    gc.collect()
    release_torch_device_memory_with_reporting(
        device=device,
        description=description,
    )
    try:
        import torch
    except ImportError:
        return
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _release_cached_runtime_for_key(key: _IndexTTSRuntimeKey) -> None:
    """Release one cached runtime entry for a gateway configuration key."""
    with _SHARED_RUNTIME_LOCK:
        model = _SHARED_INPROCESS_MODELS.pop(key, None)
        worker = _SHARED_SUBPROCESS_WORKERS.pop(key, None)
    if worker is not None:
        worker.close()
    if model is not None:
        _release_inprocess_model(
            model,
            device=key.device,
            description=(
                f"IndexTTS in-process runtime '{key.model_dir.name}' "
                f"(backend={key.execution_backend})"
            ),
        )
    with _SHARED_RUNTIME_LOCK:
        _SHARED_INPROCESS_EMO_VECTOR_CACHE.pop(key, None)


def _release_all_cached_runtime_resources() -> None:
    """Release all cached IndexTTS runtimes held by this Python process."""
    with _SHARED_RUNTIME_LOCK:
        models = list(_SHARED_INPROCESS_MODELS.items())
        workers = list(_SHARED_SUBPROCESS_WORKERS.values())
        _SHARED_INPROCESS_MODELS.clear()
        _SHARED_SUBPROCESS_WORKERS.clear()
        _SHARED_INPROCESS_EMO_VECTOR_CACHE.clear()
    for worker in workers:
        worker.close()
    for key, model in models:
        _release_inprocess_model(
            model,
            device=key.device,
            description=(
                f"IndexTTS in-process runtime '{key.model_dir.name}' "
                f"(backend={key.execution_backend})"
            ),
        )


atexit.register(_release_all_cached_runtime_resources)


@dataclass(slots=True)
class IndexTTSVoiceCloneGateway(VoiceCloneProvider):
    """
    Voice-cloning provider backed by ``indextts.infer_v2.IndexTTS2``.

    The in-process backend shares one loaded model per matching config inside the
    current Python process. The default subprocess backend keeps a persistent
    nested worker alive so repeated syntheses do not reload IndexTTS2 for every
    turn. Call ``close()`` only when the wider voice-clone module is being
    offloaded and you intentionally want to reclaim that warm runtime.

    Args:
        cfg_path: IndexTTS config YAML path.
        model_dir: IndexTTS model/checkpoints directory.
        device: Runtime device label (for example, ``cpu`` or ``cuda``).
        use_fp16: Enable FP16 inference when supported.
        use_cuda_kernel: Enable BigVGAN CUDA kernels.
        use_deepspeed: Enable DeepSpeed acceleration.
        use_accel: Enable IndexTTS acceleration engine.
        use_torch_compile: Enable ``torch.compile`` optimization.
        verbose: Enable verbose provider output.
        max_text_tokens_per_segment: Segment budget for long text synthesis.
        do_sample: Whether GPT mel-token generation should sample.
        top_p: Nucleus-sampling threshold for GPT generation.
        top_k: Top-k sampling threshold for GPT generation.
        temperature: Sampling temperature for GPT generation.
        length_penalty: Length penalty used during mel-token generation.
        num_beams: Beam width used during mel-token generation.
        repetition_penalty: Repetition penalty for GPT generation.
        max_mel_tokens: Max generated mel-token budget per segment.
        execution_backend: ``inprocess`` or ``subprocess`` execution mode.
        runner_project_dir: ``uv`` project dir containing IndexTTS dependencies.
        uv_executable: Name/path of the ``uv`` executable.
        stream_subprocess_output: Stream nested subprocess logs to terminal.
    """

    cfg_path: Path
    model_dir: Path
    device: str = "cpu"
    use_fp16: bool = False
    use_cuda_kernel: bool = False
    use_deepspeed: bool = False
    use_accel: bool = False
    use_torch_compile: bool = False
    verbose: bool = False
    max_text_tokens_per_segment: int = 120
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 0.8
    length_penalty: float = 0.0
    num_beams: int = 1
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 1500
    execution_backend: str = "subprocess"
    runner_project_dir: Path | None = None
    uv_executable: str = "uv"
    stream_subprocess_output: bool = True
    normalize_emotion_vector: bool = True
    emotion_vector_apply_bias: bool = True
    emotion_vector_max_sum: float = 0.8
    cuda_startup_min_free_memory_mb: int = _INDEXTTS_CUDA_STARTUP_MIN_FREE_MEMORY_MB
    cuda_vram_budget_mb: int = _INDEXTTS_CUDA_VRAM_BUDGET_MB
    cuda_startup_expected_worker_delta_mb: int = (
        _INDEXTTS_CUDA_STARTUP_EXPECTED_WORKER_DELTA_MB
    )
    cuda_startup_wait_seconds: float = _INDEXTTS_CUDA_STARTUP_WAIT_SECONDS
    cuda_startup_poll_seconds: float = _INDEXTTS_CUDA_STARTUP_POLL_SECONDS
    cuda_adaptive_memory_enabled: bool = True
    cuda_adaptive_reserve_free_memory_mb: int = (
        _INDEXTTS_CUDA_ADAPTIVE_RESERVE_FREE_MEMORY_MB
    )
    cuda_adaptive_low_headroom_mb: int = _INDEXTTS_CUDA_ADAPTIVE_LOW_HEADROOM_MB
    cuda_adaptive_tight_headroom_mb: int = _INDEXTTS_CUDA_ADAPTIVE_TIGHT_HEADROOM_MB
    cuda_adaptive_min_text_tokens_per_segment: int = (
        _INDEXTTS_CUDA_ADAPTIVE_TIGHT_TEXT_TOKENS
    )
    cuda_adaptive_min_mel_tokens: int = _INDEXTTS_CUDA_ADAPTIVE_TIGHT_MEL_TOKENS

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None = None,
        emotion_vector: EmotionVector | None = None,
        progress_callback: StageProgressCallback | None = None,
    ) -> Path:
        """
        Synthesize one WAV artifact from text and speaker reference audio.

        Args:
            reference_audio_path: Speaker reference sample WAV path.
            text: Text payload to synthesize.
            output_audio_path: Target WAV path.
            progress_callback: Optional callback for progress updates.

        Returns:
            Path to generated WAV artifact.

        Raises:
            NonRetryableAudioStageError: If synthesis fails or output is missing.
            DependencyMissingError: If IndexTTS is unavailable.
        """
        if not text.strip():
            raise NonRetryableAudioStageError("Voice clone text must be non-empty.")
        if not reference_audio_path.exists():
            raise NonRetryableAudioStageError(
                f"Reference audio path does not exist: {reference_audio_path}"
            )
        backend = self.execution_backend.strip().lower()
        if backend == "inprocess":
            rendered_path = self._synthesize_inprocess(
                reference_audio_path=reference_audio_path,
                text=text.strip(),
                output_audio_path=output_audio_path,
                emo_text=emo_text,
                emotion_vector=emotion_vector,
            )
        elif backend == "subprocess":
            rendered_path = self._synthesize_subprocess(
                reference_audio_path=reference_audio_path,
                text=text.strip(),
                output_audio_path=output_audio_path,
                emo_text=emo_text,
                emotion_vector=emotion_vector,
            )
        else:
            raise NonRetryableAudioStageError(
                "Unsupported IndexTTS execution backend: "
                f"{self.execution_backend!r}. Expected 'inprocess' or 'subprocess'."
            )
        if progress_callback is not None:
            try:
                progress_callback(
                    StageProgressUpdate(
                        note="indextts synthesis completed",
                    )
                )
            except Exception:
                pass
        return rendered_path

    def close(self) -> None:
        """Release the shared runtime used by this gateway configuration."""
        _release_cached_runtime_for_key(self._runtime_key())

    @classmethod
    def release_all_cached_resources(cls) -> None:
        """Release all shared IndexTTS runtimes cached by this process."""
        del cls
        _release_all_cached_runtime_resources()

    def _synthesize_inprocess(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None,
        emotion_vector: EmotionVector | None,
    ) -> Path:
        """Run synthesis by importing IndexTTS2 in current Python process."""
        tts_model = self._get_or_init_model()
        emo_vector = self._resolve_direct_or_cached_emo_vector(
            tts_model=tts_model,
            emotion_vector=emotion_vector,
            emo_text=emo_text,
        )
        normalized_emo_text = _normalize_emo_text(emo_text)
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result: Any = tts_model.infer(
                spk_audio_prompt=str(reference_audio_path),
                text=text,
                output_path=str(output_audio_path),
                verbose=self.verbose,
                max_text_tokens_per_segment=self.max_text_tokens_per_segment,
                use_emo_text=emo_vector is None and normalized_emo_text is not None,
                emo_text=normalized_emo_text if emo_vector is None else None,
                emo_vector=emo_vector,
                **self._generation_kwargs(),
            )
            del result
        except Exception as exc:
            raise NonRetryableAudioStageError(
                f"IndexTTS synthesis failed for output {output_audio_path}."
            ) from exc

        if not output_audio_path.exists():
            raise NonRetryableAudioStageError(
                f"IndexTTS synthesis completed without output file: {output_audio_path}"
            )
        return output_audio_path

    def _synthesize_subprocess(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None,
        emotion_vector: EmotionVector | None,
    ) -> Path:
        """Run synthesis via a persistent ``uv run --project`` worker."""
        worker = self._get_or_init_subprocess_worker()
        normalized_emotion_vector = self._normalize_provider_emotion_vector(
            emotion_vector
        )
        try:
            rendered_path = worker.synthesize(
                reference_audio_path=reference_audio_path,
                text=text,
                output_audio_path=output_audio_path,
                emo_text=emo_text,
                emotion_vector=normalized_emotion_vector,
                verbose=self.verbose,
                max_text_tokens_per_segment=self.max_text_tokens_per_segment,
                generation_kwargs=self._generation_kwargs(),
            )
        except NonRetryableAudioStageError as exc:
            if not self._should_retry_subprocess_on_cuda(
                attempted_device=worker.device,
                error=exc,
            ):
                raise
            self._evict_cached_subprocess_worker(expected_worker=worker)
            self._publish_subprocess_cuda_retry(
                attempted_device=worker.device,
                error=exc,
            )
            self._prepare_cuda_worker_retry()
            worker = self._get_or_init_subprocess_worker(
                device_override=worker.device,
                allow_cuda_retry=False,
            )
            rendered_path = worker.synthesize(
                reference_audio_path=reference_audio_path,
                text=text,
                output_audio_path=output_audio_path,
                emo_text=emo_text,
                emotion_vector=normalized_emotion_vector,
                verbose=self.verbose,
                max_text_tokens_per_segment=self.max_text_tokens_per_segment,
                generation_kwargs=self._generation_kwargs(),
            )
        if not rendered_path.exists():
            raise NonRetryableAudioStageError(
                f"IndexTTS subprocess completed without output file: {rendered_path}"
            )
        return rendered_path

    def _get_or_init_model(self) -> object:
        """Lazily initialize and share one in-process IndexTTS2 runtime."""
        cfg_path = self.cfg_path.resolve()
        model_dir = self.model_dir.resolve()
        _validate_model_artifact_paths(cfg_path=cfg_path, model_dir=model_dir)
        key = self._runtime_key()
        with _SHARED_RUNTIME_LOCK:
            cached_model = _SHARED_INPROCESS_MODELS.get(key)
            if cached_model is not None:
                return cached_model

        self._apply_cuda_adaptive_profile_from_telemetry(device=self.device)
        key = self._runtime_key()
        with _SHARED_RUNTIME_LOCK:
            cached_model = _SHARED_INPROCESS_MODELS.get(key)
            if cached_model is not None:
                return cached_model
            try:
                from indextts.infer_v2 import IndexTTS2
            except ImportError as exc:
                raise DependencyMissingError(
                    "IndexTTS2 is not installed. Install index-tts to enable voice cloning."
                ) from exc
            try:
                cached_model = IndexTTS2(
                    cfg_path=str(cfg_path),
                    model_dir=str(model_dir),
                    use_fp16=self.use_fp16,
                    device=self.device,
                    use_cuda_kernel=self.use_cuda_kernel,
                    use_deepspeed=self.use_deepspeed,
                    use_accel=self.use_accel,
                    use_torch_compile=self.use_torch_compile,
                )
            except Exception as exc:
                raise NonRetryableAudioStageError(
                    "Failed to initialize IndexTTS2 with configured model artifacts."
                ) from exc
            _SHARED_INPROCESS_MODELS[key] = cached_model
            return cached_model

    def _generation_kwargs(self) -> dict[str, Any]:
        """Build mel-generation settings forwarded to IndexTTS inference."""
        generation_kwargs: dict[str, Any] = {
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "max_mel_tokens": self.max_mel_tokens,
        }
        if self.num_beams > 1:
            generation_kwargs["length_penalty"] = self.length_penalty
        return generation_kwargs

    def _get_or_resolve_emo_vector(
        self,
        *,
        tts_model: object,
        emo_text: str | None,
    ) -> list[float] | None:
        """Cache repeated emotion-text analysis for the warm in-process runtime."""
        normalized_emo_text = _normalize_emo_text(emo_text)
        if normalized_emo_text is None:
            return None
        key = self._runtime_key()
        with _SHARED_RUNTIME_LOCK:
            cached_vectors = _SHARED_INPROCESS_EMO_VECTOR_CACHE.setdefault(key, {})
            cached_vector = cached_vectors.get(normalized_emo_text)
        if cached_vector is not None:
            return list(cached_vector)
        qwen_emo = getattr(tts_model, "qwen_emo", None)
        if qwen_emo is None or not hasattr(qwen_emo, "inference"):
            return None
        emo_dict = qwen_emo.inference(normalized_emo_text)
        emo_vector = tuple(float(value) for value in emo_dict.values())
        with _SHARED_RUNTIME_LOCK:
            cached_vectors = _SHARED_INPROCESS_EMO_VECTOR_CACHE.setdefault(key, {})
            cached_vectors.setdefault(normalized_emo_text, emo_vector)
            resolved_vector = cached_vectors[normalized_emo_text]
        return list(resolved_vector)

    def _resolve_direct_or_cached_emo_vector(
        self,
        *,
        tts_model: object,
        emotion_vector: EmotionVector | None,
        emo_text: str | None,
    ) -> list[float] | None:
        """Prefer direct vectors and fall back to cached emotion-text analysis."""
        direct_vector = self._normalize_provider_emotion_vector(emotion_vector)
        if direct_vector is not None:
            return direct_vector.to_list()
        return self._get_or_resolve_emo_vector(tts_model=tts_model, emo_text=emo_text)

    def _normalize_provider_emotion_vector(
        self,
        emotion_vector: EmotionVector | None,
    ) -> EmotionVector | None:
        """Normalize one direct vector before sending it to IndexTTS."""
        if emotion_vector is None:
            return None
        if not self.normalize_emotion_vector:
            return emotion_vector
        return emotion_vector.with_index_tts_normalization(
            apply_bias=self.emotion_vector_apply_bias,
            max_sum=self.emotion_vector_max_sum,
        )

    def _get_or_init_subprocess_worker(
        self,
        *,
        device_override: str | None = None,
        allow_cuda_retry: bool = True,
    ) -> _PersistentIndexTTSSubprocessWorker:
        """Lazily initialize and share one warm subprocess worker per config."""
        cfg_path = self.cfg_path.resolve()
        model_dir = self.model_dir.resolve()
        _validate_model_artifact_paths(cfg_path=cfg_path, model_dir=model_dir)
        runner_project_dir = _validate_runner_project_dir(
            runner_project_dir=self.runner_project_dir
        )
        requested_device = str(device_override or self.device).strip().lower() or "cpu"
        key = self._runtime_key()
        with _SHARED_RUNTIME_LOCK:
            cached_worker = _SHARED_SUBPROCESS_WORKERS.get(key)
            if (
                cached_worker is not None
                and cached_worker.is_alive()
                and (
                    device_override is None or cached_worker.device == requested_device
                )
            ):
                return cached_worker
            if cached_worker is not None:
                cached_worker.close()
                _SHARED_SUBPROCESS_WORKERS.pop(key, None)

        self._apply_cuda_adaptive_profile_from_telemetry(device=requested_device)
        key = self._runtime_key()
        with _SHARED_RUNTIME_LOCK:
            cached_worker = _SHARED_SUBPROCESS_WORKERS.get(key)
            if (
                cached_worker is not None
                and cached_worker.is_alive()
                and (
                    device_override is None or cached_worker.device == requested_device
                )
            ):
                return cached_worker
            if cached_worker is not None:
                cached_worker.close()
                _SHARED_SUBPROCESS_WORKERS.pop(key, None)
            try:
                cached_worker = self._build_subprocess_worker(
                    cfg_path=cfg_path,
                    model_dir=model_dir,
                    runner_project_dir=runner_project_dir,
                    device=requested_device,
                )
            except NonRetryableAudioStageError as exc:
                if not allow_cuda_retry or not self._should_retry_subprocess_on_cuda(
                    attempted_device=requested_device,
                    error=exc,
                ):
                    raise
                self._publish_subprocess_cuda_retry(
                    attempted_device=requested_device,
                    error=exc,
                )
                self._prepare_cuda_worker_retry()
                cached_worker = self._build_subprocess_worker(
                    cfg_path=cfg_path,
                    model_dir=model_dir,
                    runner_project_dir=runner_project_dir,
                    device=requested_device,
                )
            _SHARED_SUBPROCESS_WORKERS[key] = cached_worker
            return cached_worker

    def _build_subprocess_worker(
        self,
        *,
        cfg_path: Path,
        model_dir: Path,
        runner_project_dir: Path,
        device: str,
    ) -> _PersistentIndexTTSSubprocessWorker:
        """Construct one warm worker using the requested device profile."""
        self._wait_for_cuda_worker_headroom(device=device)
        return _PersistentIndexTTSSubprocessWorker(
            cfg_path=cfg_path,
            model_dir=model_dir,
            device=device,
            use_fp16=self.use_fp16,
            use_cuda_kernel=self.use_cuda_kernel,
            use_deepspeed=self.use_deepspeed,
            use_accel=self.use_accel,
            use_torch_compile=self.use_torch_compile,
            runner_project_dir=runner_project_dir,
            uv_executable=self.uv_executable,
            stream_output=self.stream_subprocess_output,
        )

    def _should_retry_subprocess_on_cuda(
        self,
        *,
        attempted_device: str,
        error: Exception,
    ) -> bool:
        """Retry one unexpected CUDA worker crash on the same device."""
        return (
            attempted_device.strip().lower() == "cuda"
            and _looks_retryable_subprocess_crash(error)
        )

    def _publish_subprocess_cuda_retry(
        self,
        *,
        attempted_device: str,
        error: Exception,
    ) -> None:
        """Tell operators when the warm worker is retried on CUDA."""
        event_bus.publish(
            "system_message",
            (
                "IndexTTS subprocess worker failed on "
                f"device='{attempted_device}'. Clearing local CUDA state and "
                f"retrying once on CUDA. Reason: {_format_one_line_error(error)}"
            ),
        )

    def _prepare_cuda_worker_retry(self) -> None:
        """Best-effort cleanup before retrying a crashed CUDA worker."""
        release_torch_device_memory_with_reporting(
            device="cuda",
            description="IndexTTS subprocess retry preparation",
        )

    def _apply_cuda_adaptive_profile_from_telemetry(self, *, device: str) -> None:
        """Reduce CUDA generation pressure when current VRAM headroom is tight."""
        if device.strip().lower() != "cuda" or not self.cuda_adaptive_memory_enabled:
            return
        snapshot = _query_cuda_memory_snapshot()
        if snapshot is None:
            return
        budget = self._resolve_cuda_startup_budget(snapshot=snapshot)
        pressure_detected = budget.projected_total_mb > int(
            max(min(self.cuda_vram_budget_mb, snapshot.total_mb), 0)
        ) or budget.projected_free_after_startup_mb < int(
            max(self.cuda_adaptive_low_headroom_mb, 0)
        )
        if not pressure_detected:
            return

        configured_budget_mb = int(
            max(min(self.cuda_vram_budget_mb, snapshot.total_mb), 0)
        )
        changes: list[str] = []
        if self.num_beams > 1:
            changes.append(f"num_beams {self.num_beams}->1")
            self.num_beams = 1
        if self.use_torch_compile:
            changes.append("use_torch_compile true->false")
            self.use_torch_compile = False
        if self.use_accel:
            changes.append("use_accel true->false")
            self.use_accel = False
        if self.use_deepspeed:
            changes.append("use_deepspeed true->false")
            self.use_deepspeed = False

        if budget.projected_free_after_startup_mb < int(
            max(self.cuda_adaptive_tight_headroom_mb, 0)
        ):
            target_text_tokens = int(
                max(self.cuda_adaptive_min_text_tokens_per_segment, 1)
            )
            target_mel_tokens = int(max(self.cuda_adaptive_min_mel_tokens, 1))
        else:
            target_text_tokens = max(
                int(max(self.cuda_adaptive_min_text_tokens_per_segment, 1)),
                _INDEXTTS_CUDA_ADAPTIVE_BALANCED_TEXT_TOKENS,
            )
            target_mel_tokens = max(
                int(max(self.cuda_adaptive_min_mel_tokens, 1)),
                _INDEXTTS_CUDA_ADAPTIVE_BALANCED_MEL_TOKENS,
            )
        if self.max_text_tokens_per_segment > target_text_tokens:
            changes.append(
                "max_text_tokens_per_segment "
                f"{self.max_text_tokens_per_segment}->{target_text_tokens}"
            )
            self.max_text_tokens_per_segment = target_text_tokens
        if self.max_mel_tokens > target_mel_tokens:
            changes.append(f"max_mel_tokens {self.max_mel_tokens}->{target_mel_tokens}")
            self.max_mel_tokens = target_mel_tokens

        if changes:
            refreshed_budget = self._resolve_cuda_startup_budget(snapshot=snapshot)
            event_bus.publish(
                "system_message",
                (
                    "Adjusted IndexTTS CUDA generation profile for current VRAM "
                    f"headroom ({'; '.join(changes)}). "
                    f"{self._format_cuda_startup_constraint_detail(snapshot=snapshot, budget=refreshed_budget)}."
                ),
            )
        elif budget.effective_budget_mb > configured_budget_mb:
            event_bus.publish(
                "system_message",
                (
                    "Using adaptive IndexTTS CUDA startup budget for current VRAM "
                    "headroom. "
                    f"{self._format_cuda_startup_constraint_detail(snapshot=snapshot, budget=budget)}."
                ),
            )

    def _snapshot_satisfies_cuda_startup_constraints(
        self,
        *,
        snapshot: _CudaMemorySnapshot,
    ) -> bool:
        """Return whether one VRAM snapshot satisfies the startup budget."""
        minimum_free_mb = int(max(self.cuda_startup_min_free_memory_mb, 0))
        if snapshot.free_mb < minimum_free_mb:
            return False
        budget = self._resolve_cuda_startup_budget(snapshot=snapshot)
        if budget.effective_budget_mb <= 0:
            return True
        return budget.projected_total_mb <= budget.effective_budget_mb

    def _format_cuda_startup_constraint_detail(
        self,
        *,
        snapshot: _CudaMemorySnapshot,
        budget: _CudaStartupBudget | None = None,
    ) -> str:
        """Render one operator-facing VRAM-budget detail string."""
        resolved_budget = budget or self._resolve_cuda_startup_budget(snapshot=snapshot)
        configured_budget_mb = int(
            max(min(self.cuda_vram_budget_mb, snapshot.total_mb), 0)
        )
        return (
            f"used={snapshot.used_mb} MiB, free={snapshot.free_mb} MiB, "
            f"total={snapshot.total_mb} MiB, "
            f"predicted_worker_total~={resolved_budget.projected_total_mb} MiB, "
            f"configured_budget<={configured_budget_mb} MiB, "
            f"effective_budget<={resolved_budget.effective_budget_mb} MiB, "
            f"projected_free_after_startup~={resolved_budget.projected_free_after_startup_mb} MiB, "
            f"reserve_free>={resolved_budget.reserve_free_memory_mb} MiB, "
            f"need_free>={int(max(self.cuda_startup_min_free_memory_mb, 0))} MiB"
        )

    def _resolve_cuda_startup_budget(
        self,
        *,
        snapshot: _CudaMemorySnapshot,
    ) -> _CudaStartupBudget:
        """Resolve the effective startup budget from config and current telemetry."""
        configured_budget_mb = int(
            max(min(self.cuda_vram_budget_mb, snapshot.total_mb), 0)
        )
        expected_delta_mb = int(max(self.cuda_startup_expected_worker_delta_mb, 0))
        projected_total_mb = snapshot.used_mb + expected_delta_mb
        projected_free_after_startup_mb = snapshot.free_mb - expected_delta_mb
        reserve_free_memory_mb = (
            int(max(self.cuda_adaptive_reserve_free_memory_mb, 0))
            if self.cuda_adaptive_memory_enabled
            else 0
        )
        if not self.cuda_adaptive_memory_enabled:
            effective_budget_mb = configured_budget_mb
        elif configured_budget_mb <= 0:
            effective_budget_mb = max(snapshot.total_mb - reserve_free_memory_mb, 0)
        elif projected_free_after_startup_mb >= reserve_free_memory_mb:
            physical_budget_mb = max(snapshot.total_mb - reserve_free_memory_mb, 0)
            effective_budget_mb = max(
                configured_budget_mb,
                min(physical_budget_mb, projected_total_mb),
            )
        else:
            effective_budget_mb = configured_budget_mb
        return _CudaStartupBudget(
            effective_budget_mb=effective_budget_mb,
            expected_delta_mb=expected_delta_mb,
            projected_total_mb=projected_total_mb,
            projected_free_after_startup_mb=projected_free_after_startup_mb,
            reserve_free_memory_mb=reserve_free_memory_mb,
        )

    def _wait_for_cuda_worker_headroom(self, *, device: str) -> None:
        """Wait briefly for enough free VRAM before starting a CUDA worker."""
        if device.strip().lower() != "cuda":
            return
        minimum_free_mb = int(max(self.cuda_startup_min_free_memory_mb, 0))
        budget_mb = int(max(self.cuda_vram_budget_mb, 0))
        if minimum_free_mb <= 0 and budget_mb <= 0:
            return
        snapshot = _query_cuda_memory_snapshot()
        if snapshot is None:
            raise NonRetryableAudioStageError(
                "Unable to query NVIDIA GPU memory before starting IndexTTS on CUDA. "
                "Ensure nvidia-smi is available and retry."
            )
        if self._snapshot_satisfies_cuda_startup_constraints(snapshot=snapshot):
            return
        release_torch_device_memory_with_reporting(
            device="cuda",
            description="IndexTTS subprocess startup preparation",
        )
        snapshot = _query_cuda_memory_snapshot()
        if snapshot is None:
            raise NonRetryableAudioStageError(
                "Unable to query NVIDIA GPU memory after preparing IndexTTS CUDA startup. "
                "Ensure nvidia-smi is available and retry."
            )
        if self._snapshot_satisfies_cuda_startup_constraints(snapshot=snapshot):
            return
        wait_seconds = max(float(self.cuda_startup_wait_seconds), 0.0)
        poll_seconds = max(float(self.cuda_startup_poll_seconds), 0.1)
        event_bus.publish(
            "system_message",
            (
                "Waiting for IndexTTS CUDA VRAM budget/headroom before startup "
                f"({self._format_cuda_startup_constraint_detail(snapshot=snapshot)})."
            ),
        )
        deadline = time.monotonic() + wait_seconds
        latest_snapshot = snapshot
        while time.monotonic() < deadline:
            time.sleep(poll_seconds)
            snapshot = _query_cuda_memory_snapshot()
            if snapshot is None:
                raise NonRetryableAudioStageError(
                    "Lost NVIDIA GPU memory telemetry while waiting to start "
                    "IndexTTS on CUDA. Ensure nvidia-smi is available and retry."
                )
            latest_snapshot = snapshot
            if self._snapshot_satisfies_cuda_startup_constraints(snapshot=snapshot):
                event_bus.publish(
                    "system_message",
                    (
                        "Detected enough IndexTTS CUDA VRAM budget/headroom to continue "
                        f"startup ({self._format_cuda_startup_constraint_detail(snapshot=snapshot)})."
                    ),
                )
                return
        raise NonRetryableAudioStageError(
            "Insufficient CUDA VRAM budget/headroom to start IndexTTS. "
            f"{self._format_cuda_startup_constraint_detail(snapshot=latest_snapshot)}. "
            "Reduce concurrent CUDA workloads and retry."
        )

    def _evict_cached_subprocess_worker(
        self,
        *,
        expected_worker: _PersistentIndexTTSSubprocessWorker | None = None,
    ) -> None:
        """Remove one cached worker so the next request recreates it."""
        key = self._runtime_key()
        cached_worker: _PersistentIndexTTSSubprocessWorker | None = None
        with _SHARED_RUNTIME_LOCK:
            existing_worker = _SHARED_SUBPROCESS_WORKERS.get(key)
            if existing_worker is None:
                return
            if expected_worker is not None and existing_worker is not expected_worker:
                return
            cached_worker = _SHARED_SUBPROCESS_WORKERS.pop(key, None)
        if cached_worker is not None:
            cached_worker.close()

    def _runtime_key(self) -> _IndexTTSRuntimeKey:
        """Build the stable cache key for this gateway configuration."""
        backend = self.execution_backend.strip().lower()
        runner_project_dir = (
            self.runner_project_dir.resolve()
            if self.runner_project_dir is not None
            else None
        )
        return _IndexTTSRuntimeKey(
            execution_backend=backend,
            cfg_path=self.cfg_path.resolve(),
            model_dir=self.model_dir.resolve(),
            device=self.device,
            use_fp16=self.use_fp16,
            use_cuda_kernel=self.use_cuda_kernel,
            use_deepspeed=self.use_deepspeed,
            use_accel=self.use_accel,
            use_torch_compile=self.use_torch_compile,
            runner_project_dir=runner_project_dir,
            uv_executable=self.uv_executable,
            stream_subprocess_output=self.stream_subprocess_output,
        )
