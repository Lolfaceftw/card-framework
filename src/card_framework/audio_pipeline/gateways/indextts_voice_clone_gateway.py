"""IndexTTS-backed voice-cloning gateway."""

from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from card_framework.audio_pipeline.errors import DependencyMissingError, NonRetryableAudioStageError
from card_framework.audio_pipeline.voice_clone_contracts import VoiceCloneProvider

_INDEXTTS_PROTOCOL_PREFIX = "__INDEXTTS_JSON__"
_INDEXTTS_READY_TIMEOUT_SECONDS = 600.0
_INDEXTTS_PROTOCOL_POLL_SECONDS = 0.2
_INDEXTTS_LOG_TAIL_LINE_LIMIT = 80
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


_SHARED_INPROCESS_MODELS: dict[_IndexTTSRuntimeKey, object] = {}
_SHARED_SUBPROCESS_WORKERS: dict[_IndexTTSRuntimeKey, "_PersistentIndexTTSSubprocessWorker"] = {}
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
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="IndexTTSPersistentWorkerReader",
            daemon=True,
        )
        self._reader_thread.start()
        ready_payload = self._wait_for_protocol_message(
            expected_type="ready",
            timeout_seconds=_INDEXTTS_READY_TIMEOUT_SECONDS,
        )
        if not bool(ready_payload.get("ok", False)):
            detail = self._format_failure_detail(
                str(ready_payload.get("error", "")).strip()
            )
            self.close()
            raise NonRetryableAudioStageError(
                "IndexTTS subprocess worker failed to initialize: "
                f"{detail}"
            )

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None,
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
                "IndexTTS subprocess synthesis failed: "
                f"{detail}"
            )
        return output_audio_path

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
                "Failed to communicate with IndexTTS subprocess worker: "
                f"{detail}"
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
                        "IndexTTS subprocess worker exited unexpectedly: "
                        f"{detail}"
                    )
                continue
            payload_type = str(payload.get("type", "")).strip()
            if payload_type == "process_exit":
                detail = self._format_failure_detail("")
                raise NonRetryableAudioStageError(
                    "IndexTTS subprocess worker exited unexpectedly: "
                    f"{detail}"
                )
            if payload_type != expected_type:
                continue
            if request_id is not None and str(payload.get("request_id", "")) != request_id:
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
        if self._stream_output:
            if normalized_error:
                return f"{normalized_error} See streamed subprocess output above."
            return "See streamed subprocess output above."
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
            "IndexTTS runner project directory does not exist: "
            f"{resolved_project_dir}"
        )
    return resolved_project_dir


def _normalize_emo_text(emo_text: str | None) -> str | None:
    """Normalize optional emotion-guidance text into a cacheable value."""
    if emo_text is None:
        return None
    normalized = emo_text.strip()
    return normalized or None


def _release_inprocess_model(model: object) -> None:
    """Best-effort release of one cached in-process IndexTTS runtime."""
    del model
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
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
        _release_inprocess_model(model)
    with _SHARED_RUNTIME_LOCK:
        _SHARED_INPROCESS_EMO_VECTOR_CACHE.pop(key, None)


def _release_all_cached_runtime_resources() -> None:
    """Release all cached IndexTTS runtimes held by this Python process."""
    with _SHARED_RUNTIME_LOCK:
        models = list(_SHARED_INPROCESS_MODELS.values())
        workers = list(_SHARED_SUBPROCESS_WORKERS.values())
        _SHARED_INPROCESS_MODELS.clear()
        _SHARED_SUBPROCESS_WORKERS.clear()
        _SHARED_INPROCESS_EMO_VECTOR_CACHE.clear()
    for worker in workers:
        worker.close()
    for model in models:
        _release_inprocess_model(model)


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

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None = None,
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
            )
        elif backend == "subprocess":
            rendered_path = self._synthesize_subprocess(
                reference_audio_path=reference_audio_path,
                text=text.strip(),
                output_audio_path=output_audio_path,
                emo_text=emo_text,
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
    ) -> Path:
        """Run synthesis by importing IndexTTS2 in current Python process."""
        tts_model = self._get_or_init_model()
        emo_vector = self._get_or_resolve_emo_vector(
            tts_model=tts_model,
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
                "IndexTTS synthesis failed for output "
                f"{output_audio_path}."
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
    ) -> Path:
        """Run synthesis via a persistent ``uv run --project`` worker."""
        worker = self._get_or_init_subprocess_worker()
        rendered_path = worker.synthesize(
            reference_audio_path=reference_audio_path,
            text=text,
            output_audio_path=output_audio_path,
            emo_text=emo_text,
            verbose=self.verbose,
            max_text_tokens_per_segment=self.max_text_tokens_per_segment,
            generation_kwargs=self._generation_kwargs(),
        )
        if not rendered_path.exists():
            raise NonRetryableAudioStageError(
                "IndexTTS subprocess completed without output file: "
                f"{rendered_path}"
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

    def _get_or_init_subprocess_worker(self) -> _PersistentIndexTTSSubprocessWorker:
        """Lazily initialize and share one warm subprocess worker per config."""
        cfg_path = self.cfg_path.resolve()
        model_dir = self.model_dir.resolve()
        _validate_model_artifact_paths(cfg_path=cfg_path, model_dir=model_dir)
        runner_project_dir = _validate_runner_project_dir(
            runner_project_dir=self.runner_project_dir
        )
        key = self._runtime_key()
        with _SHARED_RUNTIME_LOCK:
            cached_worker = _SHARED_SUBPROCESS_WORKERS.get(key)
            if cached_worker is not None and cached_worker.is_alive():
                return cached_worker
            if cached_worker is not None:
                cached_worker.close()
            cached_worker = _PersistentIndexTTSSubprocessWorker(
                cfg_path=cfg_path,
                model_dir=model_dir,
                device=self.device,
                use_fp16=self.use_fp16,
                use_cuda_kernel=self.use_cuda_kernel,
                use_deepspeed=self.use_deepspeed,
                use_accel=self.use_accel,
                use_torch_compile=self.use_torch_compile,
                runner_project_dir=runner_project_dir,
                uv_executable=self.uv_executable,
                stream_output=self.stream_subprocess_output,
            )
            _SHARED_SUBPROCESS_WORKERS[key] = cached_worker
            return cached_worker

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

