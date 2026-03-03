"""IndexTTS-backed voice-cloning gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Any

from audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from audio_pipeline.errors import DependencyMissingError, NonRetryableAudioStageError
from audio_pipeline.voice_clone_contracts import VoiceCloneProvider


@dataclass(slots=True)
class IndexTTSVoiceCloneGateway(VoiceCloneProvider):
    """
    Voice-cloning provider backed by ``indextts.infer_v2.IndexTTS2``.

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
    execution_backend: str = "subprocess"
    runner_project_dir: Path | None = None
    uv_executable: str = "uv"
    stream_subprocess_output: bool = True
    _tts_model: object | None = field(default=None, init=False, repr=False)

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
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
            )
        elif backend == "subprocess":
            rendered_path = self._synthesize_subprocess(
                reference_audio_path=reference_audio_path,
                text=text.strip(),
                output_audio_path=output_audio_path,
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

    def _synthesize_inprocess(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
    ) -> Path:
        """Run synthesis by importing IndexTTS2 in current Python process."""
        tts_model = self._get_or_init_model()
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result: Any = tts_model.infer(
                spk_audio_prompt=str(reference_audio_path),
                text=text,
                output_path=str(output_audio_path),
                verbose=self.verbose,
                max_text_tokens_per_segment=self.max_text_tokens_per_segment,
            )
            del result  # output is persisted to output_audio_path.
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
    ) -> Path:
        """Run synthesis via ``uv run --project`` in a dedicated IndexTTS env."""
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        if self.runner_project_dir is None:
            raise NonRetryableAudioStageError(
                "IndexTTS subprocess backend requires runner_project_dir."
            )
        runner_project_dir = self.runner_project_dir.resolve()
        if not runner_project_dir.exists() or not runner_project_dir.is_dir():
            raise NonRetryableAudioStageError(
                "IndexTTS runner project directory does not exist: "
                f"{runner_project_dir}"
            )
        runner_script = (
            Path(__file__).resolve().parent.parent
            / "runners"
            / "indextts_infer_runner.py"
        )
        if not runner_script.exists():
            raise NonRetryableAudioStageError(
                f"IndexTTS runner script does not exist: {runner_script}"
            )

        command = [
            self.uv_executable,
            "run",
            "--project",
            str(runner_project_dir),
            "python",
            str(runner_script),
            "--cfg-path",
            str(self.cfg_path),
            "--model-dir",
            str(self.model_dir),
            "--reference-audio-path",
            str(reference_audio_path),
            "--text",
            text,
            "--output-audio-path",
            str(output_audio_path),
            "--device",
            self.device,
            "--max-text-tokens-per-segment",
            str(self.max_text_tokens_per_segment),
        ]
        if self.use_fp16:
            command.append("--use-fp16")
        if self.use_cuda_kernel:
            command.append("--use-cuda-kernel")
        if self.use_deepspeed:
            command.append("--use-deepspeed")
        if self.use_accel:
            command.append("--use-accel")
        if self.use_torch_compile:
            command.append("--use-torch-compile")
        if self.verbose:
            command.append("--verbose")

        if self.stream_subprocess_output:
            result = subprocess.run(
                command,
                check=False,
                text=True,
            )
        else:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            if self.stream_subprocess_output:
                detail = "See streamed subprocess output above."
            else:
                stderr_tail = (result.stderr or "").strip()
                stdout_tail = (result.stdout or "").strip()
                detail = stderr_tail or stdout_tail or "No subprocess output captured."
            raise NonRetryableAudioStageError(
                "IndexTTS subprocess synthesis failed with "
                f"exit_code={result.returncode}: {detail}"
            )
        if not output_audio_path.exists():
            raise NonRetryableAudioStageError(
                "IndexTTS subprocess completed without output file: "
                f"{output_audio_path}"
            )
        return output_audio_path

    def _get_or_init_model(self) -> object:
        """Lazily initialize and cache IndexTTS2 model instance."""
        if self._tts_model is not None:
            return self._tts_model
        if not self.cfg_path.exists():
            raise NonRetryableAudioStageError(
                f"IndexTTS config file does not exist: {self.cfg_path}"
            )
        if not self.model_dir.exists() or not self.model_dir.is_dir():
            raise NonRetryableAudioStageError(
                f"IndexTTS model directory does not exist: {self.model_dir}"
            )
        try:
            from indextts.infer_v2 import IndexTTS2
        except ImportError as exc:
            raise DependencyMissingError(
                "IndexTTS2 is not installed. Install index-tts to enable voice cloning."
            ) from exc

        try:
            self._tts_model = IndexTTS2(
                cfg_path=str(self.cfg_path),
                model_dir=str(self.model_dir),
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
        return self._tts_model
