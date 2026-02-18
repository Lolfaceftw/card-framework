import argparse
import gc
import logging
import json
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Final, Iterable, Protocol

import torch
from tqdm import tqdm

THIRD_PARTY_DEPRECATION_PATTERNS: Final[tuple[str, ...]] = (
    r".*pkg_resources is deprecated as an API.*",
    r".*torchaudio\._backend\.utils\.info has been deprecated.*",
    r".*torchaudio\._backend\.common\.AudioMetaData has been deprecated.*",
    r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
    r".*load_with_torchcodec.*",
    r".*save_with_torchcodec.*",
    r".*Module 'speechbrain\.pretrained' was deprecated.*",
    r".*std\(\): degrees of freedom is <= 0.*",
)
PIPELINE_STAGE_COUNT: Final[int] = 10
PYANNOTE_MODEL_NAME: Final[str] = "pyannote/speaker-diarization-3.1"


def _parse_show_deprecation_warnings_flag(argv: list[str]) -> bool:
    """Parse ``--show-deprecation-warnings`` from partial CLI arguments.

    Args:
        argv: Raw command-line arguments excluding the script path.

    Returns:
        ``True`` when deprecation warnings should remain visible.
    """
    preflight_parser = argparse.ArgumentParser(add_help=False)
    preflight_parser.add_argument(
        "--show-deprecation-warnings",
        action="store_true",
        default=False,
    )
    known_args, _ = preflight_parser.parse_known_args(argv)
    return bool(known_args.show_deprecation_warnings)


def _apply_deprecation_warning_filters(show_deprecation_warnings: bool) -> None:
    """Filter noisy third-party warnings when requested.

    Args:
        show_deprecation_warnings: Keep warnings visible when True.
    """
    if show_deprecation_warnings:
        return
    for pattern in THIRD_PARTY_DEPRECATION_PATTERNS:
        warnings.filterwarnings("ignore", category=UserWarning, message=pattern)
        warnings.filterwarnings("ignore", category=FutureWarning, message=pattern)
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=pattern)


def _suppress_noisy_third_party_loggers(show_deprecation_warnings: bool) -> None:
    """Reduce noisy third-party DEBUG logs in normal CLI runs."""
    if show_deprecation_warnings:
        return
    for logger_name in (
        "speechbrain",
        "speechbrain.utils",
        "speechbrain.utils.checkpoints",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.getLogger("speechbrain.utils.checkpoints").disabled = True


_SHOW_DEPRECATION_WARNINGS = _parse_show_deprecation_warnings_flag(sys.argv[1:])
_apply_deprecation_warning_filters(show_deprecation_warnings=_SHOW_DEPRECATION_WARNINGS)
_suppress_noisy_third_party_loggers(
    show_deprecation_warnings=_SHOW_DEPRECATION_WARNINGS
)

import faster_whisper  # noqa: E402

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

RICH_CONSOLE: Any = Console() if RICH_AVAILABLE else None

from ctc_forced_aligner import (  # noqa: E402
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel  # noqa: E402

from .helpers import (  # noqa: E402
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)
from .logging_utils import configure_logging  # noqa: E402

# --- Environment setup for Windows compatibility ---
# Disable symlinks to avoid Windows permission issues
if os.environ.get("HF_HUB_DISABLE_SYMLINKS") != "1":
    print(
        "[INFO] Setting HF_HUB_DISABLE_SYMLINKS=1 to avoid Windows symlink permission issues..."
    )
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Prompt for HuggingFace token if not set (required for pyannote.audio)
if not os.environ.get("HF_TOKEN"):
    print("\n" + "=" * 60)
    print("[WARN] HuggingFace token required for pyannote.audio diarization")
    print("=" * 60)
    print("To get a token:")
    print("  1. Create an account at https://huggingface.co")
    print(
        "  2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
    )
    print("  3. Get your token at https://huggingface.co/settings/tokens")
    print("=" * 60)
    hf_token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("[INFO] HF_TOKEN set successfully.\n")
    else:
        print(
            "[WARNING] No token provided. Diarization may fail if pyannote is used.\n"
        )
# -------------------------------------------------------


def _is_cuda_device(device: str) -> bool:
    """Return True when the requested device targets CUDA."""
    return device.lower().startswith("cuda")


def _resolve_device(device: str) -> str:
    """Return an available runtime device."""
    if _is_cuda_device(device) and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return device


def _compute_type_for_device(device: str) -> str:
    """Return the Whisper compute type for the selected device."""
    return "float16" if _is_cuda_device(device) else "int8"


def _cuda_empty_cache(device: str) -> None:
    """Free cached CUDA memory only when running on CUDA."""
    if _is_cuda_device(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_demucs_command(cmd: list[str], hide_child_output: bool) -> int:
    """Run Demucs and emit normalized progress markers.

    Args:
        cmd: Full Demucs command line.
        hide_child_output: Suppress raw Demucs lines when True.

    Returns:
        Subprocess return code.
    """
    percent_pattern = re.compile(r"(\d{1,3})%")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print("[PROGRESS] htdemucs start")
    last_percent = -1
    assert process.stdout is not None

    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        match = percent_pattern.search(line)
        if match is not None:
            percent_value = min(100, max(0, int(match.group(1))))
            if percent_value != last_percent:
                last_percent = percent_value
                print(f"[PROGRESS] htdemucs {percent_value}")
        if not hide_child_output:
            print(line)

    return_code = process.wait()
    if return_code == 0:
        if last_percent < 100:
            print("[PROGRESS] htdemucs 100")
        print("[PROGRESS] htdemucs done")
    else:
        print("[PROGRESS] htdemucs failed")
    return return_code


class ProgressReporter(Protocol):
    """Protocol for interchangeable progress reporters."""

    def advance(self, stage_name: str) -> None:
        """Advance progress by one stage."""

    def close(self) -> None:
        """Release progress resources."""


@dataclass(slots=True)
class TqdmPipelineProgress:
    """Track pipeline progress with ``tqdm``."""

    progress_bar: tqdm

    def advance(self, stage_name: str) -> None:
        """Advance progress by one stage.

        Args:
            stage_name: Human-readable stage label.
        """
        self.progress_bar.set_postfix_str(stage_name)
        self.progress_bar.update(1)

    def close(self) -> None:
        """Close progress bar resources."""
        self.progress_bar.close()


class RichPipelineProgress:
    """Track pipeline progress with ``rich``."""

    def __init__(self, disable: bool) -> None:
        """Initialize a stage-level rich progress bar.

        Args:
            disable: Disable visual output when True.
        """
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total} stages"),
            TimeElapsedColumn(),
            disable=disable,
            console=RICH_CONSOLE,
        )
        self._task_id: Any = self._progress.add_task(
            "Audio2Script: starting",
            total=PIPELINE_STAGE_COUNT,
        )
        self._progress.start()

    def advance(self, stage_name: str) -> None:
        """Advance progress by one stage.

        Args:
            stage_name: Human-readable stage label.
        """
        self._progress.update(
            self._task_id,
            advance=1,
            description=f"Audio2Script: {stage_name}",
        )

    def close(self) -> None:
        """Stop rich progress rendering."""
        self._progress.stop()


def _build_pipeline_progress(disable: bool, plain_ui: bool) -> ProgressReporter:
    """Create a stage-level progress tracker.

    Args:
        disable: Disable visual progress output when True.
        plain_ui: Force plain progress rendering when True.

    Returns:
        A progress reporter implementation.
    """
    if RICH_AVAILABLE and not plain_ui:
        return RichPipelineProgress(disable=disable)

    return TqdmPipelineProgress(
        progress_bar=tqdm(
            total=PIPELINE_STAGE_COUNT,
            desc="Audio2Script",
            unit="stage",
            dynamic_ncols=True,
            disable=disable,
        )
    )


def _collect_transcript_text(
    transcript_segments: Iterable[Any],
    total_duration_seconds: float | None,
    disable_progress: bool,
    plain_ui: bool,
) -> str:
    """Collect transcription text while reporting progress.

    Args:
        transcript_segments: Iterable of Faster-Whisper segment objects.
        total_duration_seconds: Audio duration in seconds when available.
        disable_progress: Disable visual progress output when True.
        plain_ui: Force plain progress rendering when True.

    Returns:
        Concatenated transcript text.
    """
    if disable_progress:
        return "".join(segment.text for segment in transcript_segments)

    if RICH_AVAILABLE and not plain_ui:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=RICH_CONSOLE,
        )
        task_id = progress.add_task(
            "Whisper transcription",
            total=total_duration_seconds if total_duration_seconds else None,
        )
        current_completed = 0.0
        text_parts: list[str] = []
        with progress:
            for segment in transcript_segments:
                text_parts.append(segment.text)
                segment_end = float(getattr(segment, "end", 0.0) or 0.0)
                if total_duration_seconds and segment_end > current_completed:
                    current_completed = min(total_duration_seconds, segment_end)
                    progress.update(task_id, completed=current_completed)
                else:
                    progress.update(task_id, advance=1)
        return "".join(text_parts)

    return "".join(
        segment.text
        for segment in tqdm(
            transcript_segments,
            desc="Whisper transcription",
            unit="segment",
            dynamic_ncols=True,
        )
    )


pid = os.getpid()
temp_outputs_dir = f"temp_outputs_{pid}"
temp_path = os.path.join(os.getcwd(), "temp_outputs")
os.makedirs(temp_path, exist_ok=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="medium.en",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, "
    "set to 0 for original whisper longform inference",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)

parser.add_argument(
    "--diarizer",
    default="pyannote",
    choices=["pyannote", "msdd"],
    help="Choose the diarization model to use (pyannote is recommended, msdd requires NeMo)",
)
parser.add_argument(
    "--show-deprecation-warnings",
    action="store_true",
    default=False,
    help="Show noisy third-party deprecation warnings from audio dependencies.",
)
parser.add_argument(
    "--no-progress",
    action="store_true",
    default=False,
    help="Disable stage and transcription progress bars.",
)
parser.add_argument(
    "--plain-ui",
    action="store_true",
    default=False,
    help="Disable rich progress rendering and use plain terminal progress output.",
)
parser.add_argument(
    "--log-level",
    default=os.getenv("AUDIO2SCRIPT_LOG_LEVEL", "INFO").upper(),
    type=str.upper,
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Console log level for diarization runtime logs.",
)

args = parser.parse_args()
configure_logging(
    level=args.log_level,
    component="diarize",
)
args.device = _resolve_device(args.device)
language = process_language_arg(args.language, args.model_name)
_apply_deprecation_warning_filters(
    show_deprecation_warnings=args.show_deprecation_warnings
)
pipeline_progress = _build_pipeline_progress(
    disable=args.no_progress,
    plain_ui=args.plain_ui,
)
pipeline_progress.advance("Setup complete")

if args.stemming:
    # Isolate vocals from the rest of the audio
    print(
        f"[MODEL] Source separation model=htdemucs | backend=demucs | device={args.device}"
    )
    print("[STATUS] Running source separation (Demucs htdemucs)...")

    demucs_cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs",
        "--two-stems=vocals",
        "--jobs",
        "0",
        args.audio,
        "-o",
        temp_outputs_dir,
        "--device",
        args.device,
    ]
    return_code = _run_demucs_command(
        cmd=demucs_cmd,
        hide_child_output=args.no_progress,
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. "
            "Use --no-stem argument to disable it."
        )
        print("[STATUS] Source separation failed. Falling back to original audio.")
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            temp_outputs_dir,
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
        logging.info("Source separation complete.")
        print("[STATUS] Source separation complete. Preparing Whisper transcription.")
else:
    vocal_target = args.audio
    logging.info("Source separation skipped (--no-stem).")
    print("[STATUS] Source separation skipped (--no-stem). Using original audio.")

pipeline_progress.advance("Source separation")


# Transcribe the audio file
whisper_compute_type = _compute_type_for_device(args.device)
print(
    f"[MODEL] Whisper model={args.model_name} | device={args.device} | compute_type={whisper_compute_type}"
)
print("[STATUS] Initializing Whisper model weights...")

whisper_model = faster_whisper.WhisperModel(
    args.model_name,
    device=args.device,
    compute_type=whisper_compute_type,
)
pipeline_progress.advance("Whisper model loaded")
print("[STATUS] Whisper model initialized. Starting transcription...")
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
audio_waveform = faster_whisper.decode_audio(vocal_target)
suppress_tokens = (
    find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    if args.suppress_numerals
    else [-1]
)

if args.batch_size > 0:
    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        batch_size=args.batch_size,
    )
else:
    transcript_segments, info = whisper_model.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        vad_filter=True,
    )

full_transcript = _collect_transcript_text(
    transcript_segments=transcript_segments,
    total_duration_seconds=float(getattr(info, "duration", 0.0) or 0.0),
    disable_progress=args.no_progress,
    plain_ui=args.plain_ui,
)
pipeline_progress.advance("Whisper transcription")

# --- MEMORY FLUSH (Required for preventing crashes on low-VRAM GPUs) ---
print("[INFO] Clearing Whisper from VRAM...")
del whisper_model, whisper_pipeline
gc.collect()  # Force Python to free RAM immediately
_cuda_empty_cache(args.device)
# ----------------------------------------------------------------------

# Forced Alignment
alignment_model, alignment_tokenizer = load_alignment_model(
    args.device,
    dtype=torch.float16 if _is_cuda_device(args.device) else torch.float32,
)

emissions, stride = generate_emissions(
    alignment_model,
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device),
    batch_size=args.batch_size,
)

del alignment_model
_cuda_empty_cache(args.device)

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[info.language],
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)
pipeline_progress.advance("Forced alignment")

diarizer_model: Any
if args.diarizer == "pyannote":
    print(f"[MODEL] Diarizer model={PYANNOTE_MODEL_NAME} | backend=pyannote.audio")
    print("[STATUS] Initializing pyannote diarization pipeline...")
    from .diarization import PyannoteDiarizer

    if PyannoteDiarizer is None:
        raise ImportError(
            "pyannote diarizer is not available. Install pyannote.audio extras."
        )
    diarizer_model = PyannoteDiarizer(device=args.device)
elif args.diarizer == "msdd":
    print("[MODEL] Diarizer model=msdd | backend=nemo_toolkit[asr]")
    print("[STATUS] Initializing NeMo MSDD diarizer...")
    from .diarization import MSDDDiarizer

    if MSDDDiarizer is None:
        raise ImportError(
            "NeMo MSDD diarizer not available. Install nemo_toolkit[asr] in a separate environment "
            "or use --diarizer pyannote."
        )
    diarizer_model = MSDDDiarizer(device=args.device)

pipeline_progress.advance("Diarizer model loaded")
print("[STATUS] Running speaker diarization inference...")
speaker_ts = diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
del diarizer_model
_cuda_empty_cache(args.device)
pipeline_progress.advance("Speaker diarization")

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if info.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    def is_acronym(value: str) -> re.Match[str] | None:
        """Return a regex match when `value` is a dotted acronym."""
        return re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", value)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

else:
    logging.warning(
        f"Punctuation restoration is not available for {info.language} language."
        " Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
pipeline_progress.advance("Punctuation + sentence mapping")

with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

grouped_segments = []
if ssm:
    # Start with the first segment
    current_seg = {
        "speaker": ssm[0]["speaker"],
        "start_time": ssm[0]["start_time"],
        "end_time": ssm[0]["end_time"],
        "text": ssm[0]["text"],
    }

    for next_seg in ssm[1:]:
        if next_seg["speaker"] == current_seg["speaker"]:
            # SAME SPEAKER: Append text and extend end time
            current_seg["text"] = (
                current_seg["text"].strip() + " " + next_seg["text"].strip()
            )
            current_seg["end_time"] = next_seg["end_time"]
        else:
            # NEW SPEAKER: Save current and start new
            grouped_segments.append(current_seg)
            current_seg = {
                "speaker": next_seg["speaker"],
                "start_time": next_seg["start_time"],
                "end_time": next_seg["end_time"],
                "text": next_seg["text"],
            }
    # Don't forget the last one
    grouped_segments.append(current_seg)

# Save the GROUPED list to JSON
json_path = f"{os.path.splitext(args.audio)[0]}.json"
print(f"[INFO] Saving grouped transcript to {json_path}...")

with open(json_path, "w", encoding="utf-8") as jf:
    json.dump({"segments": grouped_segments}, jf, indent=2)
pipeline_progress.advance("Output files saved")

cleanup(temp_path)
pipeline_progress.advance("Cleanup complete")
pipeline_progress.close()
