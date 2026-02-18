"""Stage 3 voice cloning and interjection pipeline for CARD.

This module consumes the Stage 2 summary JSON and produces a merged WAV file
using IndexTTS2 voice cloning plus optional Mistral-driven interjections.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

from pydub import AudioSegment

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SUMMARY_SCAN_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
    }
)
SUMMARY_EXCLUDED_SUFFIXES = (".report.json", ".agent_buffer")
DEFAULT_INTERJECTION_MAX_RATIO = 0.35
DEFAULT_MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_MISTRAL_MAX_NEW_TOKENS = 64
DEFAULT_INTERJECTION_EMO_TEXT = "Casual listening agreement"
DEFAULT_INTERJECTION_EMO_ALPHA = 0.5
DEFAULT_SEGMENT_PAUSE_MS = 400
DEFAULT_CROSSFADE_MS = 300

STYLE_DELAY_MS: dict[str, tuple[int, int]] = {
    "agreement": (180, 420),
    "surprise": (120, 320),
    "empathy": (450, 900),
    "clarify": (260, 520),
    "humor": (140, 360),
    "neutral": (250, 600),
}


class SummaryEntryPayload(TypedDict, total=False):
    """Represent raw summary line payload produced by Stage 2."""

    speaker: str
    voice_sample: str
    text: str
    use_emo_text: bool
    emo_text: str
    emo_alpha: float


@dataclass(slots=True, frozen=True)
class SummaryEntry:
    """Represent one normalized summary entry for voice synthesis."""

    speaker: str
    voice_sample: str
    text: str
    use_emo_text: bool
    emo_text: str
    emo_alpha: float


@dataclass(slots=True, frozen=True)
class SegmentArtifact:
    """Represent synthesized main segment metadata."""

    index: int
    entry: SummaryEntry
    audio_path: Path
    audio_duration_ms: int


@dataclass(slots=True, frozen=True)
class InterjectionRequest:
    """Represent context passed to the interjection planner."""

    segment_index: int
    main_speaker: str
    main_text: str
    candidate_speaker: str
    previous_text: str | None
    next_text: str | None


@dataclass(slots=True, frozen=True)
class InterjectionCandidate:
    """Represent a planner-proposed interjection."""

    segment_index: int
    interjector_index: int
    interjection_text: str
    anchor_phrase: str
    style: str
    confidence: float


@dataclass(slots=True, frozen=True)
class Stage3Result:
    """Represent Stage 3 output artifact paths and summary metrics."""

    output_wav_path: Path
    interjection_log_path: Path
    output_duration_ms: int
    interjection_count: int
    segment_count: int
    mistral_enabled: bool


class TTSEngineProtocol(Protocol):
    """Define required methods for the TTS backend."""

    def infer(
        self,
        *,
        spk_audio_prompt: str,
        text: str,
        output_path: str,
        emo_alpha: float,
        use_emo_text: bool,
        emo_text: str,
        use_random: bool,
        verbose: bool,
    ) -> object:
        """Synthesize speech from text and a speaker reference audio."""


class InterjectionPlannerProtocol(Protocol):
    """Define required methods for interjection planning."""

    def ensure_available(self) -> bool:
        """Load planner resources if needed and return availability."""

    def propose(self, request: InterjectionRequest) -> InterjectionCandidate | None:
        """Return a candidate interjection or ``None``."""


def _is_valid_summary_payload(payload: object) -> bool:
    """Return ``True`` when payload resembles Stage 2 summary output."""
    if not isinstance(payload, list) or not payload:
        return False
    first = payload[0]
    if not isinstance(first, dict):
        return False
    return all(key in first for key in ("speaker", "voice_sample", "text"))


def discover_summary_json_files(search_root: Path) -> list[Path]:
    """Discover summary JSON files sorted by modified time, newest first.

    Args:
        search_root: Root directory to scan recursively.

    Returns:
        List of valid summary JSON files sorted newest first.
    """
    if not search_root.exists():
        return []

    candidates: list[Path] = []
    for current_root, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in SUMMARY_SCAN_EXCLUDED_DIRS]
        for filename in filenames:
            lowered = filename.lower()
            if not lowered.endswith("_summary.json"):
                continue
            if any(lowered.endswith(suffix) for suffix in SUMMARY_EXCLUDED_SUFFIXES):
                continue
            candidate_path = Path(current_root) / filename
            if candidate_path.name.startswith("."):
                continue
            try:
                if candidate_path.stat().st_size > 20 * 1024 * 1024:
                    continue
            except OSError:
                continue
            try:
                payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if _is_valid_summary_payload(payload):
                candidates.append(candidate_path.resolve())

    candidates.sort(
        key=lambda path: (
            path.stat().st_mtime if path.exists() else 0.0,
            str(path),
        ),
        reverse=True,
    )
    return candidates


def select_latest_summary_json(search_root: Path) -> Path:
    """Return the most recent summary JSON file under ``search_root``.

    Args:
        search_root: Root directory to scan.

    Raises:
        FileNotFoundError: No summary file was found.
    """
    candidates = discover_summary_json_files(search_root)
    if not candidates:
        raise FileNotFoundError(f"No summary JSON files found under {search_root}.")
    return candidates[0]


def load_summary_entries(summary_json_path: Path) -> list[SummaryEntry]:
    """Load and normalize summary entries from JSON.

    Args:
        summary_json_path: Path to Stage 2 summary JSON.

    Returns:
        Normalized summary entries.

    Raises:
        ValueError: JSON schema is invalid.
    """
    raw_payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, list):
        raise ValueError("Summary JSON must be a list of dialogue entries.")

    entries: list[SummaryEntry] = []
    for index, item in enumerate(raw_payload):
        if not isinstance(item, dict):
            raise ValueError(f"Summary entry {index} is not an object.")
        payload = SummaryEntryPayload(**item)
        speaker = str(payload.get("speaker", "")).strip()
        voice_sample = str(payload.get("voice_sample", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not speaker or not voice_sample or not text:
            raise ValueError(
                f"Summary entry {index} must include speaker, voice_sample, and text."
            )
        raw_alpha = payload.get("emo_alpha", 0.6)
        try:
            emo_alpha = float(raw_alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Entry {index} has invalid emo_alpha: {raw_alpha}"
            ) from exc
        emo_alpha = max(0.0, min(1.0, emo_alpha))
        use_emo_text = bool(payload.get("use_emo_text", True))
        emo_text = str(payload.get("emo_text", "")).strip()
        entries.append(
            SummaryEntry(
                speaker=speaker,
                voice_sample=voice_sample,
                text=text,
                use_emo_text=use_emo_text,
                emo_text=emo_text,
                emo_alpha=emo_alpha,
            )
        )
    if not entries:
        raise ValueError("Summary JSON does not contain any entries.")
    return entries


def resolve_voice_sample_path(voice_sample: str, summary_json_dir: Path) -> Path:
    """Resolve a voice sample path from summary JSON.

    Args:
        voice_sample: Voice sample path in summary JSON.
        summary_json_dir: Parent directory of summary JSON.

    Returns:
        Resolved existing path.

    Raises:
        FileNotFoundError: Voice sample does not exist.
    """
    candidate = Path(voice_sample)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    summary_relative = (summary_json_dir / candidate).resolve()
    if summary_relative.exists():
        return summary_relative
    cwd_relative = (Path.cwd() / candidate).resolve()
    if cwd_relative.exists():
        return cwd_relative
    raise FileNotFoundError(
        f"Voice sample not found for '{voice_sample}' relative to {summary_json_dir}."
    )


def select_nearest_alternate_speaker_index(
    entries: list[SummaryEntry],
    segment_index: int,
) -> int | None:
    """Find the nearest segment index with a different speaker.

    Ties are resolved by preferring the previous segment.
    """
    if segment_index < 0 or segment_index >= len(entries):
        return None
    current_speaker = entries[segment_index].speaker

    previous_index: int | None = None
    for idx in range(segment_index - 1, -1, -1):
        if entries[idx].speaker != current_speaker:
            previous_index = idx
            break

    next_index: int | None = None
    for idx in range(segment_index + 1, len(entries)):
        if entries[idx].speaker != current_speaker:
            next_index = idx
            break

    if previous_index is None and next_index is None:
        return None
    if previous_index is None:
        return next_index
    if next_index is None:
        return previous_index

    previous_distance = segment_index - previous_index
    next_distance = next_index - segment_index
    if previous_distance <= next_distance:
        return previous_index
    return next_index


def _find_anchor_end_ratio(text: str, anchor_phrase: str) -> float:
    """Compute anchor end ratio in text, falling back to midpoint when missing."""
    normalized_text = text.strip()
    if not normalized_text:
        return 0.5
    if not anchor_phrase.strip():
        return 0.5
    lowered_text = normalized_text.lower()
    lowered_anchor = anchor_phrase.strip().lower()
    start_index = lowered_text.find(lowered_anchor)
    if start_index < 0:
        return 0.5
    end_index = start_index + len(lowered_anchor)
    return min(1.0, max(0.0, end_index / max(1, len(normalized_text))))


def compute_interjection_position_ms(
    *,
    text: str,
    anchor_phrase: str,
    style: str,
    audio_duration_ms: int,
    rng: random.Random | None = None,
) -> int:
    """Compute overlap start position for an interjection.

    Args:
        text: Main segment text.
        anchor_phrase: Trigger anchor phrase from planner.
        style: Planner style token.
        audio_duration_ms: Main segment duration.
        rng: Optional random generator.

    Returns:
        Millisecond start offset for overlay.
    """
    effective_rng = rng or random
    bounded_duration = max(0, audio_duration_ms)
    if bounded_duration == 0:
        return 0

    anchor_ratio = _find_anchor_end_ratio(text=text, anchor_phrase=anchor_phrase)
    anchor_ms = int(bounded_duration * anchor_ratio)

    style_key = style.strip().lower()
    delay_min_ms, delay_max_ms = STYLE_DELAY_MS.get(
        style_key, STYLE_DELAY_MS["neutral"]
    )
    reaction_delay_ms = effective_rng.randint(delay_min_ms, delay_max_ms)
    position_ms = anchor_ms + reaction_delay_ms
    max_start_ms = max(0, bounded_duration - 500)
    return max(0, min(position_ms, max_start_ms))


def select_interjections_by_confidence(
    *,
    candidates: list[InterjectionCandidate],
    eligible_segment_count: int,
    max_ratio: float,
) -> list[InterjectionCandidate]:
    """Select highest confidence interjections under a global ratio cap."""
    normalized_ratio = max(0.0, min(1.0, max_ratio))
    max_allowed = int(math.floor(eligible_segment_count * normalized_ratio))
    if max_allowed <= 0:
        return []
    ordered = sorted(candidates, key=lambda c: (-c.confidence, c.segment_index))
    return ordered[:max_allowed]


def _extract_first_json_object(text: str) -> str | None:
    """Extract first JSON object from free-form model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    return text[start : end + 1]


class IndexTTS2Engine:
    """Lazy wrapper around ``indextts.infer_v2.IndexTTS2``."""

    def __init__(self, *, cfg_path: Path, model_dir: Path, device: str) -> None:
        self._cfg_path = cfg_path
        self._model_dir = model_dir
        self._device = device
        self._model: TTSEngineProtocol | None = None

    def _ensure_loaded(self) -> TTSEngineProtocol:
        """Load IndexTTS2 on first use."""
        if self._model is not None:
            return self._model

        from indextts.infer_v2 import IndexTTS2

        use_fp16 = self._device.lower().startswith("cuda")
        self._model = IndexTTS2(
            cfg_path=str(self._cfg_path),
            model_dir=str(self._model_dir),
            device=self._device,
            use_fp16=use_fp16,
            use_cuda_kernel=use_fp16,
        )
        return self._model

    def infer(
        self,
        *,
        spk_audio_prompt: str,
        text: str,
        output_path: str,
        emo_alpha: float,
        use_emo_text: bool,
        emo_text: str,
        use_random: bool,
        verbose: bool,
    ) -> object:
        """Proxy TTS inference call to the loaded IndexTTS2 model."""
        model = self._ensure_loaded()
        return model.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path=output_path,
            emo_alpha=emo_alpha,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            use_random=use_random,
            verbose=verbose,
        )


class MistralInterjectionPlanner:
    """Hugging Face Mistral planner with 4-bit quantized loading."""

    def __init__(self, *, model_id: str, max_new_tokens: int) -> None:
        self._model_id = model_id
        self._max_new_tokens = max(1, max_new_tokens)
        self._tokenizer: object | None = None
        self._model: object | None = None

    def ensure_available(self) -> bool:
        """Load model resources lazily."""
        if self._tokenizer is not None and self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                device_map="auto",
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            logger.info(
                "stage3_mistral_ready model_id=%s quantization=4bit",
                self._model_id,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "stage3_mistral_unavailable model_id=%s reason=%s",
                self._model_id,
                exc,
            )
            return False

    def _build_prompt(self, request: InterjectionRequest) -> list[dict[str, str]]:
        """Build planner messages for one segment decision."""
        previous_text = request.previous_text or ""
        next_text = request.next_text or ""
        prompt = (
            "You are a conversational overlap planner.\n"
            "Decide if a short listener interjection should overlap this segment.\n"
            "Return STRICT JSON only with keys:\n"
            "should_interject (bool), interjection_text (string), "
            "anchor_phrase (exact phrase from main_text), "
            "style (agreement|surprise|empathy|clarify|humor|neutral), "
            "confidence (0-1 float).\n"
            f"main_speaker: {request.main_speaker}\n"
            f"candidate_listener: {request.candidate_speaker}\n"
            f"main_text: {request.main_text}\n"
            f"previous_text: {previous_text}\n"
            f"next_text: {next_text}\n"
            "Use natural interjections such as 'uh-huh', 'hmm', short agreement, "
            "or short repeats when appropriate. Do not add prose."
        )
        return [
            {"role": "system", "content": "You return strict JSON only."},
            {"role": "user", "content": prompt},
        ]

    def propose(self, request: InterjectionRequest) -> InterjectionCandidate | None:
        """Generate one interjection proposal."""
        if not self.ensure_available():
            return None
        if self._tokenizer is None or self._model is None:
            return None
        try:
            import torch

            tokenizer = cast(Any, self._tokenizer)
            model = cast(Any, self._model)
            messages = self._build_prompt(request)
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = (
                    "\n".join(
                        f"{item['role'].upper()}: {item['content']}"
                        for item in messages
                    )
                    + "\nASSISTANT:"
                )
            encoded = tokenizer(prompt_text, return_tensors="pt")
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=True,
                    temperature=0.35,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = output_ids[0][encoded["input_ids"].shape[-1] :]
            raw_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            json_text = _extract_first_json_object(raw_text)
            if not json_text:
                return None
            payload = json.loads(json_text)
            should_interject = bool(payload.get("should_interject", False))
            if not should_interject:
                return None
            interjection_text = str(payload.get("interjection_text", "")).strip()
            anchor_phrase = str(payload.get("anchor_phrase", "")).strip()
            style = str(payload.get("style", "neutral")).strip().lower() or "neutral"
            confidence = float(payload.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            if not interjection_text or not anchor_phrase:
                return None
            return InterjectionCandidate(
                segment_index=request.segment_index,
                interjector_index=-1,
                interjection_text=interjection_text[:80],
                anchor_phrase=anchor_phrase,
                style=style,
                confidence=confidence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "stage3_planner_segment_failed segment=%d reason=%s",
                request.segment_index,
                exc,
            )
            return None


def run_stage3_pipeline(
    *,
    summary_json_path: Path,
    indextts_cfg_path: Path,
    indextts_model_dir: Path,
    output_wav_path: Path | None = None,
    device: str = "cuda",
    mistral_model_id: str = DEFAULT_MISTRAL_MODEL_ID,
    mistral_max_new_tokens: int = DEFAULT_MISTRAL_MAX_NEW_TOKENS,
    interjection_max_ratio: float = DEFAULT_INTERJECTION_MAX_RATIO,
    tts_engine: TTSEngineProtocol | None = None,
    planner: InterjectionPlannerProtocol | None = None,
    rng: random.Random | None = None,
) -> Stage3Result:
    """Run Stage 3 synthesis and optional interjection overlay.

    Args:
        summary_json_path: Path to Stage 2 summary output.
        indextts_cfg_path: IndexTTS2 config path.
        indextts_model_dir: IndexTTS2 model directory.
        output_wav_path: Optional final output WAV path.
        device: Runtime device string for TTS model.
        mistral_model_id: HF model id for interjection planner.
        mistral_max_new_tokens: Mistral generation token limit.
        interjection_max_ratio: Max fraction of eligible segments with overlaps.
        tts_engine: Optional injected TTS backend for tests.
        planner: Optional injected planner for tests.
        rng: Optional random generator.

    Returns:
        Stage 3 result metadata.
    """
    effective_rng = rng or random.Random()
    normalized_ratio = max(0.0, min(1.0, interjection_max_ratio))

    summary_path = summary_json_path.resolve()
    entries = load_summary_entries(summary_path)
    summary_dir = summary_path.parent
    resolved_output_path = (
        output_wav_path.resolve()
        if output_wav_path is not None
        else summary_path.with_name(f"{summary_path.stem}_resynth.wav")
    )
    temp_dir = resolved_output_path.with_name(f"{summary_path.stem}_stage3_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "stage3_started summary=%s segments=%d output=%s",
        summary_path,
        len(entries),
        resolved_output_path,
    )

    voice_paths: dict[str, Path] = {}
    for entry in entries:
        if entry.voice_sample in voice_paths:
            continue
        voice_paths[entry.voice_sample] = resolve_voice_sample_path(
            voice_sample=entry.voice_sample,
            summary_json_dir=summary_dir,
        )

    tts_backend = tts_engine or IndexTTS2Engine(
        cfg_path=indextts_cfg_path,
        model_dir=indextts_model_dir,
        device=device,
    )
    planner_backend = planner or MistralInterjectionPlanner(
        model_id=mistral_model_id,
        max_new_tokens=mistral_max_new_tokens,
    )

    artifacts: list[SegmentArtifact] = []
    for index, entry in enumerate(entries):
        output_path = temp_dir / f"gen_{index:05d}.wav"
        tts_backend.infer(
            spk_audio_prompt=str(voice_paths[entry.voice_sample]),
            text=entry.text,
            output_path=str(output_path),
            emo_alpha=entry.emo_alpha,
            use_emo_text=entry.use_emo_text,
            emo_text=entry.emo_text,
            use_random=False,
            verbose=False,
        )
        duration_ms = len(AudioSegment.from_wav(str(output_path)))
        artifacts.append(
            SegmentArtifact(
                index=index,
                entry=entry,
                audio_path=output_path,
                audio_duration_ms=duration_ms,
            )
        )
        logger.info(
            "stage3_main_segment_synthesized segment=%d speaker=%s duration_ms=%d",
            index,
            entry.speaker,
            duration_ms,
        )

    candidate_items: list[InterjectionCandidate] = []
    eligible_count = 0
    mistral_enabled = planner_backend.ensure_available()
    if not mistral_enabled:
        logger.warning(
            "stage3_mistral_degraded reason=planner_unavailable interjections=disabled"
        )
    else:
        for index, artifact in enumerate(artifacts):
            if index == 0:
                continue
            interjector_index = select_nearest_alternate_speaker_index(entries, index)
            if interjector_index is None:
                continue
            eligible_count += 1
            previous_text = entries[index - 1].text if index > 0 else None
            next_text = entries[index + 1].text if index + 1 < len(entries) else None
            request = InterjectionRequest(
                segment_index=index,
                main_speaker=artifact.entry.speaker,
                main_text=artifact.entry.text,
                candidate_speaker=entries[interjector_index].speaker,
                previous_text=previous_text,
                next_text=next_text,
            )
            planned = planner_backend.propose(request)
            if planned is None:
                continue
            candidate_items.append(
                InterjectionCandidate(
                    segment_index=planned.segment_index,
                    interjector_index=interjector_index,
                    interjection_text=planned.interjection_text,
                    anchor_phrase=planned.anchor_phrase,
                    style=planned.style,
                    confidence=planned.confidence,
                )
            )

    selected_candidates = select_interjections_by_confidence(
        candidates=candidate_items,
        eligible_segment_count=eligible_count,
        max_ratio=normalized_ratio,
    )
    selected_by_segment = {
        candidate.segment_index: candidate for candidate in selected_candidates
    }

    interjection_log: list[dict[str, object]] = []
    processed_segments: list[AudioSegment] = []

    for artifact in artifacts:
        base_audio = AudioSegment.from_wav(str(artifact.audio_path))
        candidate = selected_by_segment.get(artifact.index)
        if candidate is not None:
            interjector_entry = entries[candidate.interjector_index]
            interjection_wav = temp_dir / f"interjection_{artifact.index:05d}.wav"
            tts_backend.infer(
                spk_audio_prompt=str(voice_paths[interjector_entry.voice_sample]),
                text=candidate.interjection_text,
                output_path=str(interjection_wav),
                emo_alpha=DEFAULT_INTERJECTION_EMO_ALPHA,
                use_emo_text=True,
                emo_text=DEFAULT_INTERJECTION_EMO_TEXT,
                use_random=False,
                verbose=False,
            )
            interjection_audio = AudioSegment.from_wav(str(interjection_wav))
            position_ms = compute_interjection_position_ms(
                text=artifact.entry.text,
                anchor_phrase=candidate.anchor_phrase,
                style=candidate.style,
                audio_duration_ms=len(base_audio),
                rng=effective_rng,
            )
            base_audio = base_audio.overlay(interjection_audio, position=position_ms)
            interjection_log.append(
                {
                    "segment_index": artifact.index,
                    "main_speaker": artifact.entry.speaker,
                    "interjector_speaker": interjector_entry.speaker,
                    "interjection_text": candidate.interjection_text,
                    "anchor_phrase": candidate.anchor_phrase,
                    "style": candidate.style,
                    "confidence": candidate.confidence,
                    "position_ms": position_ms,
                }
            )
            logger.info(
                "stage3_interjection_applied segment=%d interjector=%s style=%s position_ms=%d",
                artifact.index,
                interjector_entry.speaker,
                candidate.style,
                position_ms,
            )
        processed_segments.append(base_audio)

    if not processed_segments:
        raise RuntimeError("Stage 3 produced no segments.")

    merged_audio = processed_segments[0]
    pause_audio = AudioSegment.silent(duration=DEFAULT_SEGMENT_PAUSE_MS)
    for segment_audio in processed_segments[1:]:
        merged_audio = merged_audio.append(pause_audio, crossfade=0)
        crossfade_ms = min(DEFAULT_CROSSFADE_MS, len(merged_audio), len(segment_audio))
        merged_audio = merged_audio.append(segment_audio, crossfade=crossfade_ms)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_audio.export(str(resolved_output_path), format="wav")

    interjection_log_path = resolved_output_path.with_name(
        f"{resolved_output_path.stem}_interjections.json"
    )
    interjection_log_path.write_text(
        json.dumps(interjection_log, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "stage3_completed output=%s duration_ms=%d interjections=%d segments=%d",
        resolved_output_path,
        len(merged_audio),
        len(interjection_log),
        len(processed_segments),
    )
    return Stage3Result(
        output_wav_path=resolved_output_path,
        interjection_log_path=interjection_log_path,
        output_duration_ms=len(merged_audio),
        interjection_count=len(interjection_log),
        segment_count=len(processed_segments),
        mistral_enabled=mistral_enabled,
    )
