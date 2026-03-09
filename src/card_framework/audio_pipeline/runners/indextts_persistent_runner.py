"""Serve repeated IndexTTS2 synthesis requests from one warm subprocess."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from typing import Any

_INDEXTTS_PROTOCOL_PREFIX = "__INDEXTTS_JSON__"


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the persistent IndexTTS2 worker."""
    parser = argparse.ArgumentParser(description="IndexTTS2 persistent worker")
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--use-cuda-kernel", action="store_true")
    parser.add_argument("--use-deepspeed", action="store_true")
    parser.add_argument("--use-accel", action="store_true")
    parser.add_argument("--use-torch-compile", action="store_true")
    return parser


def _configure_standard_streams() -> None:
    """Prefer UTF-8 line-buffered stdio for worker protocol stability."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def _emit_protocol(payload: dict[str, Any]) -> None:
    """Write one protocol payload to stdout with the reserved prefix."""
    sys.stdout.write(
        _INDEXTTS_PROTOCOL_PREFIX
        + json.dumps(payload, ensure_ascii=False)
        + "\n"
    )
    sys.stdout.flush()


def _render_exception(exc: Exception) -> str:
    """Render one compact exception string for protocol responses."""
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _build_tts(args: argparse.Namespace) -> object:
    """Construct the warm IndexTTS2 runtime."""
    from indextts.infer_v2 import IndexTTS2

    return IndexTTS2(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        use_fp16=args.use_fp16,
        device=args.device,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
        use_accel=args.use_accel,
        use_torch_compile=args.use_torch_compile,
    )


def _normalize_emo_text(emo_text: Any) -> str | None:
    """Normalize one optional emotion-guidance prompt."""
    if emo_text is None:
        return None
    normalized = str(emo_text).strip()
    return normalized or None


def _resolve_cached_emo_vector(
    *,
    tts: object,
    emo_text: str | None,
    emo_vector_cache: dict[str, tuple[float, ...]],
) -> list[float] | None:
    """Resolve one emotion vector, caching repeated text analysis results."""
    normalized_emo_text = _normalize_emo_text(emo_text)
    if normalized_emo_text is None:
        return None
    cached_vector = emo_vector_cache.get(normalized_emo_text)
    if cached_vector is not None:
        return list(cached_vector)
    qwen_emo = getattr(tts, "qwen_emo", None)
    if qwen_emo is None or not hasattr(qwen_emo, "inference"):
        return None
    with redirect_stdout(sys.stderr):
        emo_dict = qwen_emo.inference(normalized_emo_text)
    emo_vector = tuple(float(value) for value in emo_dict.values())
    emo_vector_cache[normalized_emo_text] = emo_vector
    return list(emo_vector)


def _handle_synthesize_request(
    tts: object,
    payload: dict[str, Any],
    *,
    emo_vector_cache: dict[str, tuple[float, ...]],
) -> dict[str, Any]:
    """Run one synth request against the cached IndexTTS2 runtime."""
    request_id = str(payload.get("request_id", "")).strip()
    reference_audio_path = str(payload.get("reference_audio_path", "")).strip()
    text = str(payload.get("text", "")).strip()
    output_audio_path = Path(str(payload.get("output_audio_path", "")).strip())
    emo_text = _normalize_emo_text(payload.get("emo_text"))
    verbose = bool(payload.get("verbose", False))
    max_text_tokens_per_segment = int(payload.get("max_text_tokens_per_segment", 120))
    raw_generation_kwargs = payload.get("generation_kwargs", {})
    generation_kwargs = (
        dict(raw_generation_kwargs)
        if isinstance(raw_generation_kwargs, dict)
        else {}
    )

    if not request_id:
        raise ValueError("Missing request_id.")
    if not reference_audio_path:
        raise ValueError("Missing reference_audio_path.")
    if not text:
        raise ValueError("Missing text.")
    if not str(output_audio_path).strip():
        raise ValueError("Missing output_audio_path.")

    emo_vector = _resolve_cached_emo_vector(
        tts=tts,
        emo_text=emo_text,
        emo_vector_cache=emo_vector_cache,
    )
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(sys.stderr):
        infer = getattr(tts, "infer")
        infer(
            spk_audio_prompt=reference_audio_path,
            text=text,
            output_path=str(output_audio_path),
            verbose=verbose,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            use_emo_text=emo_vector is None and emo_text is not None,
            emo_text=emo_text if emo_vector is None else None,
            emo_vector=emo_vector,
            **generation_kwargs,
        )
    if not output_audio_path.exists():
        raise RuntimeError(
            f"IndexTTS2 inference completed without output file: {output_audio_path}"
        )
    return {
        "type": "result",
        "request_id": request_id,
        "ok": True,
        "output_audio_path": str(output_audio_path),
    }


def main() -> int:
    """Start the persistent worker loop."""
    _configure_standard_streams()
    args = _build_parser().parse_args()

    try:
        with redirect_stdout(sys.stderr):
            tts = _build_tts(args)
    except Exception as exc:
        _emit_protocol(
            {
                "type": "ready",
                "ok": False,
                "error": _render_exception(exc),
            }
        )
        return 1

    emo_vector_cache: dict[str, tuple[float, ...]] = {}
    _emit_protocol({"type": "ready", "ok": True})
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit_protocol(
                {
                    "type": "result",
                    "request_id": "",
                    "ok": False,
                    "error": _render_exception(exc),
                }
            )
            continue

        request_id = str(payload.get("request_id", "")).strip()
        action = str(payload.get("action", "")).strip().lower()
        if action == "shutdown":
            _emit_protocol(
                {
                    "type": "shutdown",
                    "request_id": request_id,
                    "ok": True,
                }
            )
            break
        if action != "synthesize":
            _emit_protocol(
                {
                    "type": "result",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"Unsupported action: {action or '<empty>'}",
                }
            )
            continue

        try:
            response = _handle_synthesize_request(
                tts,
                payload,
                emo_vector_cache=emo_vector_cache,
            )
        except Exception as exc:
            response = {
                "type": "result",
                "request_id": request_id,
                "ok": False,
                "error": _render_exception(exc),
            }
        _emit_protocol(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
