"""Run one IndexTTS2 synthesis request inside the IndexTTS project environment."""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for one IndexTTS2 synthesis call."""
    parser = argparse.ArgumentParser(description="IndexTTS2 subprocess runner")
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--reference-audio-path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-audio-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--use-cuda-kernel", action="store_true")
    parser.add_argument("--use-deepspeed", action="store_true")
    parser.add_argument("--use-accel", action="store_true")
    parser.add_argument("--use-torch-compile", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    """Execute IndexTTS2 inference with validated arguments."""
    args = _build_parser().parse_args()
    output_path = Path(args.output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        use_fp16=args.use_fp16,
        device=args.device,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
        use_accel=args.use_accel,
        use_torch_compile=args.use_torch_compile,
    )
    tts.infer(
        spk_audio_prompt=args.reference_audio_path,
        text=args.text,
        output_path=str(output_path),
        verbose=args.verbose,
        max_text_tokens_per_segment=args.max_text_tokens_per_segment,
    )
    if not output_path.exists():
        raise RuntimeError(
            f"IndexTTS2 inference completed without output file: {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
