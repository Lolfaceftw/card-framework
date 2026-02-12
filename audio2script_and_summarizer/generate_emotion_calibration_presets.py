"""Generate emotion pacing preset configuration for TTS calibration.

By default this script writes a deterministic local preset file. When
``--use-deepseek`` is enabled, it can request alternative phrasing from
DeepSeek, then validates and normalizes the returned payload.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from .logging_utils import configure_logging
from .tts_pacing_calibration import DEFAULT_PRESETS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _default_output_path() -> Path:
    """Return the repository-local default preset config path."""
    return Path(__file__).resolve().with_name("emotion_pacing_presets.json")


def _default_payload() -> dict[str, list[dict[str, object]]]:
    """Build deterministic preset payload from in-code defaults."""
    return {
        "presets": [
            {
                "name": preset.name,
                "emo_text": preset.emo_text,
                "emo_alpha": preset.emo_alpha,
                "calibration_text": preset.calibration_text,
                "keywords": list(preset.keywords),
            }
            for preset in DEFAULT_PRESETS
        ]
    }


def _validate_payload(payload: Any) -> dict[str, list[dict[str, object]]]:
    """Validate and normalize payload into the preset JSON contract."""
    if not isinstance(payload, dict):
        raise ValueError("Preset payload must be a JSON object.")
    raw_presets = payload.get("presets")
    if not isinstance(raw_presets, list) or not raw_presets:
        raise ValueError("Preset payload must include a non-empty 'presets' array.")

    normalized: list[dict[str, object]] = []
    for raw_item in raw_presets:
        if not isinstance(raw_item, dict):
            continue
        name = str(raw_item.get("name", "")).strip().lower()
        emo_text = str(raw_item.get("emo_text", "")).strip()
        calibration_text = str(raw_item.get("calibration_text", "")).strip()
        if not name or not emo_text or not calibration_text:
            continue
        try:
            emo_alpha = float(raw_item.get("emo_alpha", 0.6))
        except (TypeError, ValueError):
            emo_alpha = 0.6
        emo_alpha = max(0.0, min(1.0, emo_alpha))
        raw_keywords = raw_item.get("keywords", [])
        keywords: list[str] = []
        if isinstance(raw_keywords, list):
            keywords = [
                str(keyword).strip().lower()
                for keyword in raw_keywords
                if str(keyword).strip()
            ]
        normalized.append(
            {
                "name": name,
                "emo_text": emo_text,
                "emo_alpha": emo_alpha,
                "calibration_text": calibration_text,
                "keywords": keywords,
            }
        )

    if not normalized:
        raise ValueError("No valid preset entries found in payload.")
    return {"presets": normalized}


def _build_deepseek_prompt(seed_payload_json: str) -> str:
    """Build a structured prompt for DeepSeek preset generation.

    The prompt follows repository prompt standards for objective, context,
    inputs, output contract, constraints, examples, and acceptance checks.
    """
    return (
        "Objective:\n"
        "- Produce concise emotion pacing presets for TTS calibration.\n\n"
        "Context:\n"
        "- Presets will drive speech-rate calibration in a voice cloning pipeline.\n"
        "- Output must remain deterministic-friendly and machine-parseable.\n\n"
        "Inputs:\n"
        "- Seed preset JSON (baseline style and schema):\n"
        f"{seed_payload_json}\n\n"
        "Output contract:\n"
        "- Return STRICT JSON object only with key 'presets'.\n"
        "- 'presets' must be an array of objects with keys:\n"
        "  name (string), emo_text (string), emo_alpha (0..1 float),\n"
        "  calibration_text (string), keywords (array of lowercase strings).\n"
        "- Keep 5 to 8 presets, each with distinct pacing intent.\n\n"
        "Rules:\n"
        "- No markdown, no comments, no extra top-level keys.\n"
        "- Keep calibration_text to 1-2 sentences and neutral-safe content.\n"
        "- Avoid sensitive or domain-specific claims.\n"
        "- Use lowercase snake_case for preset names.\n\n"
        "Examples:\n"
        "- Example preset name: 'excited_fast'\n"
        "- Example keyword list: ['excited', 'energetic', 'fast']\n\n"
        "Evaluation:\n"
        "- Verify JSON parses without cleanup.\n"
        "- Verify every preset has all required keys and valid types.\n"
        "- Verify emo_alpha is clamped to [0.0, 1.0].\n"
    )


def _generate_with_deepseek(
    *,
    api_key: str,
    model: str,
    seed_payload: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    """Generate preset candidates from DeepSeek and validate result."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    seed_payload_json = json.dumps(seed_payload, ensure_ascii=True, indent=2)
    prompt = _build_deepseek_prompt(seed_payload_json=seed_payload_json)

    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    if not completion.choices:
        raise ValueError("DeepSeek returned no choices.")
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("DeepSeek returned empty content.")
    payload = json.loads(content)
    return _validate_payload(payload)


def main() -> int:
    """Generate and write emotion pacing presets."""
    parser = argparse.ArgumentParser(
        description="Generate emotion pacing preset JSON for TTS calibration."
    )
    parser.add_argument(
        "--output",
        default=str(_default_output_path()),
        help="Output preset JSON path.",
    )
    parser.add_argument(
        "--use-deepseek",
        action="store_true",
        default=False,
        help="Generate candidate presets with DeepSeek before validation.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help="DeepSeek API key (required when --use-deepseek is set).",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="DeepSeek model id used with --use-deepseek.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite output file if it already exists.",
    )
    args = parser.parse_args()

    configure_logging(
        level=os.getenv("AUDIO2SCRIPT_LOG_LEVEL", "INFO"),
        component="generate_emotion_presets",
    )

    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        logger.error(
            "Output file already exists: %s. Pass --force to overwrite.",
            output_path,
        )
        return 1

    payload = _default_payload()
    if args.use_deepseek:
        if not args.api_key:
            logger.error("--use-deepseek requires --api-key or DEEPSEEK_API_KEY.")
            return 1
        logger.info("Generating preset candidates with DeepSeek model=%s", args.model)
        payload = _generate_with_deepseek(
            api_key=args.api_key,
            model=args.model,
            seed_payload=payload,
        )
    else:
        payload = _validate_payload(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    logger.info(
        "Emotion pacing presets written: %s (count=%d)",
        output_path,
        len(payload["presets"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
