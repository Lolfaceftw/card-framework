"""HF Mistral 8B helper utilities for backchanneling."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MODEL_ID = os.environ.get(
    "MISTRAL_MODEL_ID",
    "mistralai/Ministral-8B-Instruct-2410",
)
MAX_NEW_TOKENS = int(os.environ.get("MISTRAL_MAX_NEW_TOKENS", "64"))

_TOKENIZER = None
_MODEL = None


def ensure_model() -> bool:
    """Lazy-load the HF Mistral 8B model."""
    global _TOKENIZER, _MODEL  # noqa: PLW0603
    if _TOKENIZER is not None and _MODEL is not None:
        return True
    try:
        logger.info("Loading HF model: %s (4-bit)", MODEL_ID)
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        _MODEL.eval()
        logger.info("✓ HF model loaded")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load HF model: %s", exc)
        return False


def _build_prompt(messages: List[Dict[str, str]]) -> str:
    """Build a chat prompt string."""
    if _TOKENIZER is None:
        return ""
    try:
        return _TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        joined = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            joined.append(f"{role.upper()}: {content}")
        joined.append("ASSISTANT:")
        return "\n".join(joined)


def generate_text(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str | None:
    """Generate text with the HF Mistral model."""
    if not ensure_model():
        return None
    prompt = _build_prompt(messages)
    if not prompt:
        return None
    inputs = _TOKENIZER(prompt, return_tensors="pt")
    inputs = {k: v.to(_MODEL.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=_TOKENIZER.eos_token_id,
            eos_token_id=_TOKENIZER.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return _TOKENIZER.decode(generated, skip_special_tokens=True).strip()


def extract_first_json(text: str) -> Dict[str, Any] | None:
    """Extract and parse the first JSON object from text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    payload = text[start : end + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None
