"""
Transformers LLM Provider
==========================
Allows running models locally via the HuggingFace transformers library.
"""

import json
import sys
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_provider import LLMProvider
from ui import ui


class TransformersProvider(LLMProvider):
    """
    Concrete LLM strategy for local Hugging Face models using transformers.

    Args:
        model: Model identifier on HuggingFace Hub (e.g. 'Salesforce/Llama-xLAM-2-3b-fc-r').
        device_map: The device map to use (default: 'auto').
        torch_dtype: The torch data type to use (default: 'bfloat16').
    """

    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
    ) -> None:
        self.model_name = model

        dtype = (
            torch.bfloat16
            if torch_dtype == "bfloat16"
            else torch.float16
            if torch_dtype == "float16"
            else torch.float32
        )

        ui.print_system(
            f"Loading local Transformers model '{self.model_name}' (device_map={device_map}, dtype={torch_dtype})..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=device_map
        )

        ui.print_system(f"Successfully loaded model '{self.model_name}'.")

    # ── LLMProvider interface ─────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use generate kwargs since we do not have a dedicated generate without tools method
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        input_ids_len = inputs["input_ids"].shape[-1]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generate_kwargs = {**inputs}
        if max_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_tokens
        else:
            generate_kwargs["max_new_tokens"] = 1024  # default fallback

        outputs = self.model.generate(**generate_kwargs)
        generated_tokens = outputs[:, input_ids_len:]
        full_content = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )

        sys.stdout.write(f"\n{full_content}\n")
        sys.stdout.flush()

        return full_content.strip()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
    ):
        """
        Chat completion mimicking the OpenAI response format to integrate smoothly
        with the existing application, utilizing transformers chat templates.
        """

        apply_kwargs = dict(
            conversation=messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if tools:
            # We map function parameters from our tool_registry to expected format
            apply_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get("function", t).get("name"),
                        "description": t.get("function", t).get("description", ""),
                        "parameters": t.get("function", t).get("parameters", {}),
                    },
                }
                if "function" not in t
                else t
                for t in tools
            ]

        inputs = self.tokenizer.apply_chat_template(**apply_kwargs)
        input_ids_len = inputs["input_ids"].shape[-1]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generate_kwargs = {**inputs}
        if max_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_tokens
        else:
            generate_kwargs["max_new_tokens"] = 2048

        outputs = self.model.generate(**generate_kwargs)
        generated_tokens = outputs[:, input_ids_len:]
        output_text = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )

        # Construct and return an OpenAI-compatible message namespace
        # that `summarizer.py` lines 259-281 expects.

        # Some models use JSON for tool calls, let's see if we can parse it
        # Try to find a JSON block for tool calls

        tool_calls = []
        try:
            # Often xLAM formats output directly as JSON array of tool calls
            parsed = json.loads(output_text.strip())
            if isinstance(parsed, list):
                for i, call in enumerate(parsed):
                    tool_calls.append(
                        SimpleNamespace(
                            id=f"call_{i}",
                            type="function",
                            function=SimpleNamespace(
                                name=call.get("name"),
                                arguments=json.dumps(call.get("arguments", {})),
                            ),
                        )
                    )
            elif isinstance(parsed, dict) and "name" in parsed:
                tool_calls.append(
                    SimpleNamespace(
                        id="call_0",
                        type="function",
                        function=SimpleNamespace(
                            name=parsed.get("name"),
                            arguments=json.dumps(parsed.get("arguments", {})),
                        ),
                    )
                )
        except json.JSONDecodeError:
            pass  # Not a simple JSON block

        # Fallback to plain text content if we did not parse out tool calls explicitly
        # Our agent will fallback to text-parsing for some formats anyway

        mock_message = SimpleNamespace(
            role="assistant",
            content=output_text,
            tool_calls=tool_calls if tool_calls else None,
        )

        # Add model_dump method expected by utils
        mock_message.model_dump = lambda: {
            "role": "assistant",
            "content": output_text,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
            if tool_calls
            else None,
        }

        return mock_message
