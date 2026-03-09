import ast
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from card_framework.agents.tool_call_utils import dedupe_tool_calls_by_signature
from card_framework.shared.summary_xml import DEFAULT_EMO_PRESET


class OutputParser(ABC):
    @abstractmethod
    def parse(self, msg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parses the model output and returns a list of tool call dictionaries:
        [{'id': str, 'name': str, 'arguments': dict}]
        Returns empty list if no tool calls matched.
        """
        pass


class JSONToolCallParser(OutputParser):
    """Parses standard OpenAI/Gemini structured tool calls."""

    def parse(self, msg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls = []
        if msg_dict.get("tool_calls"):
            for tc in msg_dict["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                calls.append(
                    {
                        "id": tc.get("id", "unknown"),
                        "name": func.get("name"),
                        "arguments": args,
                    }
                )
        return calls


class XMLFallbackParser(OutputParser):
    """Parses <tool_call>{...}</tool_call> format used by some open weight models."""

    def parse(self, msg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls = []
        content = msg_dict.get("content") or ""
        xml_matches = re.findall(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL
        )
        for idx, raw in enumerate(xml_matches):
            try:
                parsed = json.loads(raw)
                args = parsed.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append(
                    {
                        "id": f"xml_fallback_{idx}",
                        "name": parsed["name"],
                        "arguments": args,
                    }
                )
            except (json.JSONDecodeError, KeyError):
                pass
        return calls


class TextFallbackParser(OutputParser):
    """Parses raw ad-hoc text functions like add_speaker_message(...) inline."""

    def __init__(self, *, enable_extended_patterns: bool = True) -> None:
        self._enable_extended_patterns = enable_extended_patterns

    @staticmethod
    def _decode_string_literal(raw_value: str) -> str | None:
        """Decode a quoted Python/JSON style string literal."""
        try:
            decoded = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            return None
        return decoded if isinstance(decoded, str) else None

    @staticmethod
    def _parse_json_object(raw_value: str) -> dict[str, Any] | None:
        """
        Parse a JSON object with light normalization for trailing commas.

        The fallback parser accepts slightly malformed JSON that smaller models
        often emit in text mode (for example trailing commas).
        """
        normalized = re.sub(r",(\s*[}\]])", r"\1", raw_value.strip())
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def parse(self, msg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls_with_positions: list[tuple[int, dict[str, Any]]] = []
        content = msg_dict.get("content") or ""

        add_pattern = re.compile(
            r"add_speaker_message\s*\(\s*"
            r"(?:speaker_id\s*=\s*)?[\"']?(SPEAKER_\d+)[\"']?\s*,\s*"
            r"(?:content\s*=\s*)?(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
            r"(?:\s*,\s*(?:emo_preset\s*=\s*)?"
            r"(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'))?\s*\)",
            re.DOTALL,
        )
        for match in add_pattern.finditer(content):
            decoded_content = self._decode_string_literal(match.group(2))
            if decoded_content is None:
                continue
            decoded_preset = DEFAULT_EMO_PRESET
            if match.group(3) is not None:
                maybe_preset = self._decode_string_literal(match.group(3))
                if maybe_preset is not None and maybe_preset.strip():
                    decoded_preset = maybe_preset
            calls_with_positions.append(
                (
                    match.start(),
                    {
                        "name": "add_speaker_message",
                        "arguments": {
                            "speaker_id": match.group(1),
                            "content": decoded_content,
                            "emo_preset": decoded_preset,
                        },
                    },
                )
            )

        add_json_pattern = re.compile(
            r"add_speaker_message\s*\(\s*(\{.*?\})\s*\)",
            re.DOTALL,
        )
        for match in add_json_pattern.finditer(content):
            parsed_object = self._parse_json_object(match.group(1))
            if not parsed_object:
                continue
            speaker_id = parsed_object.get("speaker_id")
            message_content = parsed_object.get("content")
            emo_preset = parsed_object.get("emo_preset", DEFAULT_EMO_PRESET)
            if isinstance(speaker_id, str) and isinstance(message_content, str):
                calls_with_positions.append(
                    (
                        match.start(),
                        {
                            "name": "add_speaker_message",
                            "arguments": {
                                "speaker_id": speaker_id,
                                "content": message_content,
                                "emo_preset": str(emo_preset),
                            },
                        },
                    )
                )

        if not self._enable_extended_patterns:
            calls = []
            for idx, (_position, parsed_call) in enumerate(
                sorted(calls_with_positions, key=lambda item: item[0])
            ):
                calls.append(
                    {
                        "id": f"text_fallback_{idx}",
                        "name": parsed_call["name"],
                        "arguments": parsed_call["arguments"],
                    }
                )
            return calls

        edit_pattern = re.compile(
            r"edit_message\s*\(\s*"
            r"(?:line\s*=\s*)?[\"']?(\d+)[\"']?\s*,\s*"
            r"(?:new_content\s*=\s*)?(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
            r"(?:\s*,\s*(?:emo_preset\s*=\s*)?"
            r"(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'))?\s*\)",
            re.DOTALL,
        )
        for match in edit_pattern.finditer(content):
            decoded_content = self._decode_string_literal(match.group(2))
            if decoded_content is None:
                continue
            decoded_preset: str | None = None
            if match.group(3) is not None:
                maybe_preset = self._decode_string_literal(match.group(3))
                if maybe_preset is not None and maybe_preset.strip():
                    decoded_preset = maybe_preset
            arguments: dict[str, Any] = {
                "line": int(match.group(1)),
                "new_content": decoded_content,
            }
            if decoded_preset is not None:
                arguments["emo_preset"] = decoded_preset
            calls_with_positions.append(
                (
                    match.start(),
                    {
                        "name": "edit_message",
                        "arguments": arguments,
                    },
                )
            )

        edit_json_pattern = re.compile(
            r"edit_message\s*\(\s*(\{.*?\})\s*\)",
            re.DOTALL,
        )
        for match in edit_json_pattern.finditer(content):
            parsed_object = self._parse_json_object(match.group(1))
            if not parsed_object:
                continue
            line_value = parsed_object.get("line")
            new_content = parsed_object.get("new_content")
            emo_preset = parsed_object.get("emo_preset")
            if isinstance(new_content, str) and isinstance(line_value, (int, str)):
                arguments: dict[str, Any] = {
                    "line": line_value,
                    "new_content": new_content,
                }
                if isinstance(emo_preset, str) and emo_preset.strip():
                    arguments["emo_preset"] = emo_preset
                calls_with_positions.append(
                    (
                        match.start(),
                        {
                            "name": "edit_message",
                            "arguments": arguments,
                        },
                    )
                )

        remove_pattern = re.compile(
            r"remove_message\s*\(\s*(?:line\s*=\s*)?[\"']?(\d+)[\"']?\s*\)",
            re.DOTALL,
        )
        for match in remove_pattern.finditer(content):
            calls_with_positions.append(
                (
                    match.start(),
                    {
                        "name": "remove_message",
                        "arguments": {"line": int(match.group(1))},
                    },
                )
            )

        remove_json_pattern = re.compile(
            r"remove_message\s*\(\s*(\{.*?\})\s*\)",
            re.DOTALL,
        )
        for match in remove_json_pattern.finditer(content):
            parsed_object = self._parse_json_object(match.group(1))
            if not parsed_object:
                continue
            line_value = parsed_object.get("line")
            if isinstance(line_value, (int, str)):
                calls_with_positions.append(
                    (
                        match.start(),
                        {
                            "name": "remove_message",
                            "arguments": {"line": line_value},
                        },
                    )
                )

        finalize_pattern = re.compile(r"finalize_draft\s*\(\s*\)")
        for match in finalize_pattern.finditer(content):
            calls_with_positions.append(
                (
                    match.start(),
                    {"name": "finalize_draft", "arguments": {}},
                )
            )

        calls = []
        for idx, (_position, parsed_call) in enumerate(
            sorted(calls_with_positions, key=lambda item: item[0])
        ):
            call = {
                "id": f"text_fallback_{idx}",
                "name": parsed_call["name"],
                "arguments": parsed_call["arguments"],
            }
            calls.append(call)

        return calls


class CompositeParser(OutputParser):
    """Tries a list of parsers in order. Returns the first one that yields >0 calls."""

    def __init__(self, parsers: List[OutputParser]):
        self.parsers = parsers

    def parse(self, msg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        for parser in self.parsers:
            calls = parser.parse(msg_dict)
            if calls:
                return dedupe_tool_calls_by_signature(calls)
        return []


def get_default_parser() -> OutputParser:
    return get_default_parser_with_options(enable_extended_text_fallback=False)


def get_default_parser_with_options(
    *, enable_extended_text_fallback: bool
) -> OutputParser:
    return CompositeParser(
        [
            JSONToolCallParser(),
            XMLFallbackParser(),
            TextFallbackParser(enable_extended_patterns=enable_extended_text_fallback),
        ]
    )

