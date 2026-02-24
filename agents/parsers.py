import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List


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
        for raw in xml_matches:
            try:
                parsed = json.loads(raw)
                args = parsed.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append(
                    {
                        "id": "xml_fallback",
                        "name": parsed["name"],
                        "arguments": args,
                    }
                )
            except (json.JSONDecodeError, KeyError):
                pass
        return calls


class TextFallbackParser(OutputParser):
    """Parses raw ad-hoc text functions like add_speaker_message(...) inline."""

    def parse(self, msg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls = []
        content = msg_dict.get("content") or ""
        fn_matches = re.findall(
            r"add_speaker_message\s*\(\s*(SPEAKER_\d+)\s*,\s*\"(.*?)\"\s*\)",
            content,
            re.DOTALL,
        )
        for speaker_id, msg_content in fn_matches:
            calls.append(
                {
                    "id": "text_fallback",
                    "name": "add_speaker_message",
                    "arguments": {
                        "speaker_id": speaker_id,
                        "content": msg_content.replace('\\"', '"'),
                    },
                }
            )
        return calls


class CompositeParser(OutputParser):
    """Tries a list of parsers in order. Returns the first one that yields >0 calls."""

    def __init__(self, parsers: List[OutputParser]):
        self.parsers = parsers

    def parse(self, msg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        for parser in self.parsers:
            calls = parser.parse(msg_dict)
            if calls:
                return calls
        return []


def get_default_parser() -> OutputParser:
    return CompositeParser(
        [JSONToolCallParser(), XMLFallbackParser(), TextFallbackParser()]
    )
