import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from agents.tool_call_utils import dedupe_tool_calls_by_signature


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

    def parse(self, msg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls = []
        content = msg_dict.get("content") or ""
        fn_matches = re.findall(
            r"add_speaker_message\s*\(\s*(SPEAKER_\d+)\s*,\s*\"(.*?)\"\s*\)",
            content,
            re.DOTALL,
        )
        for idx, (speaker_id, msg_content) in enumerate(fn_matches):
            calls.append(
                {
                    "id": f"text_fallback_{idx}",
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
                return dedupe_tool_calls_by_signature(calls)
        return []


def get_default_parser() -> OutputParser:
    return CompositeParser(
        [JSONToolCallParser(), XMLFallbackParser(), TextFallbackParser()]
    )
