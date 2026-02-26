"""Parser behavior tests for fallback tool-call formats."""

from __future__ import annotations

from agents.parsers import get_default_parser


def test_xml_fallback_assigns_unique_ids_for_distinct_calls() -> None:
    parser = get_default_parser()
    message = {
        "role": "assistant",
        "content": """
<tool_call>
{"name":"add_speaker_message","arguments":{"speaker_id":"SPEAKER_00","content":"First"}}
</tool_call>
<tool_call>
{"name":"add_speaker_message","arguments":{"speaker_id":"SPEAKER_01","content":"Second"}}
</tool_call>
""",
    }

    calls = parser.parse(message)

    assert len(calls) == 2
    assert calls[0]["id"] == "xml_fallback_0"
    assert calls[1]["id"] == "xml_fallback_1"


def test_xml_fallback_dedupes_identical_calls_by_signature() -> None:
    parser = get_default_parser()
    message = {
        "role": "assistant",
        "content": """
<tool_call>
{"name":"add_speaker_message","arguments":{"speaker_id":"SPEAKER_00","content":"Same"}}
</tool_call>
<tool_call>
{"name":"add_speaker_message","arguments":{"speaker_id":"SPEAKER_00","content":"Same"}}
</tool_call>
""",
    }

    calls = parser.parse(message)

    assert len(calls) == 1
    assert calls[0]["id"] == "xml_fallback_0"


def test_text_fallback_assigns_unique_ids() -> None:
    parser = get_default_parser()
    message = {
        "role": "assistant",
        "content": (
            'add_speaker_message(SPEAKER_00, "first line")\n'
            'add_speaker_message(SPEAKER_01, "second line")'
        ),
    }

    calls = parser.parse(message)

    assert len(calls) == 2
    assert calls[0]["id"] == "text_fallback_0"
    assert calls[1]["id"] == "text_fallback_1"
