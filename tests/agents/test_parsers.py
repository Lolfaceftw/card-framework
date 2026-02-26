"""Parser behavior tests for fallback tool-call formats."""

from __future__ import annotations

from agents.parsers import get_default_parser, get_default_parser_with_options


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


def test_text_fallback_parses_edit_remove_and_finalize_calls() -> None:
    parser = get_default_parser_with_options(enable_extended_text_fallback=True)
    message = {
        "role": "assistant",
        "content": (
            'edit_message(line=6, new_content="trim this section")\n'
            "remove_message(2)\n"
            "finalize_draft()"
        ),
    }

    calls = parser.parse(message)

    assert len(calls) == 3
    assert calls[0]["name"] == "edit_message"
    assert calls[0]["arguments"]["line"] == 6
    assert calls[1]["name"] == "remove_message"
    assert calls[1]["arguments"]["line"] == 2
    assert calls[2]["name"] == "finalize_draft"
    assert calls[2]["arguments"] == {}


def test_text_fallback_parses_json_style_call_with_trailing_comma() -> None:
    parser = get_default_parser_with_options(enable_extended_text_fallback=True)
    message = {
        "role": "assistant",
        "content": (
            'edit_message({"line": "5", "new_content": "new body",})\n'
            'remove_message({"line": "3",})'
        ),
    }

    calls = parser.parse(message)

    assert len(calls) == 2
    assert calls[0]["name"] == "edit_message"
    assert calls[0]["arguments"]["line"] == "5"
    assert calls[0]["arguments"]["new_content"] == "new body"
    assert calls[1]["name"] == "remove_message"
    assert calls[1]["arguments"]["line"] == "3"


def test_text_fallback_can_disable_extended_patterns() -> None:
    parser = get_default_parser_with_options(enable_extended_text_fallback=False)
    message = {
        "role": "assistant",
        "content": 'edit_message(line=4, new_content="x")\nfinalize_draft()',
    }

    calls = parser.parse(message)

    assert calls == []
