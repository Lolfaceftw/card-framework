from card_framework.shared.logger_utils import _summarize_tool_result


def test_summarize_tool_result_reports_dict_keys_for_json_payload() -> None:
    summary = _summarize_tool_result(
        '{"excerpt":"[SPEAKER_00]: hello\\n[SPEAKER_01]: world","match_count":2}'
    )

    assert summary == "dict keys=['excerpt', 'match_count']"


def test_summarize_tool_result_truncates_plain_text_payloads() -> None:
    summary = _summarize_tool_result("word " * 80)

    assert len(summary) <= 160
    assert summary.endswith("...")

