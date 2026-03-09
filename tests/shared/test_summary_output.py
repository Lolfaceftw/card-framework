from pathlib import Path

from card_framework.shared.summary_output import write_summary_xml_to_workspace


def test_write_summary_xml_to_workspace_creates_summary_file(tmp_path: Path) -> None:
    output_path = write_summary_xml_to_workspace(
        "<SPEAKER_00>Hello world</SPEAKER_00>",
        tmp_path,
    )

    assert output_path == tmp_path / "summary.xml"
    assert output_path.read_text(encoding="utf-8") == (
        "<SPEAKER_00>Hello world</SPEAKER_00>\n"
    )


def test_write_summary_xml_to_workspace_strips_outer_whitespace(tmp_path: Path) -> None:
    output_path = write_summary_xml_to_workspace(
        "\n  <SPEAKER_00>Hi</SPEAKER_00>\n\n",
        tmp_path,
    )

    assert output_path.read_text(encoding="utf-8") == "<SPEAKER_00>Hi</SPEAKER_00>\n"

