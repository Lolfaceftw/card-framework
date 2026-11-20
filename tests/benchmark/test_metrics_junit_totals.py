from __future__ import annotations

from pathlib import Path

from card_framework.benchmark.metrics import parse_junit_totals


def test_parse_junit_totals_returns_zeros_for_missing_file(tmp_path: Path) -> None:
    totals = parse_junit_totals(tmp_path / "missing.xml")
    assert totals == {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}


def test_parse_junit_totals_parses_testsuite_root(tmp_path: Path) -> None:
    junit_path = tmp_path / "pytest.junit.xml"
    junit_path.write_text(
        '<testsuite tests="12" failures="1" errors="2" skipped="3"></testsuite>',
        encoding="utf-8",
    )

    totals = parse_junit_totals(junit_path)

    assert totals == {"tests": 12, "failures": 1, "errors": 2, "skipped": 3}


def test_parse_junit_totals_handles_nested_testsuites_and_bad_values(
    tmp_path: Path,
) -> None:
    junit_path = tmp_path / "pytest.junit.xml"
    junit_path.write_text(
        (
            "<testsuites>"
            '<testsuite tests="not-an-int" failures="0" errors="1" skipped="2"></testsuite>'
            "</testsuites>"
        ),
        encoding="utf-8",
    )

    totals = parse_junit_totals(junit_path)

    assert totals == {"tests": 0, "failures": 0, "errors": 1, "skipped": 2}


def test_parse_junit_totals_returns_zeros_for_invalid_xml(tmp_path: Path) -> None:
    junit_path = tmp_path / "pytest.junit.xml"
    junit_path.write_text("<testsuite>", encoding="utf-8")

    totals = parse_junit_totals(junit_path)

    assert totals == {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}

