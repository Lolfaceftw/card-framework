"""Run lint, type, and test gates with machine-verifiable evidence artifacts.

This script is the canonical quality command for contributor and agent claims.
It executes scoped `ruff`, `mypy`, and `pytest` checks, writes raw logs, emits
JUnit XML for tests, and generates a JSON report with SHA-256 hashes so results
can be independently verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


LOGGER = logging.getLogger("quality_gate")
CORE_SCOPE_PATHS: tuple[str, ...] = (
    "audio2script_and_summarizer",
    "CARD-SpeakerAudioExtraction/src",
    "tests",
)


@dataclass(slots=True)
class StepResult:
    """Represent execution metadata for one quality check step.

    Args:
        name: Human-readable step name.
        command: Full command argv used to execute the step.
        return_code: Process exit code.
        started_at_utc: UTC ISO-8601 timestamp for command start.
        finished_at_utc: UTC ISO-8601 timestamp for command end.
        duration_seconds: Step runtime in seconds.
        stdout_path: Absolute path to captured stdout file.
        stderr_path: Absolute path to captured stderr file.
    """

    name: str
    command: list[str]
    return_code: int
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    stdout_path: str
    stderr_path: str


@dataclass(slots=True)
class PytestSummary:
    """Capture parsed JUnit outcome totals for test evidence validation.

    Args:
        tests: Total number of tests reported by JUnit XML.
        failures: Number of failed tests.
        errors: Number of errored tests.
        skipped: Number of skipped tests.
        junit_xml_path: Absolute path to the JUnit XML artifact.
    """

    tests: int
    failures: int
    errors: int
    skipped: int
    junit_xml_path: str


def _configure_logging() -> None:
    """Configure console logging for quality gate execution."""
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def _utc_iso_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_of_file(path: Path) -> str:
    """Compute the SHA-256 digest for a file.

    Args:
        path: File path to hash.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_step(name: str, command: list[str], cwd: Path, output_dir: Path) -> StepResult:
    """Execute one command and persist raw stdout/stderr logs.

    Args:
        name: Step name used in output artifact filenames.
        command: Command argv to execute.
        cwd: Working directory for the subprocess.
        output_dir: Evidence directory where step logs are written.

    Returns:
        StepResult containing command metadata and artifact paths.
    """
    started_at = datetime.now(timezone.utc)
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"

    LOGGER.info("Running %s: %s", name, " ".join(command))
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return_code = completed.returncode
        stdout_text = completed.stdout
        stderr_text = completed.stderr
    except FileNotFoundError as error:
        return_code = 127
        stdout_text = ""
        stderr_text = f"{error.__class__.__name__}: {error}\n"

    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")

    finished_at = datetime.now(timezone.utc)
    duration_seconds = (finished_at - started_at).total_seconds()
    return StepResult(
        name=name,
        command=command,
        return_code=return_code,
        started_at_utc=started_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at_utc=finished_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_seconds=duration_seconds,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _parse_junit(junit_xml_path: Path) -> PytestSummary:
    """Parse pytest JUnit XML totals for evidence validation.

    Args:
        junit_xml_path: Path to JUnit XML file emitted by pytest.

    Returns:
        Parsed PytestSummary totals.

    Raises:
        FileNotFoundError: If the JUnit file does not exist.
        xml.etree.ElementTree.ParseError: If the JUnit file is malformed.
        ValueError: If no test totals can be found in the XML structure.
    """
    if not junit_xml_path.exists():
        raise FileNotFoundError(f"Missing JUnit XML: {junit_xml_path}")

    root = ET.parse(junit_xml_path).getroot()

    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = [suite for suite in root.findall("testsuite")]
    else:
        raise ValueError(f"Unexpected JUnit root element: {root.tag}")

    if not suites:
        raise ValueError("JUnit XML did not contain any testsuite elements")

    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)

    return PytestSummary(
        tests=tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
        junit_xml_path=str(junit_xml_path),
    )


def _artifact_record(path: Path, repo_root: Path) -> dict[str, str | int]:
    """Build a report record for one file artifact."""
    return {
        "path": str(path.relative_to(repo_root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256_of_file(path),
    }


def main() -> int:
    """Execute quality gates and emit evidence artifacts for verification.

    Returns:
        Exit code `0` when all checks pass with valid evidence; otherwise `1`.
    """
    _configure_logging()
    repo_root = Path(__file__).resolve().parent.parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = repo_root / "artifacts" / "quality" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    run_started_at = _utc_iso_now()
    junit_xml_path = output_dir / "pytest.junit.xml"

    steps = [
        _run_step(
            name="ruff",
            command=["uv", "run", "--extra", "dev", "ruff", "check", *CORE_SCOPE_PATHS],
            cwd=repo_root,
            output_dir=output_dir,
        ),
        _run_step(
            name="mypy",
            command=["uv", "run", "--extra", "dev", "mypy", *CORE_SCOPE_PATHS],
            cwd=repo_root,
            output_dir=output_dir,
        ),
        _run_step(
            name="pytest",
            command=[
                "uv",
                "run",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "tests",
                "-q",
                "--junitxml",
                str(junit_xml_path),
            ],
            cwd=repo_root,
            output_dir=output_dir,
        ),
    ]

    evidence_errors: list[str] = []
    pytest_summary: PytestSummary | None = None
    try:
        pytest_summary = _parse_junit(junit_xml_path)
        if pytest_summary.tests <= 0:
            evidence_errors.append("Pytest JUnit XML reported zero tests")
        if pytest_summary.failures > 0:
            evidence_errors.append("Pytest JUnit XML reported test failures")
        if pytest_summary.errors > 0:
            evidence_errors.append("Pytest JUnit XML reported test errors")
    except Exception as error:  # noqa: BLE001
        evidence_errors.append(f"Pytest evidence invalid: {error}")

    command_failures = [step for step in steps if step.return_code != 0]
    all_passed = (not command_failures) and (not evidence_errors)

    artifact_paths: list[Path] = []
    for step in steps:
        artifact_paths.append(Path(step.stdout_path))
        artifact_paths.append(Path(step.stderr_path))
    if junit_xml_path.exists():
        artifact_paths.append(junit_xml_path)

    report = {
        "run_id": run_id,
        "status": "pass" if all_passed else "fail",
        "started_at_utc": run_started_at,
        "finished_at_utc": _utc_iso_now(),
        "scope_paths": list(CORE_SCOPE_PATHS),
        "steps": [asdict(step) for step in steps],
        "pytest_summary": asdict(pytest_summary) if pytest_summary else None,
        "command_failures": [step.name for step in command_failures],
        "evidence_errors": evidence_errors,
        "artifacts": [_artifact_record(path, repo_root) for path in artifact_paths if path.exists()],
    }

    report_json_path = output_dir / "report.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_hash = _sha256_of_file(report_json_path)

    report_md_path = output_dir / "report.md"
    report_md_path.write_text(
        "\n".join(
            [
                f"# Quality Gate Report: {run_id}",
                "",
                f"- status: `{report['status']}`",
                f"- report_json: `{report_json_path.relative_to(repo_root)}`",
                f"- report_json_sha256: `{report_hash}`",
                f"- evidence_errors: `{len(evidence_errors)}`",
                f"- command_failures: `{len(command_failures)}`",
                "",
                "## Steps",
                *(f"- `{step.name}`: exit `{step.return_code}`" for step in steps),
                "",
                "## Pytest Totals",
                (
                    f"- tests={pytest_summary.tests}, failures={pytest_summary.failures}, "
                    f"errors={pytest_summary.errors}, skipped={pytest_summary.skipped}"
                    if pytest_summary
                    else "- unavailable (invalid or missing JUnit XML)"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Quality report written: %s", report_json_path)
    LOGGER.info("Quality report sha256: %s", report_hash)
    if all_passed:
        LOGGER.info("All quality checks passed with verified evidence.")
        return 0

    LOGGER.error("Quality checks failed. See evidence in %s", output_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
