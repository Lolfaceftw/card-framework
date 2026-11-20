"""Tests for the readiness checks shared by infer() and `card-framework doctor`."""

from __future__ import annotations

import pytest

from card_framework.runtime.preflight import (
    VLLM_PROVIDER_TARGET,
    NotReadyError,
    compute_check,
    llm_endpoint_checks,
    speaker_check,
    storage_check,
)


def _vllm(base_url: str) -> dict[str, str]:
    return {"_target_": VLLM_PROVIDER_TARGET, "base_url": base_url, "api_key": "EMPTY"}


@pytest.mark.parametrize(
    ("status_code", "expected_status", "detail_fragment"),
    [
        (200, "ok", "answering"),
        (401, "fail", "rejected the API key"),
        (404, "fail", "not an OpenAI-compatible API"),
        (None, "fail", "nothing is answering"),
    ],
)
def test_llm_endpoint_checks_report_each_probe_outcome(
    status_code: int | None,
    expected_status: str,
    detail_fragment: str,
) -> None:
    """Turn each kind of probe result into a check a person can act on."""
    config = {"llm": _vllm("http://localhost:8000/v1")}

    checks = llm_endpoint_checks(config, probe=lambda url, key: (status_code, "refused"))

    assert len(checks) == 1
    assert checks[0].status == expected_status
    assert detail_fragment in checks[0].detail
    if expected_status == "fail":
        assert checks[0].fix


def test_llm_endpoint_checks_probe_each_url_once_and_skip_api_providers() -> None:
    """Probe shared vLLM URLs once and leave API providers to the credential step."""
    config = {
        "llm": _vllm("http://localhost:8000/v1/"),
        "stage_llm": {
            "summarizer": _vllm("http://localhost:8000/v1"),
            "critic": {"_target_": "card_framework.providers.deepseek_provider.DeepSeekProvider"},
            "interjector": {},
        },
    }
    probed: list[str] = []

    def _probe(url: str, key: str) -> tuple[int | None, str]:
        probed.append(url)
        return 200, ""

    checks = llm_endpoint_checks(config, probe=_probe)

    assert probed == ["http://localhost:8000/v1"]
    assert [check.status for check in checks] == ["ok"]


def test_not_ready_error_lists_every_problem_and_fix() -> None:
    """Explain all blocking problems at once instead of one per attempt."""
    checks = llm_endpoint_checks(
        {"llm": _vllm("http://localhost:8000/v1")},
        probe=lambda url, key: (None, "connection refused"),
    )

    error = NotReadyError(checks)

    assert isinstance(error, RuntimeError)
    assert "nothing is answering at http://localhost:8000/v1" in str(error)
    assert "Fix:" in str(error)
    assert "card-framework doctor" in str(error)


def test_speaker_check_warns_when_the_run_falls_back_to_one_speaker() -> None:
    """Pass the pipeline's own fallback reason through as a warning."""
    reason = "Configured audio.diarization.provider='nemo' is not supported on Windows x86_64."

    fallback = speaker_check("nemo", reason)

    assert fallback.status == "warn"
    assert "one speaker" in fallback.detail and reason in fallback.detail
    assert speaker_check("nemo", None).status == "ok"


def test_compute_and_storage_checks_warn_without_blocking() -> None:
    """Treat a missing GPU or a small disk as warnings, never as failures."""
    assert compute_check("NVIDIA GeForce RTX 3060", "12.6").status == "ok"
    assert compute_check(None, None).status == "warn"
    assert storage_check(50_000_000_000, "/models").status == "ok"
    assert storage_check(5_000_000_000, "/models").status == "warn"
