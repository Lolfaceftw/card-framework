"""Coverage for the SPoRC manifest audio-url filter helper."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

from card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls import (
    AudioUrlProbeResult,
    _probe_source_audio_target,
    filter_sporc_manifest_audio_urls,
)


def _write_inline_manifest(path: Path, *, samples: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(path.parent / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(path.parent / "speakerTurnData.jsonl.gz"),
                "sample_count": len(samples),
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_compact_manifest(
    path: Path,
    *,
    samples: list[dict[str, object]],
) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples_path = path.parent / "samples.jsonl.gz"
    with gzip.open(samples_path, "wt", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample))
            handle.write("\n")
    path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(path.parent / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(path.parent / "speakerTurnData.jsonl.gz"),
                "sample_count": len(samples),
                "samples_path": str(samples_path),
            }
        ),
        encoding="utf-8",
    )
    return path, samples_path


def _sample(
    sample_id: str,
    *,
    episode_key: str,
    source_audio_url: str = "",
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if source_audio_url:
        metadata["source_audio_url"] = source_audio_url
    return {
        "sample_id": sample_id,
        "dataset": "sporc_full",
        "episode_key": episode_key,
        "transcript_path": f"{sample_id}.transcript.json",
        "reference_summary": f"Summary for {sample_id}.",
        "target_seconds": 45,
        "metadata": metadata,
    }


def test_filter_sporc_manifest_audio_urls_rewrites_inline_manifest_and_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_inline_manifest(
        tmp_path / "benchmark_longform_15k_manifest.json",
        samples=[
            _sample(
                "keep_metadata",
                episode_key="https://example.com/fallback-a.mp3",
                source_audio_url="https://example.com/keep-a.mp3",
            ),
            _sample(
                "remove_missing",
                episode_key="https://example.com/remove-b.mp3",
                source_audio_url="",
            ),
            _sample(
                "keep_episode_key",
                episode_key="https://example.com/keep-c.mp3",
                source_audio_url="",
            ),
        ],
    )
    audit_path = tmp_path / "benchmark_longform_15k_manifest.audio_url_audit.json"

    probe_results = {
        "https://example.com/keep-a.mp3": AudioUrlProbeResult(
            is_valid=True,
            url="https://example.com/keep-a.mp3",
            reason="reachable",
            method="HEAD",
            status_code=200,
        ),
        "https://example.com/remove-b.mp3": AudioUrlProbeResult(
            is_valid=False,
            url="https://example.com/remove-b.mp3",
            reason="http_404",
            method="GET",
            status_code=404,
        ),
        "https://example.com/keep-c.mp3": AudioUrlProbeResult(
            is_valid=True,
            url="https://example.com/keep-c.mp3",
            reason="reachable",
            method="GET",
            status_code=206,
        ),
    }

    def _fake_probe(url: str, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return probe_results[url]

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls._probe_source_audio_target",
        _fake_probe,
    )

    result = filter_sporc_manifest_audio_urls(
        manifest_path=manifest_path,
        audit_path=audit_path,
    )

    assert result.manifest_path == manifest_path.resolve()
    assert result.audit_path == audit_path.resolve()
    assert result.original_sample_count == 3
    assert result.kept_sample_count == 2
    assert result.removed_sample_count == 1

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 2
    assert [sample["sample_id"] for sample in payload["samples"]] == [
        "keep_metadata",
        "keep_episode_key",
    ]
    assert payload["audio_url_filter"]["removed_sample_count"] == 1

    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_payload["removed_sample_count"] == 1
    assert audit_payload["removed_samples"][0]["sample_id"] == "remove_missing"
    assert audit_payload["removed_samples"][0]["reason"] == "http_404"


def test_filter_sporc_manifest_audio_urls_preserves_compact_manifest_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, original_samples_path = _write_compact_manifest(
        tmp_path / "benchmark_longform_15k_manifest.compact.json",
        samples=[
            _sample(
                "keep_sidecar",
                episode_key="https://example.com/keep-sidecar.mp3",
            ),
            _sample(
                "remove_sidecar",
                episode_key="https://example.com/remove-sidecar.mp3",
            ),
        ],
    )

    def _fake_probe(url: str, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return AudioUrlProbeResult(
            is_valid=url.endswith("keep-sidecar.mp3"),
            url=url,
            reason="reachable" if url.endswith("keep-sidecar.mp3") else "http_404",
            method="GET",
            status_code=206 if url.endswith("keep-sidecar.mp3") else 404,
        )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls._probe_source_audio_target",
        _fake_probe,
    )

    result = filter_sporc_manifest_audio_urls(manifest_path=manifest_path)

    assert result.manifest_path == manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "samples" not in payload
    assert payload["sample_count"] == 1
    assert payload["samples_path"] == str(original_samples_path.resolve())

    with gzip.open(original_samples_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert [row["sample_id"] for row in rows] == ["keep_sidecar"]


def test_filter_sporc_manifest_audio_urls_streams_compact_sidecar_without_materializing_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, original_samples_path = _write_compact_manifest(
        tmp_path / "benchmark_longform_15k_manifest.compact.json",
        samples=[
            _sample("keep_streamed", episode_key="https://example.com/keep-streamed.mp3"),
            _sample(
                "remove_streamed",
                episode_key="https://example.com/remove-streamed.mp3",
            ),
        ],
    )

    def _fake_probe(url: str, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return AudioUrlProbeResult(
            is_valid=url.endswith("keep-streamed.mp3"),
            url=url,
            reason="reachable" if url.endswith("keep-streamed.mp3") else "http_404",
            method="GET",
            status_code=206 if url.endswith("keep-streamed.mp3") else 404,
        )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls._probe_source_audio_target",
        _fake_probe,
    )

    result = filter_sporc_manifest_audio_urls(manifest_path=manifest_path)

    assert result.original_sample_count == 2
    assert result.kept_sample_count == 1
    with gzip.open(original_samples_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert [row["sample_id"] for row in rows] == ["keep_streamed"]


def test_filter_sporc_manifest_audio_urls_reports_validation_progress_and_eta(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_inline_manifest(
        tmp_path / "benchmark_longform_15k_manifest.json",
        samples=[
            _sample(
                "keep_progress",
                episode_key="https://example.com/keep-progress.mp3",
            ),
            _sample(
                "remove_progress",
                episode_key="https://example.com/remove-progress.mp3",
            ),
        ],
    )
    status_messages: list[str] = []

    def _fake_probe(url: str, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return AudioUrlProbeResult(
            is_valid=url.endswith("keep-progress.mp3"),
            url=url,
            reason="reachable" if url.endswith("keep-progress.mp3") else "http_404",
            method="GET",
            status_code=206 if url.endswith("keep-progress.mp3") else 404,
        )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls._probe_source_audio_target",
        _fake_probe,
    )

    filter_sporc_manifest_audio_urls(
        manifest_path=manifest_path,
        status_reporter=status_messages.append,
    )

    progress_messages = [
        message
        for message in status_messages
        if "validated" in message and "rolling_avg=" in message and "eta=" in message
    ]
    assert progress_messages
    assert any("validated 0 / 2 rows" in message for message in progress_messages)
    assert any("validated 2 / 2 rows" in message for message in progress_messages)
    assert any("[" in message and "]" in message for message in progress_messages)


def test_filter_sporc_manifest_audio_urls_reports_progress_through_all_valid_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_inline_manifest(
        tmp_path / "benchmark_longform_15k_manifest.json",
        samples=[
            _sample(f"valid_{index}", episode_key=f"https://example.com/{index}.mp3")
            for index in range(30)
        ],
    )
    status_messages: list[str] = []

    def _fake_probe(url: str, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return AudioUrlProbeResult(
            is_valid=True,
            url=url,
            reason="reachable",
            method="HEAD",
            status_code=200,
        )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls._probe_source_audio_target",
        _fake_probe,
    )

    filter_sporc_manifest_audio_urls(
        manifest_path=manifest_path,
        status_reporter=status_messages.append,
        max_workers=4,
    )

    progress_messages = [
        message
        for message in status_messages
        if "validated" in message and "rolling_avg=" in message and "eta=" in message
    ]
    assert any("validated 25 / 30 rows" in message for message in progress_messages)
    assert any("validated 30 / 30 rows" in message for message in progress_messages)


def test_filter_sporc_manifest_audio_urls_limits_remote_probe_concurrency_per_host(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_inline_manifest(
        tmp_path / "benchmark_longform_15k_manifest.json",
        samples=[
            _sample("same_a", episode_key="https://same.example.com/a.mp3"),
            _sample("same_b", episode_key="https://same.example.com/b.mp3"),
            _sample("same_c", episode_key="https://same.example.com/c.mp3"),
            _sample("other_a", episode_key="https://other.example.com/a.mp3"),
        ],
    )
    lock = threading.Lock()
    active_total = 0
    max_active_total = 0
    active_by_host: dict[str, int] = {}
    max_active_by_host: dict[str, int] = {}

    def _fake_probe(url: str, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        nonlocal active_total, max_active_total
        host = urlparse(url).netloc
        with lock:
            active_total += 1
            max_active_total = max(max_active_total, active_total)
            active_by_host[host] = active_by_host.get(host, 0) + 1
            max_active_by_host[host] = max(
                max_active_by_host.get(host, 0),
                active_by_host[host],
            )
        time.sleep(0.05)
        with lock:
            active_total -= 1
            active_by_host[host] -= 1
        return AudioUrlProbeResult(
            is_valid=True,
            url=url,
            reason="reachable",
            method="HEAD",
            status_code=200,
        )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls._probe_source_audio_target",
        _fake_probe,
    )

    result = filter_sporc_manifest_audio_urls(
        manifest_path=manifest_path,
        max_workers=4,
        max_workers_per_host=2,
    )

    assert result.kept_sample_count == 4
    assert max_active_total >= 2
    assert max_active_by_host["same.example.com"] <= 2
    assert max_active_by_host["other.example.com"] <= 1


def test_filter_sporc_manifest_audio_urls_caches_duplicate_url_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    shared_url = "https://cache.example.com/shared.mp3"
    manifest_path = _write_inline_manifest(
        tmp_path / "benchmark_longform_15k_manifest.json",
        samples=[
            _sample("dup_a", episode_key=shared_url),
            _sample("dup_b", episode_key=shared_url),
            _sample("dup_c", episode_key=shared_url),
        ],
    )
    call_count = 0
    call_lock = threading.Lock()

    def _fake_probe(url: str, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        nonlocal call_count
        with call_lock:
            call_count += 1
        time.sleep(0.02)
        return AudioUrlProbeResult(
            is_valid=True,
            url=url,
            reason="reachable",
            method="HEAD",
            status_code=200,
        )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls._probe_source_audio_target",
        _fake_probe,
    )

    result = filter_sporc_manifest_audio_urls(
        manifest_path=manifest_path,
        max_workers=3,
    )

    assert result.kept_sample_count == 3
    assert call_count == 1


def test_probe_source_audio_target_retries_transient_failure(
    monkeypatch,
) -> None:
    sleep_calls: list[float] = []

    class _Response:
        def __init__(self, status_code: int, url: str) -> None:
            self.status_code = status_code
            self.url = url

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def close(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 1):  # type: ignore[no-untyped-def]
            del chunk_size
            yield b"0"

    class _Session:
        def __init__(self) -> None:
            self.get_calls = 0

        def head(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            return _Response(405, "https://example.com/audio.mp3")

        def get(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            self.get_calls += 1
            if self.get_calls == 1:
                return _Response(503, "https://example.com/audio.mp3")
            return _Response(206, "https://example.com/audio.mp3")

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls.time.sleep",
        sleep_calls.append,
    )

    session = _Session()
    probe = _probe_source_audio_target(
        "https://example.com/audio.mp3",
        session=session,
        timeout_seconds=5.0,
        max_retries=1,
        retry_backoff_seconds=0.25,
    )

    assert probe.is_valid is True
    assert probe.method == "GET"
    assert probe.status_code == 206
    assert session.get_calls == 2
    assert sleep_calls == [0.25]


def test_filter_sporc_manifest_audio_urls_retries_transient_probe_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_inline_manifest(
        tmp_path / "benchmark_longform_15k_manifest.json",
        samples=[_sample("retry_me", episode_key="https://example.com/retry.mp3")],
    )
    probe_calls = 0

    def _fake_probe(url: str, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        nonlocal probe_calls
        probe_calls += 1
        if probe_calls == 1:
            return AudioUrlProbeResult(
                is_valid=False,
                url=url,
                reason="TimeoutError:simulated",
                method="GET",
                status_code=503,
            )
        return AudioUrlProbeResult(
            is_valid=True,
            url=url,
            reason="reachable",
            method="GET",
            status_code=206,
        )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls._probe_source_audio_target",
        _fake_probe,
    )

    result = filter_sporc_manifest_audio_urls(
        manifest_path=manifest_path,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )

    assert result.kept_sample_count == 1
    assert probe_calls == 2


def test_filter_sporc_manifest_audio_urls_caches_duplicate_url_probes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_inline_manifest(
        tmp_path / "benchmark_longform_15k_manifest.json",
        samples=[
            _sample("dup_a", episode_key="https://example.com/same.mp3"),
            _sample("dup_b", episode_key="https://example.com/same.mp3"),
            _sample("dup_c", episode_key="https://example.com/same.mp3"),
        ],
    )
    probe_calls = 0

    def _fake_probe(url: str, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        nonlocal probe_calls
        probe_calls += 1
        return AudioUrlProbeResult(
            is_valid=True,
            url=url,
            reason="reachable",
            method="HEAD",
            status_code=200,
        )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls._probe_source_audio_target",
        _fake_probe,
    )

    result = filter_sporc_manifest_audio_urls(
        manifest_path=manifest_path,
        max_workers=3,
        max_workers_per_host=3,
    )

    assert result.kept_sample_count == 3
    assert probe_calls == 1


def test_filter_sporc_manifest_audio_urls_drops_stale_samples_path_when_inline_samples_win(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_inline_manifest(
        tmp_path / "benchmark_longform_15k_manifest.json",
        samples=[
            _sample(
                "keep_inline",
                episode_key="https://example.com/keep-inline.mp3",
            ),
            _sample(
                "remove_inline",
                episode_key="https://example.com/remove-inline.mp3",
            ),
        ],
    )
    stale_samples_path = tmp_path / "stale.samples.jsonl.gz"
    with gzip.open(stale_samples_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(_sample("stale_row", episode_key="episode://stale")))
        handle.write("\n")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["samples_path"] = str(stale_samples_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    def _fake_probe(url: str, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return AudioUrlProbeResult(
            is_valid=url.endswith("keep-inline.mp3"),
            url=url,
            reason="reachable" if url.endswith("keep-inline.mp3") else "http_404",
            method="GET",
            status_code=206 if url.endswith("keep-inline.mp3") else 404,
        )

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.filter_manifest_audio_urls._probe_source_audio_target",
        _fake_probe,
    )

    filter_sporc_manifest_audio_urls(manifest_path=manifest_path)

    rewritten_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "samples_path" not in rewritten_payload
    assert [sample["sample_id"] for sample in rewritten_payload["samples"]] == [
        "keep_inline",
    ]


def test_probe_source_audio_target_falls_back_from_head_to_streamed_get() -> None:
    class _Response:
        def __init__(self, status_code: int, url: str) -> None:
            self.status_code = status_code
            self.url = url

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def close(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 1):  # type: ignore[no-untyped-def]
            del chunk_size
            yield b"0"

    class _Session:
        def head(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            return _Response(405, "https://example.com/audio.mp3")

        def get(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            return _Response(206, "https://example.com/audio.mp3")

    probe = _probe_source_audio_target(
        "https://example.com/audio.mp3",
        session=_Session(),
        timeout_seconds=5.0,
    )

    assert probe.is_valid is True
    assert probe.method == "GET"
    assert probe.status_code == 206
