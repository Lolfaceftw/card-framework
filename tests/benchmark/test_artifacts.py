from __future__ import annotations

from pathlib import Path

from card_framework.benchmark.artifacts import sha256_file, write_json_with_hash


def test_write_json_with_hash_matches_file_digest(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    payload = {"hello": "world", "count": 3}

    reported_hash = write_json_with_hash(target, payload)

    assert target.exists()
    assert len(reported_hash) == 64
    assert reported_hash == sha256_file(target)

