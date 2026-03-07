from __future__ import annotations

from pathlib import Path

import setup_and_run as bootstrap


def test_resolve_repo_config_boolean_reads_nested_enabled_value(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    conf_dir = repo_root / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "config.yaml").write_text(
        "audio:\n"
        "  interjector:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    assert (
        bootstrap.resolve_repo_config_boolean(
            key_path=("audio", "interjector", "enabled"),
            fallback=False,
        )
        is True
    )


def test_resolve_repo_backed_boolean_override_uses_repo_default_when_cli_absent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    conf_dir = repo_root / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "config.yaml").write_text(
        "audio:\n"
        "  interjector:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    assert (
        bootstrap.resolve_repo_backed_boolean_override(
            [],
            key="audio.interjector.enabled",
            repo_key_path=("audio", "interjector", "enabled"),
            fallback=False,
        )
        is True
    )


def test_resolve_repo_backed_boolean_override_prefers_cli_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    conf_dir = repo_root / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "config.yaml").write_text(
        "audio:\n"
        "  interjector:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(bootstrap, "REPO_ROOT", repo_root)

    assert (
        bootstrap.resolve_repo_backed_boolean_override(
            ["audio.interjector.enabled=false"],
            key="audio.interjector.enabled",
            repo_key_path=("audio", "interjector", "enabled"),
            fallback=False,
        )
        is False
    )
