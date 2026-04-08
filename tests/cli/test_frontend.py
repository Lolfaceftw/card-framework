"""Coverage for the Textual frontend's pure projection helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
from textual.widgets import Input, OptionList, Select, TextArea

from card_framework.cli.textual_frontend import SetupAndRunFrontendApp
from card_framework.cli.frontend_services import (
    FrontendValidationError,
    PipelineProfileDefaults,
    build_benchmark_request,
    build_pipeline_selection,
    compute_path_suggestions,
    load_frontend_defaults,
    save_config_text,
)


def _pipeline_defaults() -> PipelineProfileDefaults:
    return PipelineProfileDefaults(
        llm="vllm_local",
        embedding="disabled",
        stage_llm={
            "summarizer": "default",
            "critic": "default",
            "interjector": "default",
        },
        voice_clone_enabled=True,
        interjector_enabled=False,
    )


def test_build_pipeline_selection_supports_stage4_interjector() -> None:
    selection = build_pipeline_selection(
        defaults=_pipeline_defaults(),
        values={
            "mode": "stage4_interjector",
            "summary_path": "summary.xml",
            "voice_clone_manifest_path": "artifacts/audio_stage/voice_clone/manifest.json",
            "output_root": "frontend_outputs",
            "shared_llm": "vllm_local",
            "embedding": "disabled",
            "stage_strategy": "config_default",
            "voice_clone_enabled": True,
            "interjector_enabled": True,
            "extra_overrides": "audio.interjector.enabled=true",
        },
    )

    assert selection.mode == "stage4_interjector"
    assert selection.voiceclone_from_summary == "summary.xml"
    assert (
        selection.voice_clone_manifest_path
        == "artifacts/audio_stage/voice_clone/manifest.json"
    )
    assert selection.output_root == "frontend_outputs"
    assert "audio.interjector.enabled=true" in selection.overrides


def test_build_pipeline_selection_accepts_stage1_audio_paths() -> None:
    selection = build_pipeline_selection(
        defaults=_pipeline_defaults(),
        values={
            "mode": "stage1_single",
            "primary_audio_path": "audio.wav",
            "shared_llm": "vllm_local",
            "embedding": "disabled",
            "stage_strategy": "config_default",
            "voice_clone_enabled": True,
            "interjector_enabled": False,
        },
    )

    assert selection.mode == "stage1_single"
    assert selection.audio_paths == ("audio.wav",)


def test_build_pipeline_selection_combines_primary_and_additional_batch_audio() -> None:
    selection = build_pipeline_selection(
        defaults=_pipeline_defaults(),
        values={
            "mode": "stage1_batch",
            "primary_audio_path": "episode-a.mp3",
            "additional_audio_paths": "episode-b.wav\nepisode-c.mp3",
            "shared_llm": "vllm_local",
            "embedding": "disabled",
            "stage_strategy": "config_default",
            "voice_clone_enabled": True,
            "interjector_enabled": False,
        },
    )

    assert selection.audio_paths == (
        "episode-a.mp3",
        "episode-b.wav",
        "episode-c.mp3",
    )


def test_build_pipeline_selection_rejects_missing_stage4_manifest() -> None:
    with pytest.raises(FrontendValidationError, match="voice-clone manifest path"):
        build_pipeline_selection(
            defaults=_pipeline_defaults(),
            values={
                "mode": "stage4_interjector",
                "summary_path": "summary.xml",
                "shared_llm": "vllm_local",
                "embedding": "disabled",
                "stage_strategy": "config_default",
                "voice_clone_enabled": True,
                "interjector_enabled": False,
            },
        )


def test_build_pipeline_selection_stage3_preserves_runtime_overrides() -> None:
    selection = build_pipeline_selection(
        defaults=_pipeline_defaults(),
        values={
            "mode": "stage3_summary",
            "summary_path": "summary.xml",
            "shared_llm": "deepseek_chat",
            "embedding": "disabled",
            "stage_strategy": "config_default",
            "voice_clone_enabled": True,
            "interjector_enabled": True,
        },
    )

    assert selection.mode == "stage3_summary"
    assert selection.voiceclone_from_summary == "summary.xml"
    assert "selection.llm=deepseek_chat" in selection.overrides
    assert "audio.interjector.enabled=true" in selection.overrides


def test_build_pipeline_selection_stage4_forces_interjector_on_by_default() -> None:
    selection = build_pipeline_selection(
        defaults=_pipeline_defaults(),
        values={
            "mode": "stage4_interjector",
            "summary_path": "summary.xml",
            "voice_clone_manifest_path": "artifacts/audio_stage/voice_clone/manifest.json",
            "shared_llm": "vllm_local",
            "embedding": "disabled",
            "stage_strategy": "config_default",
            "voice_clone_enabled": True,
            "interjector_enabled": False,
        },
    )

    assert "audio.interjector.enabled=true" in selection.overrides


def test_build_pipeline_selection_stage4_replaces_explicit_interjector_disable() -> (
    None
):
    selection = build_pipeline_selection(
        defaults=_pipeline_defaults(),
        values={
            "mode": "stage4_interjector",
            "summary_path": "summary.xml",
            "voice_clone_manifest_path": "artifacts/audio_stage/voice_clone/manifest.json",
            "shared_llm": "vllm_local",
            "embedding": "disabled",
            "stage_strategy": "config_default",
            "voice_clone_enabled": True,
            "interjector_enabled": False,
            "extra_overrides": "audio.interjector.enabled=false",
        },
    )

    assert "audio.interjector.enabled=true" in selection.overrides
    assert "audio.interjector.enabled=false" not in selection.overrides


def test_build_pipeline_selection_rejects_conflicting_stage4_pipeline_override() -> (
    None
):
    with pytest.raises(FrontendValidationError, match="conflict"):
        build_pipeline_selection(
            defaults=_pipeline_defaults(),
            values={
                "mode": "stage4_interjector",
                "summary_path": "summary.xml",
                "voice_clone_manifest_path": "artifacts/audio_stage/voice_clone/manifest.json",
                "shared_llm": "vllm_local",
                "embedding": "disabled",
                "stage_strategy": "config_default",
                "voice_clone_enabled": True,
                "interjector_enabled": False,
                "extra_overrides": "pipeline.start_stage=stage-2",
            },
        )


def test_build_benchmark_request_for_sporc_duration_sweep() -> None:
    request = build_benchmark_request(
        kind="sporc_duration_sweep",
        values={
            "sporc_dataset_manifest": "datasets/sporc/prepared_sample/manifest.json",
            "sporc_dataset_variant": "sample",
            "sporc_sweep_root": "artifacts/sweeps/frontend",
            "sporc_sweep_mode": "batch-all",
            "sporc_sweep_duration_presets": "30s,1m,5m",
            "sporc_sweep_examples": "12",
            "sporc_sweep_parallelism": "0",
            "sporc_sweep_concurrency": "3",
            "sporc_sweep_enable_voice_clone": True,
        },
        uv_executable="uv",
    )

    assert request.kind == "sporc_duration_sweep"
    assert request.command[:6] == (
        "uv",
        "run",
        "python",
        "-m",
        "card_framework.benchmark.summarizer_critic.sporc",
        "duration-sweep",
    )
    assert "--enable-voice-clone" in request.command
    assert "--concurrency" in request.command
    assert "3" in request.command


def test_save_config_text_validates_yaml_before_write(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"

    save_config_text(
        config_path=config_path, config_text="selection:\n  llm: vllm_local\n"
    )

    assert config_path.read_text(encoding="utf-8") == "selection:\n  llm: vllm_local\n"

    with pytest.raises(FrontendValidationError, match="valid YAML"):
        save_config_text(config_path=config_path, config_text="selection: [oops")


def test_compute_path_suggestions_skips_blank_values_without_scanning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fail_scandir(_path: str) -> Any:
        raise AssertionError("blank suggestions should not scan directories")

    monkeypatch.setattr(os, "scandir", _fail_scandir)

    assert (
        compute_path_suggestions(
            raw_value="",
            workspace_root=tmp_path,
            allowed_suffixes=(".wav", ".mp3"),
        )
        == ()
    )


@pytest.mark.anyio
async def test_textual_frontend_save_config_reloads_pipeline_defaults(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "profiles:\n"
        "  llm:\n"
        "    vllm_local: {}\n"
        "    anthropic_claude_opus: {}\n"
        "  embedding:\n"
        "    disabled: {}\n"
        "selection:\n"
        "  llm: vllm_local\n"
        "  embedding: disabled\n"
        "  stage_llm:\n"
        "    summarizer: default\n"
        "    critic: default\n"
        "    interjector: default\n"
        "audio:\n"
        "  voice_clone:\n"
        "    enabled: true\n"
        "  interjector:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        editor = frontend_app.query_one("#config_editor", TextArea)
        editor.text = (
            config_path.read_text(encoding="utf-8")
            .replace(
                "    anthropic_claude_opus: {}\n",
                "    anthropic_claude_opus: {}\n    deepseek_chat: {}\n",
            )
            .replace(
                "selection:\n  llm: vllm_local",
                "selection:\n  llm: deepseek_chat",
            )
        )
        await pilot.pause()
        frontend_app._save_config()
        await pilot.pause()

        shared_llm = frontend_app.query_one("#pipeline_shared_llm", Select)
        assert str(shared_llm.value) == "deepseek_chat"


def test_compute_path_suggestions_filters_audio_files(tmp_path: Path) -> None:
    (tmp_path / "audio.wav").write_bytes(b"wav")
    (tmp_path / "episode.flac").write_bytes(b"flac")
    (tmp_path / "another.mp3").write_bytes(b"mp3")
    (tmp_path / "notes.txt").write_text("nope", encoding="utf-8")

    suggestions = compute_path_suggestions(
        raw_value=f"{tmp_path}{os.sep}",
        workspace_root=tmp_path,
        allowed_suffixes=(".wav", ".flac", ".mp3"),
    )

    assert str(tmp_path / "audio.wav").replace("\\", "/") in suggestions
    assert str(tmp_path / "another.mp3").replace("\\", "/") in suggestions
    assert str(tmp_path / "episode.flac").replace("\\", "/") in suggestions
    assert not any(path.endswith("notes.txt") for path in suggestions)


def test_compute_path_suggestions_hides_dropdown_for_exact_existing_file(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text("{}\n", encoding="utf-8")

    suggestions = compute_path_suggestions(
        raw_value=str(transcript_path),
        workspace_root=tmp_path,
        allowed_suffixes=(".json",),
    )

    assert suggestions == ()


def test_compute_path_suggestions_does_not_escape_workspace_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "inside.json").write_text("{}\n", encoding="utf-8")
    outside_dir = tmp_path.parent / f"{tmp_path.name}-outside"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "outside.json").write_text("{}\n", encoding="utf-8")

    suggestions = compute_path_suggestions(
        raw_value=f"..{os.sep}out",
        workspace_root=tmp_path,
        allowed_suffixes=(".json",),
    )

    assert str(tmp_path / "inside.json").replace("\\", "/") in suggestions
    assert not any(path.endswith("outside.json") for path in suggestions)


@pytest.mark.anyio
async def test_textual_frontend_audio_input_shows_path_suggestions(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "profiles:\n"
        "  llm:\n"
        "    vllm_local: {}\n"
        "  embedding:\n"
        "    disabled: {}\n"
        "selection:\n"
        "  llm: vllm_local\n"
        "  embedding: disabled\n"
        "  stage_llm:\n"
        "    summarizer: default\n"
        "    critic: default\n"
        "    interjector: default\n"
        "audio:\n"
        "  voice_clone:\n"
        "    enabled: true\n"
        "  interjector:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")

    (tmp_path / "picked.wav").write_bytes(b"wav")
    (tmp_path / "picked.flac").write_bytes(b"flac")
    (tmp_path / "picked.mp3").write_bytes(b"mp3")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        audio_input = frontend_app.query_one("#pipeline_primary_audio_path", Input)
        audio_input.value = str(tmp_path / "pic")
        frontend_app._refresh_path_suggestions_for("pipeline_primary_audio_path")
        await pilot.pause()

        suggestions = frontend_app.query_one(
            "#pipeline_primary_audio_suggestions", OptionList
        )
        prompts = [str(option.prompt) for option in suggestions.options]
        assert str(tmp_path / "picked.wav").replace("\\", "/") in prompts
        assert str(tmp_path / "picked.flac").replace("\\", "/") in prompts
        assert str(tmp_path / "picked.mp3").replace("\\", "/") in prompts
        assert not any(prompt.endswith("ignore.txt") for prompt in prompts)


@pytest.mark.anyio
async def test_textual_frontend_pipeline_transcript_input_shows_json_suggestions(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "selection:\n  llm: vllm_local\n",
        encoding="utf-8",
    )
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")

    (tmp_path / "transcript.json").write_text("{}\n", encoding="utf-8")

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        frontend_app.query_one("#pipeline_mode", Select).value = "stage2_transcript"
        await pilot.pause()
        frontend_app.query_one("#pipeline_transcript_path", Input).value = str(
            tmp_path / "tra"
        )
        frontend_app._refresh_path_suggestions_for("pipeline_transcript_path")
        await pilot.pause()

        suggestions = frontend_app.query_one(
            "#pipeline_transcript_suggestions", OptionList
        )
        prompts = [str(option.prompt) for option in suggestions.options]
        assert str(tmp_path / "transcript.json").replace("\\", "/") in prompts


@pytest.mark.anyio
async def test_textual_frontend_uses_frontend_workspace_root_not_process_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("selection:\n  llm: vllm_local\n", encoding="utf-8")
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")
    (tmp_path / "transcript.json").write_text("{}\n", encoding="utf-8")
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        frontend_app.query_one("#pipeline_mode", Select).value = "stage2_transcript"
        await pilot.pause()
        frontend_app.query_one("#pipeline_transcript_path", Input).value = "tra"
        frontend_app._refresh_path_suggestions_for("pipeline_transcript_path")
        await pilot.pause()

        suggestions = frontend_app.query_one(
            "#pipeline_transcript_suggestions", OptionList
        )
        prompts = [str(option.prompt) for option in suggestions.options]
        assert str(tmp_path / "transcript.json").replace("\\", "/") in prompts


@pytest.mark.anyio
async def test_textual_frontend_exact_pipeline_transcript_path_hides_dropdown(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("selection:\n  llm: vllm_local\n", encoding="utf-8")
    transcript_path = (tmp_path / "transcript.json").resolve()
    transcript_path.write_text("{}\n", encoding="utf-8")
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        frontend_app.query_one("#pipeline_mode", Select).value = "stage2_transcript"
        await pilot.pause()
        frontend_app.query_one("#pipeline_transcript_path", Input).value = str(
            transcript_path
        )
        await pilot.pause()

        suggestions = frontend_app.query_one(
            "#pipeline_transcript_suggestions", OptionList
        )
        assert suggestions.display is False
        assert len(suggestions.options) == 0


@pytest.mark.anyio
async def test_textual_frontend_stage3_summary_input_shows_xml_suggestions(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("selection:\n  llm: vllm_local\n", encoding="utf-8")
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")
    (tmp_path / "summary.xml").write_text("<summary />\n", encoding="utf-8")

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        frontend_app.query_one("#pipeline_mode", Select).value = "stage3_summary"
        await pilot.pause()
        frontend_app.query_one("#pipeline_summary_path", Input).value = str(
            tmp_path / "sum"
        )
        frontend_app._refresh_path_suggestions_for("pipeline_summary_path")
        await pilot.pause()

        suggestions = frontend_app.query_one(
            "#pipeline_summary_suggestions", OptionList
        )
        prompts = [str(option.prompt) for option in suggestions.options]
        assert str(tmp_path / "summary.xml").replace("\\", "/") in prompts


@pytest.mark.anyio
async def test_textual_frontend_stage4_manifest_input_shows_json_suggestions(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("selection:\n  llm: vllm_local\n", encoding="utf-8")
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")
    manifest_path = (
        tmp_path / "artifacts" / "audio_stage" / "voice_clone" / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"turns": []}\n', encoding="utf-8")

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        frontend_app.query_one("#pipeline_mode", Select).value = "stage4_interjector"
        await pilot.pause()
        frontend_app.query_one(
            "#pipeline_voice_clone_manifest_path", Input
        ).value = str(manifest_path.parent / "man")
        frontend_app._refresh_path_suggestions_for("pipeline_voice_clone_manifest_path")
        await pilot.pause()

        suggestions = frontend_app.query_one(
            "#pipeline_voice_clone_manifest_suggestions", OptionList
        )
        prompts = [str(option.prompt) for option in suggestions.options]
        assert manifest_path.as_posix() in prompts


@pytest.mark.anyio
async def test_textual_frontend_primary_audio_input_is_visible_near_top(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("selection:\n  llm: vllm_local\n", encoding="utf-8")
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        audio_input = frontend_app.query_one("#pipeline_primary_audio_path", Input)
        await pilot.pause()

        assert audio_input.display is True
        assert audio_input.visible is True
        assert audio_input.region.y <= 8


@pytest.mark.anyio
async def test_textual_frontend_selecting_audio_suggestion_updates_input(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("selection:\n  llm: vllm_local\n", encoding="utf-8")
    target_path = (tmp_path / "picked.wav").resolve()
    target_path.write_bytes(b"wav")
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        frontend_app.query_one("#pipeline_primary_audio_path", Input).value = str(
            tmp_path / "pic"
        )
        frontend_app._refresh_path_suggestions_for("pipeline_primary_audio_path")
        await pilot.pause()
        frontend_app.on_option_list_option_selected(
            cast(
                Any,
                type(
                    "SuggestionEvent",
                    (),
                    {
                        "option_list": type(
                            "OptionListObj",
                            (),
                            {"id": "pipeline_primary_audio_suggestions"},
                        )(),
                        "option_index": 0,
                    },
                )(),
            )
        )

        audio_input = frontend_app.query_one("#pipeline_primary_audio_path", Input)
        suggestions = frontend_app.query_one(
            "#pipeline_primary_audio_suggestions", OptionList
        )
        assert audio_input.value == target_path.as_posix()
        assert suggestions.display is False
        assert len(suggestions.options) == 0


@pytest.mark.anyio
async def test_textual_frontend_matrix_transcript_suggestions_include_json_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("selection:\n  llm: vllm_local\n", encoding="utf-8")
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        frontend_app.query_one("#matrix_input_mode", Select).value = "transcript"
        (tmp_path / "transcript.json").write_text("{}\n", encoding="utf-8")
        frontend_app.query_one("#matrix_input_path", Input).value = str(
            tmp_path / "tra"
        )
        frontend_app._refresh_path_suggestions_for("matrix_input_path")
        await pilot.pause()

        suggestions = frontend_app.query_one("#matrix_input_suggestions", OptionList)
        prompts = [str(option.prompt) for option in suggestions.options]
        assert str(tmp_path / "transcript.json").replace("\\", "/") in prompts


@pytest.mark.anyio
async def test_textual_frontend_config_editor_uses_yaml_language(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("selection:\n  llm: vllm_local\n", encoding="utf-8")
    defaults = load_frontend_defaults(config_path=config_path, workspace_root=tmp_path)
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable="uv")

    async with app.run_test() as pilot:
        frontend_app = cast(SetupAndRunFrontendApp, pilot.app)
        editor = frontend_app.query_one("#config_editor", TextArea)
        assert editor.language == "yaml"
