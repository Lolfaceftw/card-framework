"""Coverage for participant-audio Gemini judge alignment helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from card_framework.shared.paths import REPO_ROOT


def _load_module():
    script_path = (REPO_ROOT / "scripts" / "run_participant_audio_gemini_judge.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "run_participant_audio_gemini_judge",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _complete_item_scores(module) -> dict[str, float]:  # type: ignore[no-untyped-def]
    return {item_name: 0.75 for item_name in module.ITEM_NAMES}


def test_cache_entry_validation_requires_current_schema_and_trace(tmp_path: Path) -> None:
    module = _load_module()
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{}", encoding="utf-8")
    entry = {
        "schema_version": module.GEMINI_SCORE_CACHE_SCHEMA_VERSION,
        "judge_prompt_version": module.JUDGE_PROMPT_CACHE_VERSION,
        "judge_provider": "gemini",
        "model": "gemini-3.1-pro-preview",
        "audio_mode": "source-and-summary",
        "item_scores": _complete_item_scores(module),
        "dimension_scores": {
            dimension_name: 0.75 for dimension_name in module.DIMENSION_NAMES
        },
        "trace_path": str(trace_path),
    }

    assert module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="gemini",
        model="gemini-3.1-pro-preview",
        audio_mode="source-and-summary",
        thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        prompt_profile="default",
    )

    stale_entry = {**entry, "judge_prompt_version": "old"}
    assert not module._cache_entry_is_valid(  # noqa: SLF001
        stale_entry,
        judge_provider="gemini",
        model="gemini-3.1-pro-preview",
        audio_mode="source-and-summary",
        thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        prompt_profile="default",
    )


def test_deepseek_cache_identity_includes_reasoning_effort(tmp_path: Path) -> None:
    module = _load_module()
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{}", encoding="utf-8")
    entry = {
        "schema_version": module.GEMINI_SCORE_CACHE_SCHEMA_VERSION,
        "judge_prompt_version": module.JUDGE_PROMPT_CACHE_VERSION,
        "judge_provider": "deepseek",
        "model": "deepseek-v4-pro",
        "audio_mode": "none",
        "thinking_mode": "enabled",
        "deepseek_reasoning_effort": "high",
        "item_scores": _complete_item_scores(module),
        "dimension_scores": {
            dimension_name: 0.75 for dimension_name in module.DIMENSION_NAMES
        },
        "trace_path": str(trace_path),
    }

    assert module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        audio_mode="none",
        thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        prompt_profile="default",
    )
    assert not module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        audio_mode="none",
        thinking_mode="enabled",
        deepseek_reasoning_effort="max",
        prompt_profile="default",
    )


def test_cache_identity_includes_generation_controls(tmp_path: Path) -> None:
    module = _load_module()
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{}", encoding="utf-8")
    entry = {
        "schema_version": module.GEMINI_SCORE_CACHE_SCHEMA_VERSION,
        "judge_prompt_version": module.JUDGE_PROMPT_CACHE_VERSION,
        "judge_provider": "gemini",
        "model": "gemini-3.1-flash",
        "audio_mode": "human-folder",
        "requested_max_tokens": 8192,
        "judge_seed": 123,
        "judge_temperature": 0.0,
        "item_scores": _complete_item_scores(module),
        "dimension_scores": {
            dimension_name: 0.75 for dimension_name in module.DIMENSION_NAMES
        },
        "trace_path": str(trace_path),
    }

    assert module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="gemini",
        model="gemini-3.1-flash",
        audio_mode="human-folder",
        thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        prompt_profile="default",
        requested_max_tokens=8192,
        judge_seed=123,
        judge_temperature=0.0,
    )
    assert not module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="gemini",
        model="gemini-3.1-flash",
        audio_mode="human-folder",
        thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        prompt_profile="default",
        requested_max_tokens=8192,
        judge_seed=456,
        judge_temperature=0.0,
    )
    assert not module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="gemini",
        model="gemini-3.1-flash",
        audio_mode="human-folder",
        thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        prompt_profile="default",
        requested_max_tokens=8192,
        judge_seed=123,
        judge_temperature=0.2,
    )


def test_profile_specific_legacy_cache_may_omit_prompt_profile(tmp_path: Path) -> None:
    module = _load_module()
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{}", encoding="utf-8")
    entry = {
        "schema_version": module.GEMINI_SCORE_CACHE_SCHEMA_VERSION,
        "judge_prompt_version": module.JUDGE_PROMPT_CACHE_VERSION,
        "judge_provider": "deepseek",
        "model": "deepseek-v4-pro",
        "audio_mode": "none",
        "thinking_mode": "disabled",
        "item_scores": _complete_item_scores(module),
        "dimension_scores": {
            dimension_name: 0.75 for dimension_name in module.DIMENSION_NAMES
        },
        "trace_path": str(trace_path),
    }

    assert module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        audio_mode="none",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        prompt_profile="human_calibrated_style_sensitive",
    )


def test_vllm_cache_identity_includes_base_url_and_thinking(tmp_path: Path) -> None:
    module = _load_module()
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{}", encoding="utf-8")
    entry = {
        "schema_version": module.GEMINI_SCORE_CACHE_SCHEMA_VERSION,
        "judge_prompt_version": module.JUDGE_PROMPT_CACHE_VERSION,
        "judge_provider": "vllm",
        "model": "Qwen/Qwen3.5-27B",
        "vllm_base_url": "http://127.0.0.1:8000/v1",
        "audio_mode": "none",
        "thinking_mode": "enabled",
        "item_scores": _complete_item_scores(module),
        "dimension_scores": {
            dimension_name: 0.75 for dimension_name in module.DIMENSION_NAMES
        },
        "trace_path": str(trace_path),
    }

    assert module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="vllm",
        model="Qwen/Qwen3.5-27B",
        vllm_base_url="http://127.0.0.1:8000/v1/",
        audio_mode="none",
        thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        prompt_profile="default",
    )
    assert not module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="vllm",
        model="Qwen/Qwen3.5-27B",
        vllm_base_url="http://127.0.0.1:8001/v1",
        audio_mode="none",
        thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        prompt_profile="default",
    )
    assert not module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="vllm",
        model="Qwen/Qwen3.5-27B",
        vllm_base_url="http://127.0.0.1:8000/v1",
        audio_mode="none",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        prompt_profile="default",
    )


def test_deepseek_run_label_includes_reasoning_effort() -> None:
    module = _load_module()

    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="enabled",
        deepseek_reasoning_effort="max",
        audio_mode="none",
        prompt_profile="default",
    ) == "deepseek_deepseek_v4_pro_thinking_enabled_effort_max"

    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="human_calibrated",
    ) == "deepseek_deepseek_v4_pro_thinking_disabled_prompt_human_calibrated"

    assert module._run_label(  # noqa: SLF001
        judge_provider="vllm",
        model="Qwen/Qwen3.5-27B",
        thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="default",
    ) == "vllm_qwen_qwen3_5_27b_thinking_enabled"

    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="human_calibrated_style_sensitive",
        max_tokens=8192,
        judge_temperature=0.0,
    ) == (
        "deepseek_deepseek_v4_pro_thinking_disabled_prompt_"
        "human_calibrated_style_sensitive_max_tokens_8192_temp_0p0"
    )

    assert module._run_label(  # noqa: SLF001
        judge_provider="gemini",
        model="gemini-3.1-flash",
        thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        audio_mode="human-folder",
        prompt_profile="default",
        gemini_thinking_level="low",
        judge_seed=123,
    ) == "gemini_gemini_3_1_flash_thinking_low_audio_human_folder_seed_123"


def test_vllm_env_defaults_are_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()

    for env_name in (
        "CARD_FRAMEWORK_VLLM_MODEL",
        "VLLM_MODEL",
        "VLLM_URL",
        "CARD_FRAMEWORK_VLLM_URL",
        "VLLM_BASE_URL",
        "CARD_FRAMEWORK_VLLM_BASE_URL",
        "VLLM_API_KEY",
        "CARD_FRAMEWORK_VLLM_API_KEY",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen/Qwen3.6-27B")
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")

    assert (
        module._resolve_model(judge_provider="vllm", raw_model="")  # noqa: SLF001
        == "Qwen/Qwen3.6-27B"
    )
    assert (
        module._resolve_vllm_base_url("")  # noqa: SLF001
        == "http://127.0.0.1:8000/v1"
    )
    assert module._resolve_vllm_api_key("") == "EMPTY"  # noqa: SLF001


def test_human_listener_profile_is_non_leaking_and_labeled() -> None:
    module = _load_module()

    profile_note = module.PROMPT_PROFILE_NOTES["human_listener_tolerant_ranked"]
    predicted_note = module.PROMPT_PROFILE_NOTES["predicted_human_listener_v1"]
    hybrid_note = module.PROMPT_PROFILE_NOTES["listener_factual_style_sensitive_v1"]
    synthetic_anchor_note = module.PROMPT_PROFILE_NOTES["synthetic_anchor_listener_v1"]
    qwen_note = module.PROMPT_PROFILE_NOTES["qwen_item_separated_listener_v1"]
    qwen_guarded_note = module.PROMPT_PROFILE_NOTES["qwen_duration_guarded_style_v1"]
    factual_duration_note = module.PROMPT_PROFILE_NOTES[
        "human_calibrated_style_sensitive_factual_duration_v1"
    ]
    factual_duration_speaker_guard_note = module.PROMPT_PROFILE_NOTES[
        "human_calibrated_style_sensitive_factual_duration_speaker_guard_v1"
    ]
    guarded_style_note = module.PROMPT_PROFILE_NOTES[
        "human_calibrated_style_sensitive_guarded_v1"
    ]
    guarded_speaker_restore_note = module.PROMPT_PROFILE_NOTES[
        "human_calibrated_style_sensitive_guarded_speaker_restore_v1"
    ]

    for note in (
        profile_note,
        predicted_note,
        hybrid_note,
        synthetic_anchor_note,
        qwen_note,
        qwen_guarded_note,
        factual_duration_note,
        factual_duration_speaker_guard_note,
        guarded_style_note,
        guarded_speaker_restore_note,
    ):
        assert "artifact-level human mean" not in note
        assert "copy these values exactly" not in note
    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="human_listener_tolerant_ranked",
    ) == (
        "deepseek_deepseek_v4_pro_thinking_disabled_prompt_"
        "human_listener_tolerant_ranked"
    )
    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="predicted_human_listener_v1",
    ) == (
        "deepseek_deepseek_v4_pro_thinking_disabled_prompt_"
        "predicted_human_listener_v1"
    )
    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="listener_factual_style_sensitive_v1",
    ) == (
        "deepseek_deepseek_v4_pro_thinking_disabled_prompt_"
        "listener_factual_style_sensitive_v1"
    )
    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="synthetic_anchor_listener_v1",
    ) == (
        "deepseek_deepseek_v4_pro_thinking_disabled_prompt_"
        "synthetic_anchor_listener_v1"
    )
    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="human_calibrated_style_sensitive_factual_duration_v1",
    ) == (
        "deepseek_deepseek_v4_pro_thinking_disabled_prompt_"
        "human_calibrated_style_sensitive_factual_duration_v1"
    )
    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="human_calibrated_style_sensitive_factual_duration_speaker_guard_v1",
    ) == (
        "deepseek_deepseek_v4_pro_thinking_disabled_prompt_"
        "human_calibrated_style_sensitive_factual_duration_speaker_guard_v1"
    )
    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="human_calibrated_style_sensitive_guarded_v1",
    ) == (
        "deepseek_deepseek_v4_pro_thinking_disabled_prompt_"
        "human_calibrated_style_sensitive_guarded_v1"
    )
    assert module._run_label(  # noqa: SLF001
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="human_calibrated_style_sensitive_guarded_speaker_restore_v1",
    ) == (
        "deepseek_deepseek_v4_pro_thinking_disabled_prompt_"
        "human_calibrated_style_sensitive_guarded_speaker_restore_v1"
    )
    assert module._run_label(  # noqa: SLF001
        judge_provider="vllm",
        model="Qwen/Qwen3.6-27B",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="qwen_item_separated_listener_v1",
    ) == (
        "vllm_qwen_qwen3_6_27b_thinking_disabled_prompt_"
        "qwen_item_separated_listener_v1"
    )
    assert module._run_label(  # noqa: SLF001
        judge_provider="vllm",
        model="Qwen/Qwen3.6-27B",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        audio_mode="none",
        prompt_profile="qwen_duration_guarded_style_v1",
    ) == (
        "vllm_qwen_qwen3_6_27b_thinking_disabled_prompt_"
        "qwen_duration_guarded_style_v1"
    )


def test_leave_one_episode_out_selects_only_from_training_groups() -> None:
    module = _load_module()

    def make_response(
        artifact_key: str,
        participant_id: str,
        score: float,
    ) -> object:
        _prefix, episode, duration = artifact_key.split("_")
        item_scores = {item_name: score for item_name in module.ITEM_NAMES}
        return module.HumanResponse(
            participant_id=participant_id,
            participant_email=f"{participant_id}@example.com",
            assignment=module.Assignment(
                participant_id=participant_id,
                participant_email=f"{participant_id}@example.com",
                podcast_id=f"Podcast {episode}",
                duration=duration,
                folder_link="",
            ),
            item_scores=item_scores,
            dimension_scores=module._dimension_means_from_items(item_scores),  # noqa: SLF001
            comment="",
        )

    def make_score(value: float) -> dict[str, object]:
        item_scores = {item_name: value for item_name in module.ITEM_NAMES}
        dimension_scores = module._dimension_means_from_items(item_scores)  # noqa: SLF001
        return {
            "item_scores": item_scores,
            "dimension_scores": dimension_scores,
            "composite": value,
        }

    responses = [
        make_response("Podcast_01_30s", "01", 0.0),
        make_response("Podcast_01_1m", "02", 1.0),
        make_response("Podcast_02_30s", "03", 0.0),
        make_response("Podcast_02_1m", "04", 1.0),
    ]
    candidate_scores = {
        "train_ep02_good": {
            "Podcast_01_30s": make_score(1.0),
            "Podcast_01_1m": make_score(0.0),
            "Podcast_02_30s": make_score(0.0),
            "Podcast_02_1m": make_score(1.0),
        },
        "train_ep01_good": {
            "Podcast_01_30s": make_score(0.0),
            "Podcast_01_1m": make_score(1.0),
            "Podcast_02_30s": make_score(1.0),
            "Podcast_02_1m": make_score(0.0),
        },
    }

    report = module._build_leave_one_episode_out_report(  # noqa: SLF001
        responses=responses,
        candidate_scores_by_profile=candidate_scores,
        selection_metric="spearman",
        bootstrap_samples=50,
        bootstrap_seed=7,
    )

    fold_selections = {
        fold["held_out_episode_group"]: fold["selected_prompt_profile"]
        for fold in report["folds"]
    }
    assert fold_selections == {
        "01": "train_ep02_good",
        "02": "train_ep01_good",
    }
    composite = report["pooled_out_of_fold"]["artifact_level"]["dimension_means"][
        "Composite"
    ]
    assert composite["n"] == 4
    assert composite["spearman"] == -1.0
    assert composite["pearson"] == -1.0


def test_dimension_ensemble_takes_items_from_mapped_profiles() -> None:
    module = _load_module()
    dimension_profile_map = module._parse_dimension_ensemble_profile_map(  # noqa: SLF001
        "factual=human_calibrated_style_sensitive_guarded_v1; "
        "natural=human_calibrated_style_sensitive_guarded_v1; "
        "speaker=human_calibrated_style_sensitive"
    )
    assert dimension_profile_map == {
        "Factualness": ("human_calibrated_style_sensitive_guarded_v1",),
        "Naturalness": ("human_calibrated_style_sensitive_guarded_v1",),
        "Speaker Grammar Similarity": ("human_calibrated_style_sensitive",),
    }

    def make_score(value: float) -> dict[str, object]:
        item_scores = {item_name: value for item_name in module.ITEM_NAMES}
        return {
            "item_scores": item_scores,
            "dimension_scores": module._dimension_means_from_items(item_scores),  # noqa: SLF001
            "composite": value,
            "trace_path": f"trace-{value}.json",
        }

    ensemble = module._build_dimension_ensemble_scores(  # noqa: SLF001
        candidate_scores_by_profile={
            "human_calibrated_style_sensitive_guarded_v1": {
                "Podcast_01_30s": make_score(0.25)
            },
            "human_calibrated_style_sensitive": {
                "Podcast_01_30s": make_score(0.75)
            },
        },
        dimension_profile_map=dimension_profile_map,
    )

    item_scores = ensemble["Podcast_01_30s"]["item_scores"]
    assert all(item_scores[item_name] == 0.25 for item_name in ("F1", "F2", "F3"))
    assert all(item_scores[item_name] == 0.25 for item_name in ("N1", "N2", "N3"))
    assert all(item_scores[item_name] == 0.75 for item_name in ("S1", "S2", "S3"))
    assert ensemble["Podcast_01_30s"]["dimension_scores"] == {
        "Factualness": 0.25,
        "Naturalness": 0.25,
        "Speaker Grammar Similarity": 0.75,
    }
    assert ensemble["Podcast_01_30s"]["composite"] == pytest.approx(5 / 12)


def test_dimension_ensemble_averages_multiple_profiles_per_dimension() -> None:
    module = _load_module()
    dimension_profile_map = module._parse_dimension_ensemble_profile_map(  # noqa: SLF001
        "Factualness=human_calibrated_style_sensitive_guarded_v1;"
        "Naturalness=human_calibrated_style_sensitive+predicted_human_listener_v1;"
        "Speaker Grammar Similarity=human_calibrated_style_sensitive"
    )

    def make_score(value: float) -> dict[str, object]:
        item_scores = {item_name: value for item_name in module.ITEM_NAMES}
        return {
            "item_scores": item_scores,
            "dimension_scores": module._dimension_means_from_items(item_scores),  # noqa: SLF001
            "composite": value,
            "trace_path": f"trace-{value}.json",
        }

    ensemble = module._build_dimension_ensemble_scores(  # noqa: SLF001
        candidate_scores_by_profile={
            "human_calibrated_style_sensitive_guarded_v1": {
                "Podcast_01_30s": make_score(0.25)
            },
            "human_calibrated_style_sensitive": {
                "Podcast_01_30s": make_score(0.50)
            },
            "predicted_human_listener_v1": {
                "Podcast_01_30s": make_score(1.00)
            },
        },
        dimension_profile_map=dimension_profile_map,
    )

    assert all(
        ensemble["Podcast_01_30s"]["item_scores"][item_name] == 0.75
        for item_name in ("N1", "N2", "N3")
    )
    assert ensemble["Podcast_01_30s"]["dimension_scores"]["Naturalness"] == 0.75
    assert ensemble["Podcast_01_30s"]["component_trace_paths"]["Naturalness"] == [
        "trace-0.5.json",
        "trace-1.0.json",
    ]


def test_item_ensemble_averages_multiple_profiles_per_item() -> None:
    module = _load_module()
    profile_a = "human_calibrated_style_sensitive"
    profile_b = "judge_reliability_v1"
    item_profile_map = {
        item_name: (profile_a,) for item_name in module.ITEM_NAMES
    }
    item_profile_map["F3"] = (profile_a, profile_b)

    def make_score(value: float) -> dict[str, object]:
        item_scores = {item_name: value for item_name in module.ITEM_NAMES}
        return {
            "item_scores": item_scores,
            "dimension_scores": module._dimension_means_from_items(item_scores),  # noqa: SLF001
            "composite": value,
            "trace_path": f"trace-{value}.json",
        }

    ensemble = module._build_item_ensemble_scores(  # noqa: SLF001
        candidate_scores_by_profile={
            profile_a: {"Podcast_01_30s": make_score(0.25)},
            profile_b: {"Podcast_01_30s": make_score(0.75)},
        },
        item_profile_map=item_profile_map,
    )

    assert ensemble["Podcast_01_30s"]["item_scores"]["F1"] == 0.25
    assert ensemble["Podcast_01_30s"]["item_scores"]["F3"] == 0.50
    assert ensemble["Podcast_01_30s"]["component_trace_paths"]["F3"] == [
        "trace-0.25.json",
        "trace-0.75.json",
    ]
    assert ensemble["Podcast_01_30s"]["dimension_scores"]["Factualness"] == pytest.approx(
        1 / 3
    )


def test_in_sample_gold_cache_requires_calibration_fingerprint(tmp_path: Path) -> None:
    module = _load_module()
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{}", encoding="utf-8")
    entry = {
        "schema_version": module.GEMINI_SCORE_CACHE_SCHEMA_VERSION,
        "judge_prompt_version": module.JUDGE_PROMPT_CACHE_VERSION,
        "judge_provider": "deepseek",
        "model": "deepseek-v4-pro",
        "audio_mode": "none",
        "thinking_mode": "disabled",
        "prompt_profile": "in_sample_gold_anchor",
        "calibration_fingerprint": "abc123",
        "item_scores": _complete_item_scores(module),
        "dimension_scores": {
            dimension_name: 0.75 for dimension_name in module.DIMENSION_NAMES
        },
        "trace_path": str(trace_path),
    }

    assert module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        audio_mode="none",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        prompt_profile="in_sample_gold_anchor",
        calibration_fingerprint="abc123",
    )
    assert not module._cache_entry_is_valid(  # noqa: SLF001
        entry,
        judge_provider="deepseek",
        model="deepseek-v4-pro",
        audio_mode="none",
        thinking_mode="disabled",
        deepseek_reasoning_effort="high",
        prompt_profile="in_sample_gold_anchor",
        calibration_fingerprint="changed",
    )


def test_deepseek_provider_passes_timeout_and_thinking_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    captured_client_kwargs: dict[str, object] = {}
    captured_create_kwargs: dict[str, object] = {}

    class _Message:
        content = '{"Factualness": 1.0}'
        reasoning_content = "reasoning"

    class _Choice:
        message = _Message()

    class _Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            captured_create_kwargs.update(kwargs)
            return SimpleNamespace(choices=[_Choice()])

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured_client_kwargs.update(kwargs)
            self.chat = _Chat()

    monkeypatch.setattr(module, "OpenAI", _FakeOpenAI)

    provider = module._DeepSeekJudgeProvider(  # noqa: SLF001
        api_key="test-key",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        thinking_mode="enabled",
        reasoning_effort="max",
        request_timeout_seconds=123,
        temperature=0.0,
    )
    provider.generate("system", "user", max_tokens=456)

    assert captured_client_kwargs["timeout"] == 123
    assert captured_create_kwargs["reasoning_effort"] == "max"
    assert captured_create_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert captured_create_kwargs["max_tokens"] == 456
    assert captured_create_kwargs["temperature"] == 0.0
    assert "seed" not in captured_create_kwargs


def test_deepseek_provider_defaults_to_provider_max_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    captured_create_kwargs: dict[str, object] = {}

    class _Message:
        content = '{"Factualness": 1.0}'
        reasoning_content = ""

    class _Choice:
        message = _Message()

    class _Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            captured_create_kwargs.update(kwargs)
            return SimpleNamespace(choices=[_Choice()])

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = _Chat()

    monkeypatch.setattr(module, "OpenAI", _FakeOpenAI)

    provider = module._DeepSeekJudgeProvider(  # noqa: SLF001
        api_key="test-key",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        thinking_mode="enabled",
        reasoning_effort="high",
        request_timeout_seconds=123,
    )
    provider.generate("system", "user", max_tokens=None)

    assert captured_create_kwargs["max_tokens"] == module.DEEPSEEK_MAX_OUTPUT_TOKENS
    assert (
        provider.last_response_details["request"]["max_tokens"]
        == module.DEEPSEEK_MAX_OUTPUT_TOKENS
    )


def test_vllm_provider_uses_pinned_model_and_thinking_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    captured_client_kwargs: dict[str, object] = {}
    captured_create_kwargs: dict[str, object] = {}

    class _Message:
        content = '{"item_scores":{}}'
        reasoning_content = "reasoning"

    class _Choice:
        message = _Message()

    class _Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            captured_create_kwargs.update(kwargs)
            return SimpleNamespace(choices=[_Choice()])

    class _Chat:
        completions = _Completions()

    class _Models:
        @staticmethod
        def list() -> object:
            return SimpleNamespace(
                data=[SimpleNamespace(id="Qwen/Qwen3.5-27B")]
            )

    class _FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured_client_kwargs.update(kwargs)
            self.models = _Models()
            self.chat = _Chat()

    monkeypatch.setattr(module, "OpenAI", _FakeOpenAI)

    provider = module._VLLMJudgeProvider(  # noqa: SLF001
        api_key="EMPTY",
        model="Qwen/Qwen3.5-27B",
        base_url="http://127.0.0.1:8000/v1/",
        thinking_mode="enabled",
        request_timeout_seconds=123,
        seed=123,
        temperature=0.2,
    )
    provider.generate("system", "user", max_tokens=456)

    assert captured_client_kwargs["base_url"] == "http://127.0.0.1:8000/v1"
    assert captured_client_kwargs["timeout"] == 123
    assert captured_create_kwargs["model"] == "Qwen/Qwen3.5-27B"
    assert captured_create_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    assert captured_create_kwargs["max_tokens"] == 456
    assert captured_create_kwargs["seed"] == 123
    assert captured_create_kwargs["temperature"] == 0.2


def test_vllm_provider_defaults_to_advertised_context_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    captured_create_kwargs: dict[str, object] = {}

    class _Message:
        content = '{"item_scores":{}}'
        reasoning_content = ""

    class _Choice:
        message = _Message()

    class _Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            captured_create_kwargs.update(kwargs)
            return SimpleNamespace(choices=[_Choice()])

    class _Chat:
        completions = _Completions()

    class _Models:
        @staticmethod
        def list() -> object:
            return SimpleNamespace(
                data=[SimpleNamespace(id="Qwen/Qwen3.6-27B", max_model_len=262144)]
            )

    class _FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.models = _Models()
            self.chat = _Chat()

    monkeypatch.setattr(module, "OpenAI", _FakeOpenAI)

    provider = module._VLLMJudgeProvider(  # noqa: SLF001
        api_key="EMPTY",
        model="Qwen/Qwen3.6-27B",
        base_url="http://127.0.0.1:8000/v1/",
        thinking_mode="enabled",
        request_timeout_seconds=123,
    )
    provider.generate("system", "user", max_tokens=None)

    expected_max_tokens = 262144 - module.VLLM_CONTEXT_TOKEN_HEADROOM
    assert captured_create_kwargs["max_tokens"] == expected_max_tokens
    assert provider.last_response_details["request"]["max_tokens"] == expected_max_tokens
    assert provider.last_response_details["request"]["max_model_len"] == 262144


def test_incomplete_item_scores_are_rejected() -> None:
    module = _load_module()

    with pytest.raises(RuntimeError, match="omitted item_scores"):
        module._require_complete_item_scores(  # noqa: SLF001
            {"F1": 1.0},
            artifact_key="Podcast_01_30s",
        )


def test_artifact_discovery_rejects_duplicate_summary_audio(tmp_path: Path) -> None:
    module = _load_module()
    folder = tmp_path / "Podcast_01_30s"
    folder.mkdir()
    (folder / "[SOURCE] 01 - Episode - 10m audio slice transcript.txt").write_text(
        "<SPEAKER_00>source</SPEAKER_00>",
        encoding="utf-8",
    )
    (folder / "[SOURCE] 01 - Episode - 10m audio slice.mp3").write_bytes(b"source")
    (folder / "[SUMMARY] 01 - Episode - 30s - summary.txt").write_text(
        "<SPEAKER_00>summary</SPEAKER_00>",
        encoding="utf-8",
    )
    (folder / "[SUMMARY] 01 - Episode - 30s.wav").write_bytes(b"summary")
    (folder / "[SUMMARY] 01 - Episode - duplicate.wav").write_bytes(b"duplicate")

    with pytest.raises(RuntimeError, match="Expected one summary audio file"):
        module._discover_participant_artifacts(  # noqa: SLF001
            tmp_path,
            tmp_path / "out",
        )


def test_human_folder_audio_mode_attaches_human_visible_audio(tmp_path: Path) -> None:
    module = _load_module()
    folder = tmp_path / "Podcast_01_30s"
    folder.mkdir()
    (folder / "[SOURCE] 01 - Episode - 10m audio slice transcript.txt").write_text(
        "<SPEAKER_00>source</SPEAKER_00>",
        encoding="utf-8",
    )
    (folder / "[SOURCE] 01 - Episode - 10m audio slice.mp3").write_bytes(b"source")
    (folder / "[SOURCE] 01 - Episode - SPEAKER_00 voice sample.wav").write_bytes(
        b"speaker-00",
    )
    (folder / "[SOURCE] 01 - Episode - SPEAKER_01 voice sample.wav").write_bytes(
        b"speaker-01",
    )
    (folder / "[SUMMARY] 01 - Episode - 30s - summary.txt").write_text(
        "<SPEAKER_00>summary</SPEAKER_00>",
        encoding="utf-8",
    )
    (folder / "[SUMMARY] 01 - Episode - 30s.wav").write_bytes(b"summary")

    artifacts = module._discover_participant_artifacts(  # noqa: SLF001
        tmp_path,
        tmp_path / "out",
    )
    artifact = artifacts["Podcast_01_30s"]
    attachments = module._audio_paths_for_mode(artifact, "human-folder")  # noqa: SLF001

    assert [label for label, _path in attachments] == [
        "source_10m_audio_slice",
        "source_SPEAKER_00_voice_sample",
        "source_SPEAKER_01_voice_sample",
        "candidate_summary_audio",
    ]
    assert [path.name for _label, path in attachments] == [
        "[SOURCE] 01 - Episode - 10m audio slice.mp3",
        "[SOURCE] 01 - Episode - SPEAKER_00 voice sample.wav",
        "[SOURCE] 01 - Episode - SPEAKER_01 voice sample.wav",
        "[SUMMARY] 01 - Episode - 30s.wav",
    ]

    manifest_path = tmp_path / "manifest.json"
    module._write_artifact_manifest(manifest_path, artifacts)  # noqa: SLF001
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [Path(path).name for path in manifest["Podcast_01_30s"]["speaker_sample_audio_paths"]] == [
        "[SOURCE] 01 - Episode - SPEAKER_00 voice sample.wav",
        "[SOURCE] 01 - Episode - SPEAKER_01 voice sample.wav",
    ]


def test_gemini_workflow_feeds_raw_xml_without_per_speaker_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    source_text = (
        "<SPEAKER_00>Raw <INTERJECTION speaker=\"SPEAKER_01\">yeah</INTERJECTION> "
        "transcript.</SPEAKER_00>"
    )
    summary_text = "<SPEAKER_00>Raw summary.</SPEAKER_00>"
    source_path = tmp_path / "source.txt"
    summary_path = tmp_path / "summary.txt"
    source_path.write_text(source_text, encoding="utf-8")
    summary_path.write_text(summary_text, encoding="utf-8")
    artifact = module.ParticipantArtifact(
        artifact_key="Podcast_01_30s",
        podcast_id="Podcast 01",
        duration="30s",
        source_transcript_path=source_path,
        summary_transcript_path=summary_path,
        source_audio_path=None,
        summary_audio_path=None,
        speaker_sample_audio_paths=(),
        local_source_transcript_path=source_path,
        local_summary_transcript_path=summary_path,
    )
    captured_calls: list[dict[str, object]] = []
    captured_provider_kwargs: dict[str, object] = {}

    class _FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            captured_provider_kwargs.update(kwargs)
            self.last_response_details = {
                "response": {
                    "content": "{}",
                    "reasoning_content": "provider thought summary",
                }
            }

    class _FakeJudge:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def evaluate(self, **kwargs: object) -> object:
            captured_calls.append(kwargs)
            return SimpleNamespace(
                status="ok",
                scores={dimension_name: 0.0 for dimension_name in module.DIMENSION_NAMES},
                item_scores=_complete_item_scores(module),
                confidence=0.8,
                rationale="ok",
            )

    monkeypatch.setattr(module, "_GeminiAudioJudgeProvider", _FakeProvider)
    monkeypatch.setattr(module, "ConversationSummaryJudge", _FakeJudge)
    monkeypatch.setattr(module, "load_repo_dotenv", lambda _root: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    scores = module._load_judge_scores(  # noqa: SLF001
        {"Podcast_01_30s": artifact},
        tmp_path / "scores.json",
        model="gemini-3.1-pro-preview",
        judge_provider="gemini",
        audio_mode="none",
        thinking_mode="enabled",
        gemini_thinking_level="low",
        deepseek_base_url="https://api.deepseek.com",
        vllm_base_url="",
        vllm_api_key="EMPTY",
        deepseek_reasoning_effort="high",
        prompt_profile="human_calibrated",
        request_timeout_seconds=1.0,
        max_tokens=128,
        judge_seed=123,
        judge_temperature=0.0,
        max_judge_attempts=1,
        trace_dir=tmp_path / "traces",
        skip_gemini=False,
        force_gemini=False,
    )

    assert scores["Podcast_01_30s"]["item_scores"]["F1"] == 0.75
    assert scores["Podcast_01_30s"]["dimension_scores"]["Factualness"] == 0.75
    assert captured_provider_kwargs["api_key"] == "test-key"
    assert captured_provider_kwargs["thinking_level"] == "low"
    assert captured_provider_kwargs["audio_attachments"] == []
    assert captured_provider_kwargs["seed"] == 123
    assert captured_provider_kwargs["temperature"] == 0.0
    assert captured_calls[0]["source_text"] == source_text
    assert captured_calls[0]["summary_xml"] == summary_text
    assert captured_calls[0]["transcript_per_speaker_messages"] == {}
    assert captured_calls[0]["summary_per_speaker_messages"] == {}
    scoring_profile_note = str(captured_calls[0]["scoring_profile_note"])
    assert "Score the dimensions independently" in scoring_profile_note
    assert "Human-aligned" not in scoring_profile_note
    trace_path = Path(scores["Podcast_01_30s"]["trace_path"])
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert (
        trace_payload["provider_last_response_details"]["response"]["reasoning_content"]
        == "provider thought summary"
    )
    assert trace_payload["requested_max_tokens"] == 128
    assert trace_payload["judge_seed"] == 123
    assert trace_payload["judge_temperature"] == 0.0
