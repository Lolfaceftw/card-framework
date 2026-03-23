"""Coverage for the SPoRC human-eval stage-2 sweep driver."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import sys

from card_framework.shared.paths import REPO_ROOT


def _load_module():
    script_path = (REPO_ROOT / "scripts" / "run_sporc_human_eval_stage2_sweep.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "run_sporc_human_eval_stage2_sweep",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_to_container_path_uses_posix_workspace_root() -> None:
    """Container-bound paths should stay Linux-style even on Windows hosts."""
    module = _load_module()
    host_path = (REPO_ROOT / "artifacts" / "tmp" / "transcript.json").resolve()

    container_path = module._to_container_path(host_path)

    assert container_path == "/workspace/card-framework/artifacts/tmp/transcript.json"
    assert "\\" not in container_path
    assert not container_path.startswith("C:")


def test_machine_episode_label_avoids_spaces_for_hydra_paths() -> None:
    """Internal run directories should avoid Hydra-hostile spaces and punctuation."""
    module = _load_module()

    label = module._machine_episode_label(
        1,
        "The Lion, the Lamb & the Meadow (Chap. 12-13)",
    )

    assert label == "01_the_lion_the_lamb_the_meadow_chap_12_13"
    assert " " not in label
    assert "(" not in label
    assert ")" not in label


def test_sanitize_console_text_replaces_unencodable_characters() -> None:
    """Console tee output should survive cp1252-only terminals."""
    module = _load_module()
    original_stdout = module.sys.stdout
    module.sys.stdout = SimpleNamespace(encoding="cp1252")
    try:
        sanitized = module._sanitize_console_text("Quote \u2014 arrow \U0001F449")
    finally:
        module.sys.stdout = original_stdout

    assert sanitized == 'Quote - arrow ?'


def test_build_run_command_appends_quota_control_overrides() -> None:
    """Optional loop-budget overrides should be forwarded to setup_and_run."""
    module = _load_module()

    command = module._build_run_command(
        output_root=(REPO_ROOT / "artifacts" / "tmp" / "quota-test").resolve(),
        transcript_path=(REPO_ROOT / "artifacts" / "tmp" / "transcript.json").resolve(),
        target_seconds=300,
        llm_profile="deepseek_chat",
        llm_model="deepseek-chat",
        orchestrator_max_iterations=2,
        summarizer_max_tool_turns=8,
        critic_max_tool_turns=3,
        disable_interjector_tool_call_planning=True,
    )

    joined = " ".join(command)
    assert "selection.llm=deepseek_chat" in joined
    assert "selection.stage_llm.summarizer=deepseek_chat" in joined
    assert "profiles.llm.deepseek_chat.model=deepseek-chat" in joined
    assert "orchestrator.max_iterations=2" in joined
    assert "agents.summarizer.max_tool_turns=8" in joined
    assert "agents.critic.max_tool_turns=3" in joined
    assert "audio.interjector.tool_call_planning_enabled=false" in joined
