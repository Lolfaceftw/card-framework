"""Real smoke tests for packaged resources and data contracts."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from card_framework.orchestration.transcript import Transcript
from card_framework.runtime.pipeline_plan import build_pipeline_stage_plan
from card_framework.shared.paths import (
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_JUDGE_RUBRIC_PATH,
    DEFAULT_PROVIDER_PROFILES_PATH,
    DEFAULT_QA_CONFIG_PATH,
    INDEX_TTS_CHECKPOINTS_DIR,
    PROMPTS_DIR,
    VENDOR_INDEX_TTS_DIR,
)
from card_framework.shared.prompt_manager import PromptManager
from card_framework.shared.summary_output import write_summary_xml_to_workspace
from card_framework.shared.summary_xml import parse_summary_xml, serialize_summary_turns

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "real"


@pytest.mark.integration
def test_packaged_resource_paths_and_prompt_loading() -> None:
    """Load real packaged resources from the installed src layout."""
    assert DEFAULT_CONFIG_PATH.exists()
    assert DEFAULT_BENCHMARK_MANIFEST_PATH.exists()
    assert DEFAULT_PROVIDER_PROFILES_PATH.exists()
    assert DEFAULT_QA_CONFIG_PATH.exists()
    assert DEFAULT_JUDGE_RUBRIC_PATH.exists()
    assert PROMPTS_DIR.exists()
    assert VENDOR_INDEX_TTS_DIR.exists()
    assert INDEX_TTS_CHECKPOINTS_DIR.exists()

    config = OmegaConf.load(DEFAULT_CONFIG_PATH)
    assert str(config.audio.voice_clone.runner_project_dir) == "src/card_framework/_vendor/index_tts"
    assert str(config.audio.voice_clone.cfg_path) == "checkpoints/index_tts/config.yaml"
    assert str(config.audio.voice_clone.model_dir) == "checkpoints/index_tts"

    rendered_prompt = PromptManager.get_prompt("corrector_system", max_examples=2)
    assert "Provide correction instructions" in rendered_prompt
    assert "Include 2 or fewer few-shot examples." in rendered_prompt


@pytest.mark.integration
def test_transcript_summary_and_stage_plan_roundtrip(tmp_path: Path) -> None:
    """Exercise real transcript and summary fixtures through packaged helpers."""
    transcript_payload = json.loads(
        (FIXTURES_DIR / "transcript.json").read_text(encoding="utf-8")
    )
    transcript = Transcript.from_mapping(transcript_payload)

    assert len(transcript.segments) == 2
    assert "[SPEAKER_00]:" in transcript.to_full_text()
    assert transcript.metadata["speaker_samples_manifest_path"].endswith("manifest.json")

    summary_xml = (FIXTURES_DIR / "summary.xml").read_text(encoding="utf-8")
    turns = parse_summary_xml(summary_xml)
    assert [turn.speaker for turn in turns] == ["SPEAKER_00", "SPEAKER_01"]

    serialized_summary = serialize_summary_turns(turns)
    written_summary_path = write_summary_xml_to_workspace(serialized_summary, tmp_path)
    assert written_summary_path == tmp_path / "summary.xml"
    assert written_summary_path.exists()

    manifest_path = tmp_path / "voice_clone_manifest.json"
    manifest_path.write_text('{"segments": []}\n', encoding="utf-8")
    stage_plan = build_pipeline_stage_plan(
        {
            "start_stage": "stage-4",
            "final_summary_path": str(written_summary_path),
            "voice_clone_manifest_path": str(manifest_path),
        },
        project_root=tmp_path,
    )
    assert stage_plan.start_stage == "stage-4"
    assert stage_plan.final_summary_path == written_summary_path.resolve()
    assert stage_plan.voice_clone_manifest_path == manifest_path.resolve()
