from __future__ import annotations

from pathlib import Path

from agents.loop_context import SummarizerLoopMemory


def test_issue_extraction_dedupes_duplicate_feedback_fragments() -> None:
    memory = SummarizerLoopMemory(target_seconds=74)

    issues = memory.extract_issue_signatures(
        feedback=(
            "[] Chronology jump in middle section.\n"
            "[] Chronology jump in middle section.\n"
            "[] Incorrect attribution."
        ),
        estimated_seconds=72,
    )

    signatures = [issue.signature for issue in issues]
    assert len(signatures) == len(set(signatures))
    assert any("chronology_jump_in_middle_section" in signature for signature in signatures)
    assert any("incorrect_attribution" in signature for signature in signatures)


def test_compact_prompt_serializer_respects_length_cap() -> None:
    memory = SummarizerLoopMemory(target_seconds=74, prompt_char_limit=220)
    memory.update_from_critic(
        iteration=1,
        critic_status="fail",
        feedback=(
            "[] Missing key points and omits details.\n"
            "[] Awkward and robotic flow.\n"
            "[] Incorrect attribution in closing section."
        ),
        estimated_seconds=72,
    )

    compact = memory.to_compact_prompt_block()

    assert compact.startswith("Unresolved issues from prior critic passes:")
    assert "Strategy shift" not in compact
    assert len(compact) <= 220


def test_stale_issue_resolution_and_reopen_across_iterations() -> None:
    memory = SummarizerLoopMemory(target_seconds=74)

    first = memory.update_from_critic(
        iteration=1,
        critic_status="fail",
        feedback="[] Missing middle section.\n[] Incorrect attribution.",
        estimated_seconds=72,
    )
    second = memory.update_from_critic(
        iteration=2,
        critic_status="fail",
        feedback="[] Incorrect attribution.",
        estimated_seconds=72,
    )
    third = memory.update_from_critic(
        iteration=3,
        critic_status="fail",
        feedback="[] Missing middle section.\n[] Incorrect attribution.",
        estimated_seconds=72,
    )

    first_signatures = {
        issue.signature
        for issue in memory.extract_issue_signatures(
            feedback="[] Missing middle section.\n[] Incorrect attribution.",
            estimated_seconds=72,
        )
    }
    second_signatures = {
        issue.signature
        for issue in memory.extract_issue_signatures(
            feedback="[] Incorrect attribution.",
            estimated_seconds=72,
        )
    }
    reopened_signature = next(iter(first_signatures - second_signatures))
    persisted_signature = next(iter(second_signatures))

    assert first.unresolved_issue_count == 2
    assert second.unresolved_issue_count == 1
    assert second.persisted_issue_signatures == [persisted_signature]
    assert third.unresolved_issue_count == 2
    assert third.persisted_issue_signatures == [persisted_signature]

    compact = memory.to_compact_prompt_block().lower()
    assert persisted_signature in compact
    assert reopened_signature in compact


def test_stagnation_strategy_shift_and_early_stop_thresholds() -> None:
    memory = SummarizerLoopMemory(
        target_seconds=74,
        early_stop_stagnation_threshold=3,
    )

    first = memory.update_from_critic(
        iteration=1,
        critic_status="fail",
        feedback="[] Tighten chronology.",
        estimated_seconds=60,
    )
    second = memory.update_from_critic(
        iteration=2,
        critic_status="fail",
        feedback="[] Tighten chronology.",
        estimated_seconds=60,
    )
    third = memory.update_from_critic(
        iteration=3,
        critic_status="fail",
        feedback="[] Tighten chronology.",
        estimated_seconds=60,
    )
    fourth = memory.update_from_critic(
        iteration=4,
        critic_status="fail",
        feedback="[] Tighten chronology.",
        estimated_seconds=60,
    )

    assert first.stagnation_detected is False
    assert first.early_stop_recommended is False
    assert second.stagnation_detected is True
    assert second.strategy_shift_hint
    assert second.early_stop_recommended is False
    assert third.stagnation_detected is True
    assert third.early_stop_recommended is False
    assert fourth.stagnation_detected is True
    assert fourth.early_stop_recommended is True


def test_repeated_remedy_detection_tracks_previously_attempted_fix_patterns() -> None:
    """Detect when critic guidance cycles back to a prior failed remedy pattern."""
    memory = SummarizerLoopMemory(target_seconds=74)

    first = memory.update_from_critic(
        iteration=1,
        critic_status="fail",
        feedback="[] Expand missing middle section coverage.",
        estimated_seconds=60,
    )
    second = memory.update_from_critic(
        iteration=2,
        critic_status="fail",
        feedback="[] Tighten chronology transitions.",
        estimated_seconds=61,
    )
    third = memory.update_from_critic(
        iteration=3,
        critic_status="fail",
        feedback="[] Include missing middle section coverage.",
        estimated_seconds=62,
    )

    assert first.repeated_remedy_detected is False
    assert second.repeated_remedy_detected is False
    assert third.repeated_remedy_detected is True
    assert third.repeated_remedy_signatures

    compact = memory.to_compact_prompt_block().lower()
    assert "repeated remedy alert" in compact
    assert "already failed to converge" in compact


def test_loop_memory_artifact_roundtrip_preserves_repeated_remedy_history(
    tmp_path: Path,
) -> None:
    """Persist repeated-remedy memory so later runs can reload the warning state."""
    artifact_path = tmp_path / "loop_memory.json"
    context = {
        "transcript_sha256": "abc123",
        "target_seconds": "74",
        "duration_tolerance_ratio": "0.050000",
    }
    memory = SummarizerLoopMemory(target_seconds=74)
    memory.update_from_critic(
        iteration=1,
        critic_status="fail",
        feedback="[] Expand missing middle section coverage.",
        estimated_seconds=60,
    )
    memory.update_from_critic(
        iteration=2,
        critic_status="fail",
        feedback="[] Include missing middle section coverage.",
        estimated_seconds=61,
    )

    memory.save_artifact(artifact_path, context=context)

    restored = SummarizerLoopMemory(target_seconds=74)
    assert restored.load_artifact(artifact_path, context=context) is True
    compact = restored.to_compact_prompt_block().lower()
    assert "repeated remedy alert" in compact
    assert "attempts=2" in compact

    mismatched = SummarizerLoopMemory(target_seconds=74)
    assert (
        mismatched.load_artifact(
            artifact_path,
            context={
                **context,
                "transcript_sha256": "different",
            },
        )
        is False
    )
