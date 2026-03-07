"""Loop-memory utilities for summarizer/critic orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import ClassVar, Literal

IssueBucket = Literal[
    "duration",
    "truncation",
    "coherence",
    "naturalness",
    "coverage",
    "quality",
]


def build_loop_context_prompt_block(
    raw_loop_context: str | None, *, char_cap: int = 1024
) -> str:
    """Return bounded loop-context text for safe prompt injection."""
    if raw_loop_context is None:
        return ""
    text = str(raw_loop_context).strip()
    if not text:
        return ""
    if len(text) <= char_cap:
        return text
    available = char_cap
    suffix = ""
    while True:
        truncated = text[:available].rstrip()
        omitted = max(len(text) - len(truncated), 0)
        candidate_suffix = f"\n[truncated {omitted} trailing characters]"
        candidate_available = max(char_cap - len(candidate_suffix), 0)
        if candidate_available == available:
            suffix = candidate_suffix
            break
        available = candidate_available
    if available <= 0:
        return suffix[:char_cap]
    return f"{truncated}{suffix}"


@dataclass(slots=True, frozen=True)
class LoopIssue:
    """Represent one normalized unresolved issue extracted from critic feedback."""

    signature: str
    bucket: IssueBucket
    description: str


@dataclass(slots=True)
class UnresolvedIssueState:
    """Track recurrence metadata for one unresolved signature."""

    issue: LoopIssue
    first_seen_iteration: int
    last_seen_iteration: int
    recurrence_count: int = 1


@dataclass(slots=True, frozen=True)
class LoopMemoryUpdate:
    """Expose computed state after ingesting one critic outcome."""

    unresolved_issue_count: int
    persisted_issue_signatures: list[str]
    stagnation_detected: bool
    strategy_shift_hint: str
    early_stop_recommended: bool
    stagnation_streak: int


@dataclass(slots=True)
class SummarizerLoopMemory:
    """
    Maintain unresolved issue memory for iterative summarizer revisions.

    The memory tracks normalized issue signatures, recurrence, stagnation, and emits
    compact loop context text suitable for prompt injection.
    """

    target_seconds: int
    duration_tolerance_ratio: float = 0.05
    prompt_char_limit: int = 1024
    max_unresolved_issues: int = 5
    stagnation_seconds_delta_threshold: float = 4.0
    early_stop_stagnation_threshold: int = 3
    _unresolved: dict[str, UnresolvedIssueState] = field(default_factory=dict, init=False)
    _previous_unresolved_signatures: frozenset[str] = field(
        default_factory=frozenset, init=False
    )
    _last_estimated_seconds: float | None = field(default=None, init=False)
    _stagnation_streak: int = field(default=0, init=False)
    _last_strategy_shift_hint: str = field(default="", init=False)

    _BUCKET_KEYWORDS: ClassVar[tuple[tuple[IssueBucket, tuple[str, ...]], ...]] = (
        ("truncation", ("truncat", "abrupt", "cut off", "incomplete", "unfinished")),
        ("coherence", ("coher", "flow", "logic", "transition", "structure")),
        ("naturalness", ("natural", "awkward", "stilt", "tone", "robotic")),
        ("coverage", ("cover", "missing", "omit", "key point", "comprehensive")),
        (
            "quality",
            (
                "quality",
                "accuracy",
                "factual",
                "hallucinat",
                "error",
                "incorrect",
                "clarity",
            ),
        ),
    )

    def extract_issue_signatures(
        self, feedback: str, estimated_seconds: float | None = None
    ) -> list[LoopIssue]:
        """Extract deterministic, deduplicated issue signatures from critic feedback."""
        issues: list[LoopIssue] = []
        normalized = self._normalize_text(feedback)
        if normalized:
            fragments = self._split_feedback_fragments(normalized)
            for fragment in fragments:
                bucket = self._bucket_for_text(fragment)
                signature = self._build_signature(bucket=bucket, text=fragment)
                issues.append(
                    LoopIssue(
                        signature=signature,
                        bucket=bucket,
                        description=self._compact_description(fragment),
                    )
                )

        if estimated_seconds is not None:
            duration_issue = self._duration_issue(estimated_seconds)
            if duration_issue is not None:
                issues.append(duration_issue)

        deduped: dict[str, LoopIssue] = {}
        for issue in issues:
            deduped[issue.signature] = issue
        return sorted(deduped.values(), key=lambda issue: issue.signature)

    def update_from_critic(
        self,
        *,
        iteration: int,
        critic_status: str,
        feedback: str,
        estimated_seconds: float | None,
    ) -> LoopMemoryUpdate:
        """Update unresolved memory state from one critic result."""
        del critic_status  # status is currently informational; unresolved issues drive memory.
        current_issues = self.extract_issue_signatures(
            feedback=feedback,
            estimated_seconds=estimated_seconds,
        )
        current_map = {issue.signature: issue for issue in current_issues}
        current_signatures = frozenset(current_map.keys())

        persisted_signatures = sorted(
            signature
            for signature in current_signatures
            if signature in self._previous_unresolved_signatures
        )

        previous_estimated_seconds = self._last_estimated_seconds
        duration_delta = (
            abs(estimated_seconds - previous_estimated_seconds)
            if estimated_seconds is not None and previous_estimated_seconds is not None
            else self.stagnation_seconds_delta_threshold + 1.0
        )
        stagnation_detected = bool(
            current_signatures
            and current_signatures == self._previous_unresolved_signatures
            and duration_delta <= self.stagnation_seconds_delta_threshold
        )

        if stagnation_detected:
            self._stagnation_streak += 1
        else:
            self._stagnation_streak = 0

        next_unresolved: dict[str, UnresolvedIssueState] = {}
        for signature in sorted(current_signatures):
            issue = current_map[signature]
            prior = self._unresolved.get(signature)
            if prior is None:
                next_unresolved[signature] = UnresolvedIssueState(
                    issue=issue,
                    first_seen_iteration=iteration,
                    last_seen_iteration=iteration,
                    recurrence_count=1,
                )
                continue
            next_unresolved[signature] = UnresolvedIssueState(
                issue=issue,
                first_seen_iteration=prior.first_seen_iteration,
                last_seen_iteration=iteration,
                recurrence_count=prior.recurrence_count + 1,
            )

        self._unresolved = next_unresolved
        self._previous_unresolved_signatures = current_signatures
        self._last_estimated_seconds = estimated_seconds

        strategy_shift_hint = ""
        if stagnation_detected:
            strategy_shift_hint = (
                "Strategy shift required: stop repeating prior edits; make one substantial "
                "change targeting the top unresolved issue bucket."
            )
        self._last_strategy_shift_hint = strategy_shift_hint
        early_stop_recommended = self._stagnation_streak >= self.early_stop_stagnation_threshold

        return LoopMemoryUpdate(
            unresolved_issue_count=len(current_signatures),
            persisted_issue_signatures=persisted_signatures,
            stagnation_detected=stagnation_detected,
            strategy_shift_hint=strategy_shift_hint,
            early_stop_recommended=early_stop_recommended,
            stagnation_streak=self._stagnation_streak,
        )

    def to_compact_prompt_block(self) -> str:
        """Render a compact loop-context prompt block with deterministic size bounds."""
        if not self._unresolved:
            return ""

        states = sorted(
            self._unresolved.values(),
            key=lambda state: (-state.recurrence_count, state.issue.signature),
        )[: self.max_unresolved_issues]

        lines = [
            (
                "Unresolved issues from prior critic passes: "
                f"{len(states)} active"
            ),
            "Prioritize these before stylistic edits and avoid repeated failures.",
        ]
        for state in states:
            lines.append(
                (
                    f"- [{state.issue.bucket}] {state.issue.description} "
                    f"(sig={state.issue.signature}, recurrence={state.recurrence_count})"
                )
            )
        if self._last_strategy_shift_hint:
            lines.append(f"Strategy shift: {self._last_strategy_shift_hint}")
        if self._stagnation_streak > 0:
            lines.append(f"Stagnation streak: {self._stagnation_streak}")
        return build_loop_context_prompt_block(
            "\n".join(lines), char_cap=self.prompt_char_limit
        )

    @staticmethod
    def _normalize_text(feedback: str) -> str:
        """Normalize free-form feedback for deterministic parsing."""
        return " ".join(feedback.strip().lower().split())

    @staticmethod
    def _split_feedback_fragments(normalized_feedback: str) -> list[str]:
        """Split feedback into short deterministic fragments."""
        fragments = [
            part.strip(" -.;,")
            for part in re.split(r"[.;\n]|(?:\s+-\s+)|(?:\s+and\s+)", normalized_feedback)
            if part.strip(" -.;,")
        ]
        return fragments or [normalized_feedback]

    def _bucket_for_text(self, text: str) -> IssueBucket:
        """Map feedback text to a deterministic issue bucket."""
        if (
            "duration" in text
            or "seconds" in text
            or "too short" in text
            or "too long" in text
        ):
            return "duration"
        for bucket, keywords in self._BUCKET_KEYWORDS:
            if any(keyword in text for keyword in keywords):
                return bucket
        return "quality"

    @staticmethod
    def _compact_description(fragment: str) -> str:
        """Build a concise issue description from a feedback fragment."""
        tokens = fragment.split()
        return " ".join(tokens[:14])

    @staticmethod
    def _build_signature(*, bucket: IssueBucket, text: str) -> str:
        """Build a deterministic issue signature from bucket and normalized text."""
        token_source = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [token for token in token_source.split() if token]
        signature_tokens = tokens[:8] if tokens else ["unspecified"]
        return f"{bucket}:{'_'.join(signature_tokens)}"

    def _duration_issue(self, estimated_seconds: float) -> LoopIssue | None:
        """Return deterministic duration issue when current draft is out of bounds."""
        tolerance_seconds = self.target_seconds * self.duration_tolerance_ratio
        min_seconds = self.target_seconds - tolerance_seconds
        max_seconds = self.target_seconds + tolerance_seconds
        rounded_seconds = round(estimated_seconds, 2)
        if estimated_seconds < min_seconds:
            return LoopIssue(
                signature=f"duration:below_min_{round(min_seconds, 2)}",
                bucket="duration",
                description=(
                    f"estimated duration {rounded_seconds}s below minimum {round(min_seconds, 2)}s; expand coverage"
                ),
            )
        if estimated_seconds > max_seconds:
            return LoopIssue(
                signature=f"duration:above_max_{round(max_seconds, 2)}",
                bucket="duration",
                description=(
                    f"estimated duration {rounded_seconds}s above maximum {round(max_seconds, 2)}s; compress content"
                ),
            )
        return None
