"""Loop-memory utilities for summarizer/critic orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import ClassVar, Literal

IssueBucket = Literal[
    "word_count",
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
    omitted = len(text) - char_cap
    truncated = text[:char_cap].rstrip()
    return f"{truncated}\n[truncated {omitted} trailing characters]"


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

    min_words: int
    max_words: int
    prompt_char_limit: int = 1024
    max_unresolved_issues: int = 5
    stagnation_word_delta_threshold: int = 4
    early_stop_stagnation_threshold: int = 3
    _unresolved: dict[str, UnresolvedIssueState] = field(default_factory=dict, init=False)
    _previous_unresolved_signatures: frozenset[str] = field(
        default_factory=frozenset, init=False
    )
    _last_word_count: int | None = field(default=None, init=False)
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
        self, feedback: str, word_count: int | None = None
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

        if word_count is not None:
            wc_issue = self._word_count_issue(word_count)
            if wc_issue is not None:
                issues.append(wc_issue)

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
        word_count: int | None,
    ) -> LoopMemoryUpdate:
        """Update unresolved memory state from one critic result."""
        del critic_status  # status is currently informational; unresolved issues drive memory.
        current_issues = self.extract_issue_signatures(feedback=feedback, word_count=word_count)
        current_map = {issue.signature: issue for issue in current_issues}
        current_signatures = frozenset(current_map.keys())

        persisted_signatures = sorted(
            signature
            for signature in current_signatures
            if signature in self._previous_unresolved_signatures
        )

        previous_word_count = self._last_word_count
        word_delta = (
            abs(word_count - previous_word_count)
            if word_count is not None and previous_word_count is not None
            else self.stagnation_word_delta_threshold + 1
        )
        stagnation_detected = bool(
            current_signatures
            and current_signatures == self._previous_unresolved_signatures
            and word_delta <= self.stagnation_word_delta_threshold
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
        self._last_word_count = word_count

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
        if "word count" in text or "too short" in text or "too long" in text:
            return "word_count"
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

    def _word_count_issue(self, word_count: int) -> LoopIssue | None:
        """Return deterministic word-count issue when current draft is out of bounds."""
        if word_count < self.min_words:
            return LoopIssue(
                signature=f"word_count:below_min_{self.min_words}",
                bucket="word_count",
                description=(
                    f"word count {word_count} below minimum {self.min_words}; expand coverage"
                ),
            )
        if word_count > self.max_words:
            return LoopIssue(
                signature=f"word_count:above_max_{self.max_words}",
                bucket="word_count",
                description=(
                    f"word count {word_count} above maximum {self.max_words}; compress content"
                ),
            )
        return None
