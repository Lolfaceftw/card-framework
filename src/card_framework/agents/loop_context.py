"""Loop-memory utilities for summarizer/critic orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
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
RemedyAction = Literal[
    "expand",
    "compress",
    "repair",
    "verify",
    "resequence",
    "other",
]
_ISSUE_BUCKET_VALUES: frozenset[str] = frozenset(IssueBucket.__args__)
_REMEDY_ACTION_VALUES: frozenset[str] = frozenset(RemedyAction.__args__)


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


@dataclass(slots=True, frozen=True)
class LoopRemedy:
    """Represent one normalized remedy pattern extracted from critic feedback."""

    signature: str
    bucket: IssueBucket
    action: RemedyAction
    description: str


@dataclass(slots=True)
class UnresolvedIssueState:
    """Track recurrence metadata for one unresolved signature."""

    issue: LoopIssue
    first_seen_iteration: int
    last_seen_iteration: int
    recurrence_count: int = 1


@dataclass(slots=True)
class AttemptedRemedyState:
    """Track one remedy pattern that the summarizer has already attempted."""

    remedy: LoopRemedy
    first_seen_iteration: int
    last_seen_iteration: int
    attempt_count: int = 1


@dataclass(slots=True, frozen=True)
class LoopMemoryUpdate:
    """Expose computed state after ingesting one critic outcome."""

    unresolved_issue_count: int
    persisted_issue_signatures: list[str]
    stagnation_detected: bool
    strategy_shift_hint: str
    early_stop_recommended: bool
    stagnation_streak: int
    repeated_remedy_detected: bool
    repeated_remedy_signatures: list[str]


@dataclass(slots=True)
class SummarizerLoopMemory:
    """
    Maintain unresolved issue and remedy-attempt memory across critic passes.

    The memory tracks:
    1. Normalized unresolved issue signatures and recurrence.
    2. Previously attempted remedy patterns so the summarizer can avoid cycling
       back to the same failed fix.
    3. Stagnation signals and compact prompt-ready guidance.
    """

    target_seconds: int
    duration_tolerance_ratio: float = 0.05
    prompt_char_limit: int = 1024
    max_unresolved_issues: int = 5
    max_attempted_remedies: int = 6
    stagnation_seconds_delta_threshold: float = 4.0
    early_stop_stagnation_threshold: int = 3
    _unresolved: dict[str, UnresolvedIssueState] = field(default_factory=dict, init=False)
    _previous_unresolved_signatures: frozenset[str] = field(
        default_factory=frozenset, init=False
    )
    _last_estimated_seconds: float | None = field(default=None, init=False)
    _stagnation_streak: int = field(default=0, init=False)
    _last_strategy_shift_hint: str = field(default="", init=False)
    _attempted_remedies: dict[str, AttemptedRemedyState] = field(
        default_factory=dict,
        init=False,
    )
    _last_repeated_remedy_signatures: list[str] = field(default_factory=list, init=False)

    _BUCKET_KEYWORDS: ClassVar[tuple[tuple[IssueBucket, tuple[str, ...]], ...]] = (
        ("truncation", ("truncat", "abrupt", "cut off", "incomplete", "unfinished")),
        (
            "coherence",
            ("coher", "flow", "logic", "transition", "structure", "chronolog"),
        ),
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
                "attribution",
            ),
        ),
    )
    _ACTION_KEYWORDS: ClassVar[tuple[tuple[RemedyAction, tuple[str, ...]], ...]] = (
        (
            "expand",
            (
                "add",
                "expand",
                "include",
                "restore",
                "elaborate",
                "develop",
                "insert",
                "extend",
                "cover",
            ),
        ),
        (
            "compress",
            (
                "trim",
                "remove",
                "delete",
                "shorten",
                "compress",
                "cut",
                "drop",
                "reduce",
            ),
        ),
        (
            "repair",
            (
                "fix",
                "correct",
                "repair",
                "replace",
                "rewrite",
                "adjust",
                "clarify",
                "change",
                "tighten",
            ),
        ),
        ("verify", ("verify", "confirm", "check", "query", "ground", "fact")),
        ("resequence", ("reorder", "chronolog", "resequenc", "transition")),
    )
    _ACTION_STOPWORDS: ClassVar[frozenset[str]] = frozenset(
        {
            "add",
            "expand",
            "include",
            "restore",
            "elaborate",
            "develop",
            "insert",
            "extend",
            "cover",
            "trim",
            "remove",
            "delete",
            "shorten",
            "compress",
            "cut",
            "drop",
            "reduce",
            "fix",
            "correct",
            "repair",
            "replace",
            "rewrite",
            "adjust",
            "clarify",
            "change",
            "tighten",
            "verify",
            "confirm",
            "check",
            "query",
            "ground",
            "reorder",
            "chronology",
            "chronological",
            "transition",
        }
    )
    _GENERIC_REMEDY_TOKENS: ClassVar[frozenset[str]] = frozenset(
        {
            "line",
            "lines",
            "speaker",
            "speakers",
            "sentence",
            "sentences",
            "section",
            "sections",
            "phrase",
            "phrases",
            "clause",
            "clauses",
            "summary",
            "draft",
            "critic",
            "feedback",
            "specific",
            "exact",
            "following",
            "provide",
            "provided",
            "phrasing",
            "detail",
            "details",
            "task",
            "tasks",
            "use",
            "using",
            "make",
            "ensure",
            "need",
            "needs",
            "must",
            "should",
        }
    )
    _STOPWORDS: ClassVar[frozenset[str]] = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "for",
            "from",
            "if",
            "in",
            "into",
            "is",
            "it",
            "its",
            "of",
            "on",
            "or",
            "that",
            "the",
            "their",
            "then",
            "there",
            "these",
            "this",
            "to",
            "too",
            "was",
            "with",
            "you",
            "your",
        }
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
                signature = self._build_issue_signature(bucket=bucket, text=fragment)
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

    def extract_remedy_signatures(self, feedback: str) -> list[LoopRemedy]:
        """Extract deduplicated remedy patterns from critic feedback tasks."""
        normalized = self._normalize_text(feedback)
        if not normalized:
            return []

        remedies: dict[str, LoopRemedy] = {}
        for fragment in self._split_feedback_fragments(normalized):
            bucket = self._bucket_for_text(fragment)
            action = self._remedy_action_for_text(fragment=fragment, bucket=bucket)
            signature = self._build_remedy_signature(
                bucket=bucket,
                action=action,
                text=fragment,
            )
            remedies[signature] = LoopRemedy(
                signature=signature,
                bucket=bucket,
                action=action,
                description=self._compact_description(fragment),
            )
        return sorted(remedies.values(), key=lambda remedy: remedy.signature)

    def update_from_critic(
        self,
        *,
        iteration: int,
        critic_status: str,
        feedback: str,
        estimated_seconds: float | None,
    ) -> LoopMemoryUpdate:
        """Update unresolved issue and remedy-attempt memory from one critic pass."""
        del critic_status  # Status is informational; the normalized feedback drives memory.
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

        current_remedies = self.extract_remedy_signatures(feedback)
        repeated_remedy_signatures = sorted(
            remedy.signature
            for remedy in current_remedies
            if remedy.signature in self._attempted_remedies
        )
        for remedy in current_remedies:
            prior = self._attempted_remedies.get(remedy.signature)
            if prior is None:
                self._attempted_remedies[remedy.signature] = AttemptedRemedyState(
                    remedy=remedy,
                    first_seen_iteration=iteration,
                    last_seen_iteration=iteration,
                    attempt_count=1,
                )
                continue
            self._attempted_remedies[remedy.signature] = AttemptedRemedyState(
                remedy=remedy,
                first_seen_iteration=prior.first_seen_iteration,
                last_seen_iteration=iteration,
                attempt_count=prior.attempt_count + 1,
            )
        self._attempted_remedies = self._trim_attempted_remedies(self._attempted_remedies)

        self._unresolved = next_unresolved
        self._previous_unresolved_signatures = current_signatures
        self._last_estimated_seconds = estimated_seconds
        self._last_repeated_remedy_signatures = repeated_remedy_signatures

        strategy_shift_parts: list[str] = []
        if repeated_remedy_signatures:
            strategy_shift_parts.append(
                "Repeated critic remedy detected: do not repeat the same fix verbatim; "
                "change the edit strategy or gather fresh transcript evidence first."
            )
        if stagnation_detected:
            strategy_shift_parts.append(
                "Strategy shift required: stop repeating prior edits; make one "
                "substantial change targeting the top unresolved issue bucket."
            )
        strategy_shift_hint = " ".join(strategy_shift_parts).strip()
        self._last_strategy_shift_hint = strategy_shift_hint
        early_stop_recommended = (
            self._stagnation_streak >= self.early_stop_stagnation_threshold
        )

        return LoopMemoryUpdate(
            unresolved_issue_count=len(current_signatures),
            persisted_issue_signatures=persisted_signatures,
            stagnation_detected=stagnation_detected,
            strategy_shift_hint=strategy_shift_hint,
            early_stop_recommended=early_stop_recommended,
            stagnation_streak=self._stagnation_streak,
            repeated_remedy_detected=bool(repeated_remedy_signatures),
            repeated_remedy_signatures=repeated_remedy_signatures,
        )

    def to_compact_prompt_block(self) -> str:
        """Render bounded unresolved-issue and remedy-history guidance for prompts."""
        if not self._unresolved and not self._attempted_remedies:
            return ""

        lines: list[str] = []
        if self._unresolved:
            states = sorted(
                self._unresolved.values(),
                key=lambda state: (-state.recurrence_count, state.issue.signature),
            )[: self.max_unresolved_issues]
            lines.extend(
                [
                    (
                        "Unresolved issues from prior critic passes: "
                        f"{len(states)} active"
                    ),
                    "Prioritize these before stylistic edits and avoid repeated failures.",
                ]
            )
            for state in states:
                lines.append(
                    (
                        f"- [{state.issue.bucket}] {state.issue.description} "
                        f"(sig={state.issue.signature}, recurrence={state.recurrence_count})"
                    )
                )

        repeated_states = [
            state
            for state in self._attempted_remedies.values()
            if state.attempt_count > 1
            or state.remedy.signature in self._last_repeated_remedy_signatures
        ]
        if repeated_states:
            repeated_states = sorted(
                repeated_states,
                key=lambda state: (
                    -state.attempt_count,
                    -state.last_seen_iteration,
                    state.remedy.signature,
                ),
            )[: self.max_attempted_remedies]
            lines.extend(
                [
                    (
                        "Previously attempted remedy patterns that already failed "
                        "to converge:"
                    ),
                    "Do not replay these verbatim. Make a materially different edit.",
                ]
            )
            for state in repeated_states:
                lines.append(
                    (
                        f"- [{state.remedy.bucket}/{state.remedy.action}] "
                        f"{state.remedy.description} "
                        f"(sig={state.remedy.signature}, attempts={state.attempt_count})"
                    )
                )
        if self._last_repeated_remedy_signatures:
            lines.append(
                "Repeated remedy alert: the current critic feedback overlaps an "
                "already-attempted fix. Change the line selection, combine fixes, "
                "or verify against the transcript before editing."
            )
        if self._last_strategy_shift_hint:
            lines.append(f"Strategy shift: {self._last_strategy_shift_hint}")
        if self._stagnation_streak > 0:
            lines.append(f"Stagnation streak: {self._stagnation_streak}")

        return build_loop_context_prompt_block(
            "\n".join(lines), char_cap=self.prompt_char_limit
        )

    def load_artifact(
        self,
        artifact_path: Path,
        *,
        context: Mapping[str, str] | None = None,
    ) -> bool:
        """Load loop memory from a JSON artifact when the scope matches."""
        if not artifact_path.exists():
            return False
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(payload, Mapping):
            return False

        expected_context = self._normalize_context(context)
        payload_context = self._normalize_context(payload.get("context", {}))
        if expected_context and payload_context != expected_context:
            return False

        memory_payload = payload.get("memory", {})
        if not isinstance(memory_payload, Mapping):
            return False

        self._unresolved = self._deserialize_unresolved(memory_payload.get("unresolved"))
        previous_signatures = memory_payload.get("previous_unresolved_signatures", [])
        if isinstance(previous_signatures, list):
            self._previous_unresolved_signatures = frozenset(
                str(signature).strip()
                for signature in previous_signatures
                if str(signature).strip()
            )
        else:
            self._previous_unresolved_signatures = frozenset()
        self._last_estimated_seconds = self._coerce_optional_float(
            memory_payload.get("last_estimated_seconds")
        )
        self._stagnation_streak = self._coerce_non_negative_int(
            memory_payload.get("stagnation_streak"),
            default=0,
        )
        self._last_strategy_shift_hint = str(
            memory_payload.get("last_strategy_shift_hint", "")
        ).strip()
        self._attempted_remedies = self._deserialize_attempted_remedies(
            memory_payload.get("attempted_remedies")
        )
        repeated_signatures = memory_payload.get("last_repeated_remedy_signatures", [])
        if isinstance(repeated_signatures, list):
            self._last_repeated_remedy_signatures = [
                str(signature).strip()
                for signature in repeated_signatures
                if str(signature).strip()
            ]
        else:
            self._last_repeated_remedy_signatures = []
        return True

    def save_artifact(
        self,
        artifact_path: Path,
        *,
        context: Mapping[str, str] | None = None,
    ) -> None:
        """Persist loop memory into a JSON artifact."""
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "context": self._normalize_context(context),
            "memory": {
                "unresolved": self._serialize_unresolved(),
                "previous_unresolved_signatures": sorted(
                    self._previous_unresolved_signatures
                ),
                "last_estimated_seconds": self._last_estimated_seconds,
                "stagnation_streak": self._stagnation_streak,
                "last_strategy_shift_hint": self._last_strategy_shift_hint,
                "attempted_remedies": self._serialize_attempted_remedies(),
                "last_repeated_remedy_signatures": self._last_repeated_remedy_signatures,
            },
        }
        temp_path = artifact_path.with_name(f"{artifact_path.name}.tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(artifact_path)

    @staticmethod
    def _normalize_text(feedback: str) -> str:
        """Normalize free-form feedback for deterministic parsing."""
        return " ".join(feedback.strip().lower().split())

    @staticmethod
    def _split_feedback_fragments(normalized_feedback: str) -> list[str]:
        """Split free-form feedback into short deterministic fragments."""
        fragments = [
            part.strip(" -.;,:[]()")
            for part in re.split(
                r"[.;\n]|(?:\s+-\s+)|(?:\s+and\s+)",
                normalized_feedback,
            )
            if part.strip(" -.;,:[]()")
        ]
        return fragments or [normalized_feedback]

    def _bucket_for_text(self, text: str) -> IssueBucket:
        """Map one feedback fragment to a deterministic issue bucket."""
        if (
            "duration" in text
            or "seconds" in text
            or "too short" in text
            or "too long" in text
            or "below minimum" in text
            or "above maximum" in text
        ):
            return "duration"
        for bucket, keywords in self._BUCKET_KEYWORDS:
            if any(keyword in text for keyword in keywords):
                return bucket
        return "quality"

    @staticmethod
    def _compact_description(fragment: str) -> str:
        """Build a concise issue or remedy description from a feedback fragment."""
        tokens = fragment.split()
        return " ".join(tokens[:14])

    @staticmethod
    def _build_issue_signature(*, bucket: IssueBucket, text: str) -> str:
        """Build a deterministic unresolved-issue signature from one fragment."""
        token_source = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [token for token in token_source.split() if token]
        signature_tokens = tokens[:8] if tokens else ["unspecified"]
        return f"{bucket}:{'_'.join(signature_tokens)}"

    def _build_remedy_signature(
        self,
        *,
        bucket: IssueBucket,
        action: RemedyAction,
        text: str,
    ) -> str:
        """Build a remedy signature that clusters similarly worded critic fixes."""
        focus_tokens = self._extract_focus_tokens(text)
        signature_tokens = focus_tokens[:4] if focus_tokens else ["generic"]
        return f"{bucket}:{action}:{'_'.join(signature_tokens)}"

    def _remedy_action_for_text(
        self,
        *,
        fragment: str,
        bucket: IssueBucket,
    ) -> RemedyAction:
        """Infer the high-level remedy action requested by the critic."""
        if bucket == "duration":
            if any(
                keyword in fragment
                for keyword in (
                    "too short",
                    "below minimum",
                    "add",
                    "expand",
                    "include",
                    "restore",
                )
            ):
                return "expand"
            if any(
                keyword in fragment
                for keyword in (
                    "too long",
                    "above maximum",
                    "trim",
                    "remove",
                    "delete",
                    "shorten",
                    "compress",
                    "cut",
                )
            ):
                return "compress"

        for action, keywords in self._ACTION_KEYWORDS:
            if any(keyword in fragment for keyword in keywords):
                return action
        return "other"

    def _extract_focus_tokens(self, text: str) -> list[str]:
        """Extract stable content tokens for remedy signature clustering."""
        normalized = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens: list[str] = []
        seen: set[str] = set()
        for raw_token in normalized.split():
            if raw_token.isdigit():
                continue
            token = self._normalize_token(raw_token)
            if (
                not token
                or token in self._STOPWORDS
                or token in self._ACTION_STOPWORDS
                or token in self._GENERIC_REMEDY_TOKENS
            ):
                continue
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Normalize one token for coarse similarity matching."""
        normalized = token.strip().lower()
        if len(normalized) > 5 and normalized.endswith("ing"):
            normalized = normalized[:-3]
        elif len(normalized) > 4 and normalized.endswith("ed"):
            normalized = normalized[:-2]
        elif len(normalized) > 4 and normalized.endswith("es"):
            normalized = normalized[:-2]
        elif len(normalized) > 3 and normalized.endswith("s"):
            normalized = normalized[:-1]
        return normalized

    def _trim_attempted_remedies(
        self,
        remedies: dict[str, AttemptedRemedyState],
    ) -> dict[str, AttemptedRemedyState]:
        """Keep remedy memory bounded to the most relevant recent patterns."""
        if len(remedies) <= self.max_attempted_remedies:
            return dict(remedies)
        kept_states = sorted(
            remedies.values(),
            key=lambda state: (
                -state.attempt_count,
                -state.last_seen_iteration,
                state.remedy.signature,
            ),
        )[: self.max_attempted_remedies]
        return {state.remedy.signature: state for state in kept_states}

    def _duration_issue(self, estimated_seconds: float) -> LoopIssue | None:
        """Return a deterministic duration issue when the current draft is out of bounds."""
        tolerance_seconds = self.target_seconds * self.duration_tolerance_ratio
        min_seconds = self.target_seconds - tolerance_seconds
        max_seconds = self.target_seconds + tolerance_seconds
        rounded_seconds = round(estimated_seconds, 2)
        if estimated_seconds < min_seconds:
            return LoopIssue(
                signature=f"duration:below_min_{round(min_seconds, 2)}",
                bucket="duration",
                description=(
                    f"estimated duration {rounded_seconds}s below minimum "
                    f"{round(min_seconds, 2)}s; expand coverage"
                ),
            )
        if estimated_seconds > max_seconds:
            return LoopIssue(
                signature=f"duration:above_max_{round(max_seconds, 2)}",
                bucket="duration",
                description=(
                    f"estimated duration {rounded_seconds}s above maximum "
                    f"{round(max_seconds, 2)}s; compress content"
                ),
            )
        return None

    def _serialize_unresolved(self) -> list[dict[str, object]]:
        """Serialize unresolved issue state into JSON-safe payloads."""
        return [
            {
                "signature": state.issue.signature,
                "bucket": state.issue.bucket,
                "description": state.issue.description,
                "first_seen_iteration": state.first_seen_iteration,
                "last_seen_iteration": state.last_seen_iteration,
                "recurrence_count": state.recurrence_count,
            }
            for state in sorted(
                self._unresolved.values(),
                key=lambda value: value.issue.signature,
            )
        ]

    def _serialize_attempted_remedies(self) -> list[dict[str, object]]:
        """Serialize attempted remedy history into JSON-safe payloads."""
        return [
            {
                "signature": state.remedy.signature,
                "bucket": state.remedy.bucket,
                "action": state.remedy.action,
                "description": state.remedy.description,
                "first_seen_iteration": state.first_seen_iteration,
                "last_seen_iteration": state.last_seen_iteration,
                "attempt_count": state.attempt_count,
            }
            for state in sorted(
                self._attempted_remedies.values(),
                key=lambda value: value.remedy.signature,
            )
        ]

    def _deserialize_unresolved(
        self,
        payload: object,
    ) -> dict[str, UnresolvedIssueState]:
        """Parse unresolved issue state from artifact payload."""
        if not isinstance(payload, list):
            return {}
        loaded: dict[str, UnresolvedIssueState] = {}
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            signature = str(item.get("signature", "")).strip()
            bucket = str(item.get("bucket", "")).strip()
            description = str(item.get("description", "")).strip()
            if not signature or bucket not in _ISSUE_BUCKET_VALUES:
                continue
            issue = LoopIssue(
                signature=signature,
                bucket=bucket,  # type: ignore[arg-type]
                description=description,
            )
            loaded[signature] = UnresolvedIssueState(
                issue=issue,
                first_seen_iteration=self._coerce_non_negative_int(
                    item.get("first_seen_iteration"),
                    default=0,
                ),
                last_seen_iteration=self._coerce_non_negative_int(
                    item.get("last_seen_iteration"),
                    default=0,
                ),
                recurrence_count=max(
                    1,
                    self._coerce_non_negative_int(
                        item.get("recurrence_count"),
                        default=1,
                    ),
                ),
            )
        return loaded

    def _deserialize_attempted_remedies(
        self,
        payload: object,
    ) -> dict[str, AttemptedRemedyState]:
        """Parse attempted remedy state from artifact payload."""
        if not isinstance(payload, list):
            return {}
        loaded: dict[str, AttemptedRemedyState] = {}
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            signature = str(item.get("signature", "")).strip()
            bucket = str(item.get("bucket", "")).strip()
            action = str(item.get("action", "")).strip()
            description = str(item.get("description", "")).strip()
            if (
                not signature
                or bucket not in _ISSUE_BUCKET_VALUES
                or action not in _REMEDY_ACTION_VALUES
            ):
                continue
            remedy = LoopRemedy(
                signature=signature,
                bucket=bucket,  # type: ignore[arg-type]
                action=action,  # type: ignore[arg-type]
                description=description,
            )
            loaded[signature] = AttemptedRemedyState(
                remedy=remedy,
                first_seen_iteration=self._coerce_non_negative_int(
                    item.get("first_seen_iteration"),
                    default=0,
                ),
                last_seen_iteration=self._coerce_non_negative_int(
                    item.get("last_seen_iteration"),
                    default=0,
                ),
                attempt_count=max(
                    1,
                    self._coerce_non_negative_int(
                        item.get("attempt_count"),
                        default=1,
                    ),
                ),
            )
        return self._trim_attempted_remedies(loaded)

    @staticmethod
    def _normalize_context(context: Mapping[str, str] | object) -> dict[str, str]:
        """Normalize persistence context into stable string key-value pairs."""
        if not isinstance(context, Mapping):
            return {}
        normalized: dict[str, str] = {}
        for key, value in context.items():
            key_text = str(key).strip()
            value_text = str(value).strip()
            if not key_text or not value_text:
                continue
            normalized[key_text] = value_text
        return normalized

    @staticmethod
    def _coerce_non_negative_int(value: object, *, default: int) -> int:
        """Parse a non-negative integer with fallback."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(0, parsed)

    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        """Parse an optional float value from artifact payload."""
        if value in {None, ""}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
