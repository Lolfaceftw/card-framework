from typing import Any, List, Literal, cast

from pydantic import BaseModel, Field, field_validator


# Retrieval DTOs
class SegmentDTO(BaseModel):
    speaker: str = "UNKNOWN"
    text: str = ""


class IndexTaskRequest(BaseModel):
    action: str = Field(pattern="^index$")
    segments: List[SegmentDTO]


class IndexTaskResponse(BaseModel):
    status: str
    count: int


class RetrieveTaskRequest(BaseModel):
    action: str = Field(pattern="^retrieve$")
    query: str
    top_k: int = 15
    lambda_param: float = 0.7


class RetrieveTaskResponse(BaseModel):
    segments: List[SegmentDTO]
    total_words: int


# Summarizer DTOs
class SummarizerTaskRequest(BaseModel):
    min_words: int = 50
    max_words: int = 100
    retrieval_port: int
    feedback: str = ""
    previous_draft: str = ""
    loop_context: str = ""
    full_transcript: str = ""


class SummarizerTaskResponse(BaseModel):
    summary_xml: str


# Critic DTOs
class CriticTaskRequest(BaseModel):
    draft: str
    min_words: int = 50
    max_words: int = 100
    full_transcript: str = ""


CriticStatus = Literal["pass", "fail"]


class CriticTaskResponse(BaseModel):
    status: CriticStatus
    word_count: int
    feedback: str

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: object) -> CriticStatus:
        """Normalize critic status and reject unknown values."""
        if not isinstance(value, str):
            raise TypeError("status must be a string")
        normalized = value.strip().lower()
        if normalized not in {"pass", "fail"}:
            raise ValueError("status must be 'pass' or 'fail'")
        return cast(CriticStatus, normalized)


# QA benchmark DTOs
class GroundTruthCreatorTaskRequest(BaseModel):
    """Request payload for source-grounded QA question generation."""

    source_text: str
    factual_question_count: int = 50
    naturalness_question_count: int = 50


class GroundTruthCreatorTaskResponse(BaseModel):
    """Response payload for generated QA ground-truth questions."""

    questions: list[dict[str, Any]]


class QAEvaluatorTaskRequest(BaseModel):
    """Request payload for QA evaluator execution."""

    summary_xml: str
    source_text: str
    questions: list[dict[str, Any]]


class QAEvaluatorTaskResponse(BaseModel):
    """Response payload for QA evaluator scoring and trace records."""

    score: dict[str, Any]
    answers: list[dict[str, Any]]


CorrectorTargetAgent = Literal["GroundTruthCreator", "Evaluator"]


class CorrectorFewShotExample(BaseModel):
    """One bad-to-good correction example produced by Corrector."""

    bad_example: str
    corrected_example: str
    rationale: str


class CorrectorTaskRequest(BaseModel):
    """Request payload for corrective guidance generation."""

    target_agent: CorrectorTargetAgent
    failure_type: str
    failure_context: str
    latest_output: str = ""
    attempt: int = 1
    max_attempts: int = 1
    expected_contract: str = ""


class CorrectorTaskResponse(BaseModel):
    """Response payload containing correction instruction and examples."""

    correction_instruction: str
    few_shot_examples: list[CorrectorFewShotExample] = Field(default_factory=list)


# Common LLM DTOs
class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Function

    def model_dump(self, *args, **kwargs) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: str = ""
    tool_calls: List[ToolCall] | None = None
    reasoning_content: str | None = None

    def model_dump(self, *args, **kwargs) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        return d
