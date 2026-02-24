from typing import List

from pydantic import BaseModel, Field


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
    full_transcript: str = ""


class SummarizerTaskResponse(BaseModel):
    summary_xml: str


# Critic DTOs
class CriticTaskRequest(BaseModel):
    draft: str
    min_words: int = 50
    max_words: int = 100
    full_transcript: str = ""


class CriticTaskResponse(BaseModel):
    status: str
    word_count: int
    feedback: str


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
