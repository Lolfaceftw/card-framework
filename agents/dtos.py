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
