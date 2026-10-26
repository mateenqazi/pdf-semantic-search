from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

DocumentId = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class Document(BaseModel):
    document_id: DocumentId
    filename: str
    pages: int
    chunks: int


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)
    document_id: DocumentId | None = None
    min_score: float = Field(default=-1.0, ge=-1.0, le=1.0, allow_inf_nan=False)


class SearchHit(BaseModel):
    document_id: DocumentId
    filename: str
    page: int
    chunk_id: int
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchHit]
