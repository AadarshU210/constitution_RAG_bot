from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceItem(BaseModel):
    chunk_id: str | None = None
    article: str | None = None
    article_title: str | None = None
    part: str | None = None
    part_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    rerank_score: float | None = None
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    retrieved_count: int
