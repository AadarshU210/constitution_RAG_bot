"""RAG chat orchestration: retrieve → rerank → generate."""

from __future__ import annotations

from app.retrieval.pipeline import retrieve
from app.schemas.chat import ChatResponse, SourceItem
from app.services.generator import generate_answer


def _to_sources(chunks: list[dict]) -> list[SourceItem]:
    sources: list[SourceItem] = []
    for chunk in chunks:
        text = chunk.get("text") or ""
        excerpt = text if len(text) <= 400 else text[:400].rstrip() + "…"
        sources.append(
            SourceItem(
                chunk_id=chunk.get("chunk_id"),
                article=chunk.get("article"),
                article_title=chunk.get("article_title"),
                part=chunk.get("part"),
                part_title=chunk.get("part_title"),
                page_start=chunk.get("page_start"),
                page_end=chunk.get("page_end"),
                rerank_score=chunk.get("rerank_score"),
                excerpt=excerpt,
            )
        )
    return sources


async def answer_question(question: str) -> ChatResponse:
    cleaned = question.strip()
    if not cleaned:
        return ChatResponse(
            answer="Please ask a non-empty question about the Constitution of India.",
            sources=[],
            retrieved_count=0,
        )

    chunks = retrieve(cleaned)
    answer = await generate_answer(cleaned, chunks)
    return ChatResponse(
        answer=answer,
        sources=_to_sources(chunks),
        retrieved_count=len(chunks),
    )
