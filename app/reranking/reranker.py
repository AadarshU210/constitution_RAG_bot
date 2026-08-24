"""Cross-encoder reranking for retrieved Constitution chunks."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.core.config import settings


def load_reranker(model_name: str | None = None) -> CrossEncoder:
    name = model_name or settings.RERANKER_MODEL
    return CrossEncoder(name)


def rerank(
    query: str,
    candidates: list[dict],
    model: CrossEncoder,
    top_k: int | None = None,
    text_key: str = "text",
) -> list[dict]:
    """Score (query, chunk) pairs and return top_k candidates with rerank_score."""
    if not candidates:
        return []

    k = top_k if top_k is not None else settings.RERANK_TOP_K
    pairs = [(query, c.get(text_key) or "") for c in candidates]
    scores = model.predict(pairs)

    ranked: list[dict] = []
    for candidate, score in zip(candidates, scores, strict=True):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        ranked.append(item)

    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:k]
