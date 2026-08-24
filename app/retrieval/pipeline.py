"""Retrieve pipeline: hybrid search then cross-encoder rerank."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.reranking.reranker import rerank
from app.retrieval.article_boost import merge_exact_article_hits, pin_exact_articles
from app.retrieval.query_utils import build_retrieval_query
from app.retrieval.runtime import get_reranker, get_store
from app.retrieval.store import IndexStore


def retrieve(
    query: str,
    *,
    store: IndexStore | None = None,
    reranker: CrossEncoder | None = None,
    fetch_k: int | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Return top reranked chunks (full metadata + text + scores)."""
    store = store or get_store()
    reranker = reranker or get_reranker()
    fetch_k = fetch_k if fetch_k is not None else settings.RETRIEVE_TOP_K
    top_k = top_k if top_k is not None else settings.RERANK_TOP_K

    search_query = build_retrieval_query(query)
    hits = store.search_hybrid(search_query, top_k=fetch_k, fetch_k=max(fetch_k, 20))
    hits = merge_exact_article_hits(search_query, hits, store, top_k=fetch_k)

    candidates: list[dict] = []
    for idx, hybrid_score in hits:
        chunk = dict(store.metadata[idx])
        chunk["hybrid_score"] = float(hybrid_score)
        chunk["index"] = idx
        candidates.append(chunk)

    ranked = rerank(search_query, candidates, reranker, top_k=top_k)
    return pin_exact_articles(search_query, ranked, store, top_k=top_k)
