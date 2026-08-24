"""Lazy-loaded runtime singletons for index + reranker."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.reranking.reranker import load_reranker
from app.retrieval.store import IndexStore

_store: IndexStore | None = None
_reranker: CrossEncoder | None = None


def get_store() -> IndexStore:
    global _store
    if _store is None:
        store = IndexStore()
        store.load()
        _store = store
    return _store


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = load_reranker()
    return _reranker


def reset_runtime() -> None:
    """Clear singletons (useful in tests)."""
    global _store, _reranker
    _store = None
    _reranker = None
