"""Force-include exact Article chunks when the user names them."""

from __future__ import annotations

from app.retrieval.query_utils import extract_article_numbers
from app.retrieval.store import IndexStore


def indices_for_articles(store: IndexStore, article_nos: list[str]) -> list[int]:
    wanted = {a.upper() for a in article_nos}
    hits: list[int] = []
    for idx, chunk in enumerate(store.metadata):
        art = chunk.get("article")
        if art is None:
            continue
        if str(art).upper() in wanted:
            hits.append(idx)
    return hits


def merge_exact_article_hits(
    query: str,
    hybrid_hits: list[tuple[int, float]],
    store: IndexStore,
    top_k: int,
) -> list[tuple[int, float]]:
    """Prepend exact metadata matches so named articles survive the candidate pool."""
    refs = extract_article_numbers(query)
    if not refs:
        return hybrid_hits[:top_k]

    exact = indices_for_articles(store, refs)
    if not exact:
        return hybrid_hits[:top_k]

    boosted: dict[int, float] = {idx: 2.0 + (0.01 * i) for i, idx in enumerate(exact)}
    for idx, score in hybrid_hits:
        if idx not in boosted:
            boosted[idx] = float(score)

    ranked = sorted(boosted.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def pin_exact_articles(
    query: str,
    ranked: list[dict],
    store: IndexStore,
    top_k: int,
) -> list[dict]:
    """Keep named Articles in the final list even if the reranker ranks them lower."""
    refs = extract_article_numbers(query)
    if not refs:
        return ranked[:top_k]

    by_art: dict[str, dict] = {}
    for idx, chunk in enumerate(store.metadata):
        art = chunk.get("article")
        if art is None:
            continue
        key = str(art).upper()
        if key not in by_art:
            item = dict(chunk)
            item["index"] = idx
            by_art[key] = item

    pinned: list[dict] = []
    for ref in refs:
        key = ref.upper()
        if key not in by_art:
            continue
        # Prefer the already-reranked copy when present.
        existing = next(
            (c for c in ranked if str(c.get("article") or "").upper() == key),
            None,
        )
        if existing is not None:
            pinned.append(existing)
            continue
        item = dict(by_art[key])
        ceiling = max((c.get("rerank_score") or 0.0) for c in ranked) if ranked else 0.0
        item["rerank_score"] = float(ceiling) + 1.0
        item["hybrid_score"] = float(item.get("hybrid_score") or 2.0)
        pinned.append(item)

    if not pinned:
        return ranked[:top_k]

    pinned_keys = {str(p.get("article")).upper() for p in pinned}
    rest = [c for c in ranked if str(c.get("article") or "").upper() not in pinned_keys]
    return (pinned + rest)[:top_k]
