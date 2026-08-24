"""BM25 keyword index over Constitution chunks."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_bm25(texts: list[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(t) for t in texts])


def save_bm25(bm25: BM25Okapi, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bm25, f)


def load_bm25(path: Path) -> BM25Okapi:
    with path.open("rb") as f:
        return pickle.load(f)


def search_bm25(bm25: BM25Okapi, query: str, top_k: int = 5) -> list[tuple[int, float]]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
