"""Load FAISS + BM25 indexes and run dense / sparse / hybrid search."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.retrieval.bm25_index import load_bm25, search_bm25
from app.retrieval.embedder import embed_query, load_embedder


def rrf_merge(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Reciprocal rank fusion over two ranked id lists."""
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(sparse, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


class IndexStore:
    def __init__(
        self,
        index_dir: Path | None = None,
        embedder: SentenceTransformer | None = None,
    ) -> None:
        self.index_dir = index_dir or settings.DATA_INDEX_DIR
        self.metadata: list[dict] = []
        self.faiss_index = None
        self.bm25: BM25Okapi | None = None
        self.embedder = embedder

    def load(self) -> None:
        meta_path = self.index_dir / "metadata.json"
        faiss_path = self.index_dir / "faiss.index"
        bm25_path = self.index_dir / "bm25.pkl"
        if not meta_path.exists() or not faiss_path.exists() or not bm25_path.exists():
            raise FileNotFoundError(
                f"Missing index files in {self.index_dir.resolve()}. "
                "Run: uv run python -m app.retrieval.indexer"
            )
        with meta_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.faiss_index = faiss.read_index(str(faiss_path))
        self.bm25 = load_bm25(bm25_path)
        if self.embedder is None:
            self.embedder = load_embedder()

    def search_dense(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        assert self.faiss_index is not None and self.embedder is not None
        vector = embed_query(self.embedder, query).astype("float32")
        scores, ids = self.faiss_index.search(vector, top_k)
        return [
            (int(idx), float(score))
            for idx, score in zip(ids[0], scores[0], strict=False)
            if idx >= 0
        ]

    def search_sparse(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        assert self.bm25 is not None
        return search_bm25(self.bm25, query, top_k=top_k)

    def search_hybrid(
        self, query: str, top_k: int = 5, fetch_k: int = 20
    ) -> list[tuple[int, float]]:
        dense = self.search_dense(query, top_k=fetch_k)
        sparse = self.search_sparse(query, top_k=fetch_k)
        return rrf_merge(dense, sparse, top_k=top_k)

    def format_hits(self, hits: list[tuple[int, float]]) -> list[dict]:
        rows = []
        for idx, score in hits:
            chunk = self.metadata[idx]
            rows.append(
                {
                    "score": round(score, 4),
                    "chunk_id": chunk.get("chunk_id"),
                    "article": chunk.get("article"),
                    "article_title": chunk.get("article_title"),
                    "part": chunk.get("part"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                }
            )
        return rows
