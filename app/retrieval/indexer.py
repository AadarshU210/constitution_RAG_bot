"""Build FAISS + BM25 indexes from constitution_articles.json."""

from __future__ import annotations

import json
from pathlib import Path

import faiss

from app.core.config import settings
from app.retrieval.bm25_index import build_bm25, save_bm25
from app.retrieval.embedder import embed_passages, load_embedder, passage_text
from app.retrieval.store import IndexStore


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = [
        {
            "chunk_id": c.get("chunk_id"),
            "part": c.get("part"),
            "part_title": c.get("part_title"),
            "article": c.get("article"),
            "article_title": c.get("article_title"),
            "text": c.get("text"),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "chunk_type": c.get("chunk_type"),
        }
        for c in chunks
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)


def build_indexes(
    articles_path: Path | None = None,
    index_dir: Path | None = None,
) -> None:
    articles_path = articles_path or (settings.DATA_PROCESSED_DIR / "constitution_articles.json")
    index_dir = index_dir or settings.DATA_INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    if not articles_path.exists():
        raise FileNotFoundError(
            f"Missing {articles_path}. Run: uv run python -m app.ingestion.structure"
        )

    chunks = load_chunks(articles_path)
    texts = [passage_text(c) for c in chunks]
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding model: {settings.EMBEDDING_MODEL}")

    model = load_embedder()
    vectors = embed_passages(model, texts).astype("float32")
    dim = vectors.shape[1]
    print(f"Vectors: {vectors.shape}")

    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)
    faiss_path = index_dir / "faiss.index"
    faiss.write_index(faiss_index, str(faiss_path))
    print(f"Wrote {faiss_path}")

    save_metadata(chunks, index_dir / "metadata.json")
    print(f"Wrote {index_dir / 'metadata.json'}")

    bm25 = build_bm25(texts)
    save_bm25(bm25, index_dir / "bm25.pkl")
    print(f"Wrote {index_dir / 'bm25.pkl'}")


def smoke_test() -> None:
    store = IndexStore()
    store.load()
    queries = [
        "right to life and personal liberty",
        "Article 21",
        "equality before law",
    ]
    for query in queries:
        print(f"\n=== Query: {query} ===")
        print("Dense:")
        for row in store.format_hits(store.search_dense(query, top_k=3)):
            print(f"  {row['chunk_id']:16} art={row['article']}  {row['article_title']}")
        print("BM25:")
        for row in store.format_hits(store.search_sparse(query, top_k=3)):
            print(f"  {row['chunk_id']:16} art={row['article']}  {row['article_title']}")
        print("Hybrid:")
        for row in store.format_hits(store.search_hybrid(query, top_k=3)):
            print(f"  {row['chunk_id']:16} art={row['article']}  {row['article_title']}")


def main() -> None:
    build_indexes()
    smoke_test()


if __name__ == "__main__":
    main()
