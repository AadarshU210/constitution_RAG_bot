"""Embedding helpers for Constitution chunks."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.core.config import settings

# bge-small/base: documents are encoded as-is; queries get this prefix.
QUERY_PREFIX = "Represent this sentence for retrieving relevant passages: "


def passage_text(chunk: dict) -> str:
    """Text we embed: article identity + body."""
    parts: list[str] = []
    if chunk.get("article"):
        title = chunk.get("article_title") or ""
        parts.append(f"Article {chunk['article']}: {title}".strip())
    elif chunk.get("article_title"):
        parts.append(str(chunk["article_title"]))
    if chunk.get("part"):
        part_line = chunk["part"]
        if chunk.get("part_title"):
            part_line = f"{part_line} — {chunk['part_title']}"
        parts.append(part_line)
    parts.append(chunk.get("text") or "")
    return "\n".join(parts)


def load_embedder(model_name: str | None = None) -> SentenceTransformer:
    name = model_name or settings.EMBEDDING_MODEL
    return SentenceTransformer(name)


def embed_passages(model: SentenceTransformer, texts: list[str]):
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_query(model: SentenceTransformer, query: str):
    return model.encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
