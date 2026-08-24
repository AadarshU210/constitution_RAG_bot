from app.retrieval.bm25_index import tokenize
from app.retrieval.embedder import passage_text
from app.retrieval.store import rrf_merge


def test_tokenize() -> None:
    tokens = tokenize("Article 21 Protection of life")
    assert "article" in tokens
    assert "21" in tokens
    assert "protection" in tokens


def test_passage_text_includes_article() -> None:
    text = passage_text(
        {
            "article": "21",
            "article_title": "Protection of life and personal liberty",
            "part": "PART III",
            "part_title": "FUNDAMENTAL RIGHTS",
            "text": "No person shall be deprived of his life...",
        }
    )
    assert "Article 21" in text
    assert "FUNDAMENTAL RIGHTS" in text
    assert "deprived" in text


def test_rrf_merge_prefers_overlap() -> None:
    dense = [(10, 0.9), (21, 0.8), (3, 0.1)]
    sparse = [(21, 12.0), (10, 5.0), (7, 1.0)]
    merged = rrf_merge(dense, sparse, top_k=3)
    ids = [idx for idx, _ in merged]
    assert ids[0] in {10, 21}
    assert 21 in ids
    assert 10 in ids
