from app.retrieval.article_boost import merge_exact_article_hits
from app.retrieval.query_utils import build_retrieval_query, extract_article_numbers


def test_extract_article_numbers() -> None:
    assert extract_article_numbers("What is Article 21?") == ["21"]
    assert extract_article_numbers("Compare Article 14 and Art. 15") == ["14", "15"]
    assert extract_article_numbers("Tell me about article 21A") == ["21A"]
    assert extract_article_numbers("How to cook pasta?") == []


def test_build_retrieval_query_short_passthrough() -> None:
    q = "What does Article 14 say?"
    assert build_retrieval_query(q) == q


def test_build_retrieval_query_truncates_noise() -> None:
    noise = ("pad " * 400) + "What is Article 14 about equality?"
    out = build_retrieval_query(noise)
    assert len(out) < len(noise)
    assert "Article 14" in out


def test_merge_exact_article_hits_prefers_named() -> None:
    class FakeStore:
        metadata = [
            {"article": "13"},
            {"article": "12"},
            {"article": "14"},
        ]

    hybrid = [(0, 0.9), (2, 0.8)]  # 13 and 14
    merged = merge_exact_article_hits("What is Article 12?", hybrid, FakeStore(), top_k=3)
    ids = [idx for idx, _ in merged]
    assert ids[0] == 1  # article 12 forced first
    assert 1 in ids


def test_pin_exact_articles_inserts_missing() -> None:
    from app.retrieval.article_boost import pin_exact_articles

    class FakeStore:
        metadata = [
            {"article": "13", "text": "thirteen", "chunk_id": "a13"},
            {"article": "12", "text": "twelve", "chunk_id": "a12"},
        ]

    ranked = [
        {"article": "13", "text": "thirteen", "chunk_id": "a13", "rerank_score": 0.9},
        {"article": "4", "text": "four", "chunk_id": "a4", "rerank_score": 0.8},
    ]
    pinned = pin_exact_articles("Explain Article 12", ranked, FakeStore(), top_k=2)
    assert pinned[0]["article"] == "12"
    assert len(pinned) == 2
