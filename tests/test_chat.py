from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.reranking.reranker import rerank
from app.schemas.chat import ChatResponse, SourceItem
from app.services.generator import build_messages

client = TestClient(app)


def test_rerank_orders_by_score() -> None:
    model = MagicMock()
    model.predict.return_value = [0.1, 0.9, 0.4]
    candidates = [
        {"chunk_id": "a", "text": "low"},
        {"chunk_id": "b", "text": "high"},
        {"chunk_id": "c", "text": "mid"},
    ]
    ranked = rerank("query", candidates, model, top_k=2)
    assert [c["chunk_id"] for c in ranked] == ["b", "c"]
    assert ranked[0]["rerank_score"] == pytest.approx(0.9)


def test_build_messages_includes_article_context() -> None:
    chunks = [
        {
            "article": "21",
            "article_title": "Protection of life and personal liberty",
            "part": "PART III",
            "page_start": 42,
            "page_end": 42,
            "text": "No person shall be deprived of his life...",
            "chunk_id": "article_21",
        }
    ]
    messages = build_messages("What is Article 21?", chunks)
    assert messages[0]["role"] == "system"
    assert "Article 21" in messages[1]["content"]
    assert "No person shall be deprived" in messages[1]["content"]


def test_chat_endpoint_success() -> None:
    fake = ChatResponse(
        answer="Article 21 protects life and personal liberty.",
        sources=[
            SourceItem(
                chunk_id="article_21",
                article="21",
                article_title="Protection of life and personal liberty",
                part="PART III",
                part_title="FUNDAMENTAL RIGHTS",
                page_start=42,
                page_end=42,
                rerank_score=1.2,
                excerpt="No person shall be deprived...",
            )
        ],
        retrieved_count=1,
    )
    with patch(
        "app.api.v1.router.chat_service.answer_question",
        new=AsyncMock(return_value=fake),
    ):
        response = client.post("/api/v1/chat", json={"question": "What is Article 21?"})
    assert response.status_code == 200
    body = response.json()
    assert "Article 21" in body["answer"]
    assert body["retrieved_count"] == 1
    assert body["sources"][0]["article"] == "21"


def test_chat_endpoint_rejects_empty_question() -> None:
    response = client.post("/api/v1/chat", json={"question": ""})
    assert response.status_code == 422
