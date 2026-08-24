# Samvidhaan

**Samvidhaan** is a Constitution-of-India–only RAG chatbot. It answers from indexed constitutional text and cites Article / Part / page sources — it does not invent law.

> Ask the Constitution. Nothing beyond the text.

## Features

- Structure-aware article chunks (1 Article ≈ 1 chunk)
- Hybrid retrieval: FAISS dense search + BM25, fused with RRF
- Cross-encoder reranking
- Exact Article pinning when the user names an Article (e.g. 21, 368)
- Grounded LLM answers via any OpenAI-compatible API (Ollama, Groq, …)
- Chat UI at `/` plus `POST /api/v1/chat`

## Stack

| Layer | Choice |
|-------|--------|
| API / UI host | FastAPI |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector index | FAISS (local) |
| Sparse index | BM25 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | OpenAI-compatible (`LLM_BASE_URL` / `LLM_MODEL`) |
| Package mgmt | [uv](https://github.com/astral-sh/uv) · Python 3.11+ |

## Quick start

```bash
git clone https://github.com/AadarshU210/constitution_RAG_bot.git
cd constitution_RAG_bot
uv sync
```

1. Copy env template and edit secrets locally (never commit `.env`):

```bash
cp .env.example .env
```

2. Place the Constitution PDF at `data/raw/constitution_of_india.pdf` (gitignored).

3. Build (or rebuild) indexes if needed:

```bash
uv run python -m app.ingestion.parser
uv run python -m app.ingestion.structure
uv run python -m app.retrieval.indexer
```

This repo may already include `data/processed/` and `data/index/` so you can skip rebuild on first run.

4. Start the server:

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

| URL | What |
|-----|------|
| http://127.0.0.1:8000/ | **Samvidhaan** chat UI |
| http://127.0.0.1:8000/docs | OpenAPI / Swagger |
| `POST /api/v1/chat` | JSON API |

### Example API call

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is Article 21?\"}"
```

Response shape:

```json
{
  "answer": "...",
  "sources": [
    {
      "article": "21",
      "article_title": "Protection of life and personal liberty",
      "part": "PART III",
      "page_start": 42,
      "page_end": 42,
      "excerpt": "..."
    }
  ],
  "retrieved_count": 5
}
```

## LLM configuration

Defaults in `.env.example` target **Ollama**. For Groq (or another OpenAI-compatible host):

```env
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_key_here
LLM_MODEL=openai/gpt-oss-120b
```

## Project layout

```text
constitution_RAG_bot/
├── frontend/                 # Samvidhaan static chat UI
├── app/
│   ├── main.py               # FastAPI + static mount
│   ├── api/v1/               # /health, /chat
│   ├── ingestion/            # PDF → pages → articles
│   ├── retrieval/            # embed, FAISS, BM25, hybrid, boost
│   ├── reranking/            # cross-encoder
│   ├── services/             # chat orchestration + LLM
│   └── schemas/              # request/response models
├── data/
│   ├── raw/                  # PDF (local only)
│   ├── processed/            # cleaned pages + article JSON
│   └── index/                # FAISS + BM25 + metadata
├── tests/
├── scripts/e2e_api_check.py  # live API scenario suite
├── .env.example
└── pyproject.toml
```

## Tests

```bash
uv run pytest
uv run ruff check app tests scripts
```

Optional live E2E (server must be running with a working LLM):

```bash
uv run python scripts/e2e_api_check.py
```

## Notes

- Answers are **not legal advice**.
- Off-topic questions should be refused; the model is instructed to stay within provided Constitution excerpts.
- `.env` and local `docs/` are gitignored and are not published with this repository.
