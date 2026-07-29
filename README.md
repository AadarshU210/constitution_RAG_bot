# Constitution RAG Bot

A RAG-based chatbot for the Constitution of India.

## Setup

```bash
cd E:\projects\AI-ML\constitution_RAG_bot
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the API docs.

## Test

```bash
uv run pytest
```
