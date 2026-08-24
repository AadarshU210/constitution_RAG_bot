"""LLM answer generation over retrieved Constitution chunks."""

from __future__ import annotations

from openai import AsyncOpenAI

from app.core.config import settings

SYSTEM_PROMPT = """You are a careful assistant for the Constitution of India.
Answer ONLY using the provided context excerpts.
Rules:
- If the context is insufficient, say you could not find it in the provided Constitution text.
- Cite Article numbers (and Part when available) in your answer.
- Do not invent articles, cases, or legal advice beyond the given text.
- Prefer quoting short exact phrases when stating legal wording.
- Keep answers clear and concise.
"""


def _format_context(chunks: list[dict]) -> str:
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        article = chunk.get("article")
        title = chunk.get("article_title") or ""
        part = chunk.get("part") or "N/A"
        pages = f"{chunk.get('page_start')}-{chunk.get('page_end')}"
        header = f"[{i}] "
        if article:
            header += f"Article {article}: {title}"
        else:
            header += title or chunk.get("chunk_id") or "Excerpt"
        header += f" | Part: {part} | Pages: {pages}"
        blocks.append(f"{header}\n{chunk.get('text') or ''}".strip())
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict]) -> list[dict[str, str]]:
    context = _format_context(chunks)
    user = (
        f"Context from the Constitution of India:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )


async def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return (
            "I could not find relevant provisions in the indexed Constitution text "
            "for this question."
        )

    client = get_llm_client()
    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=build_messages(question, chunks),
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
    content = response.choices[0].message.content
    return (content or "").strip()
