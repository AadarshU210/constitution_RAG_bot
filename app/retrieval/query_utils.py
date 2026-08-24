"""Query helpers: article refs + retrieval-friendly query shaping."""

from __future__ import annotations

import re
from collections import Counter

ARTICLE_REF_RE = re.compile(
    r"(?i)\b(?:articles?|arts?\.?)\s*(\d+[A-Za-z]*(?:-[A-Za-z])?)\b"
)

RETRIEVAL_QUERY_MAX = 480


def extract_article_numbers(query: str) -> list[str]:
    """Return unique Article numbers mentioned in query (preserve order)."""
    seen: set[str] = set()
    out: list[str] = []
    for match in ARTICLE_REF_RE.finditer(query):
        raw = match.group(1)
        m = re.match(r"^(\d+)([A-Za-z]*(?:-[A-Za-z])?)$", raw)
        if not m:
            continue
        key = f"{m.group(1)}{m.group(2).upper()}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _strip_filler(text: str) -> str:
    """Collapse repeated padding tokens that drown hybrid search."""
    tokens = text.split()
    if len(tokens) < 12:
        return text
    counts = Counter(t.lower() for t in tokens)
    filler = {tok for tok, n in counts.items() if n >= 8 and len(tok) <= 4}
    if not filler:
        return text
    kept = [t for t in tokens if t.lower() not in filler]
    return " ".join(kept) if kept else text


def build_retrieval_query(question: str) -> str:
    """Compact query for search; keep full question for the LLM separately."""
    cleaned = " ".join(question.split())
    refs = extract_article_numbers(cleaned)
    ref_prefix = " ".join(f"Article {r}" for r in refs)
    cleaned = _strip_filler(cleaned)

    if refs and len(cleaned) > 300:
        rest = ARTICLE_REF_RE.sub(" ", cleaned)
        rest = " ".join(rest.split())[:180]
        return f"{ref_prefix} {rest}".strip()

    if len(cleaned) <= RETRIEVAL_QUERY_MAX:
        return cleaned

    head = cleaned[:200]
    tail = cleaned[-200:]
    parts = [p for p in (ref_prefix, head, tail) if p]
    return " ".join(parts)
