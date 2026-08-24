"""Structure-aware split: pages -> article chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.ingestion.clean import clean_page_text

PROCESSED_DIR = Path("data/processed")
PAGES_JSON = PROCESSED_DIR / "constitution_of_india.json"
OUT_JSON = PROCESSED_DIR / "constitution_articles.json"

PAGE_MARKER_RE = re.compile(r"<<<PAGE\s+(\d+)>>>")

# PART III / PART IVA / PART IXA etc.
PART_RE = re.compile(r"(?m)^PART\s+([IVXLC]+[A-Z]?)\s*$")

# Body articles usually look like:
# 21. Protection of life and personal liberty.—No person...
# 2[21A. Right to education.—The State...
# 368. 1[Power of Parliament ... therefor].— ...
# Em/en dash only (not ASCII hyphen) to avoid "Forty-second" false positives.
ARTICLE_RE = re.compile(
    r"(?m)"
    r"(?:^\d+\s*\[|^\[)?"  # optional footnote bracket at line start
    r"(\d+[A-Z]*(?:-[A-Z])?)\.\s+"  # article number
    r"(?:\d+\s*\[)?"  # optional footnote before title (e.g. 1[Power...)
    r"("
    r"[A-Z][^\n—–\-\]]{2,}?"  # first title line
    r"(?:\n[^\n—–\-\]]{2,}?){0,4}"  # wrapped titles (e.g. Art 369)
    r")"
    r"\]?"  # closing bracket if title was footnoted
    r"\.?"  # occasional period after bracketed title (Art 368)
    r"\s*[—–]\s*"
)

FOOTNOTE_TITLE_RE = re.compile(
    r"^(Subs\.|Ins\.|Added|Omitted|Rep\.|Omit|See |Cl\.|Sub-clause|Explanation)",
    re.IGNORECASE,
)


def load_pages(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_corpus(pages: list[dict]) -> str:
    """Join cleaned pages with page markers for later page_start/page_end."""
    parts: list[str] = []
    for page in pages:
        page_no = page["page_number"]
        text = clean_page_text(page["text"])
        parts.append(f"<<<PAGE {page_no}>>>\n{text}")
    return "\n".join(parts)


def page_at(corpus: str, pos: int) -> int:
    markers = list(PAGE_MARKER_RE.finditer(corpus[: max(pos, 0)]))
    if not markers:
        return 1
    return int(markers[-1].group(1))


def strip_markers(text: str) -> str:
    text = PAGE_MARKER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_footnote_match(article_title: str, span_text: str) -> bool:
    title = " ".join(article_title.split())
    if FOOTNOTE_TITLE_RE.match(title):
        return True
    # real articles are usually longer than a one-line footnote
    if len(span_text.strip()) < 80 and ("Subs." in span_text or "Ins." in span_text):
        return True
    return False


def dedupe_articles(articles: list[dict]) -> list[dict]:
    """Keep one chunk per article number.

    Prefer the earliest occurrence (lower page_start). Later Schedules reuse
    numbers like "21." for paragraphs, which are longer but not the Article.
    """
    best: dict[str, dict] = {}
    kept: list[dict] = []

    for a in articles:
        if a["chunk_type"] == "preamble":
            kept.append(a)
            continue

        key = a["article"]
        prev = best.get(key)
        if prev is None:
            best[key] = a
            continue

        prev_page = prev.get("page_start") or 10**9
        curr_page = a.get("page_start") or 10**9
        if curr_page < prev_page:
            best[key] = a
        elif curr_page == prev_page and a.get("part") and not prev.get("part"):
            best[key] = a

    body = list(best.values())
    body.sort(key=lambda x: (x.get("page_start") or 0, str(x.get("article") or "")))
    return kept + body


def split_articles(pages: list[dict]) -> list[dict]:
    corpus = build_corpus(pages)

    preamble_idx = corpus.find("WE, THE PEOPLE OF INDIA")
    if preamble_idx != -1:
        start_at = max(0, corpus.rfind("PREAMBLE", 0, preamble_idx))
        corpus = corpus[start_at:]

    article_matches = list(ARTICLE_RE.finditer(corpus))
    if not article_matches:
        return []

    part_matches = list(PART_RE.finditer(corpus))

    def part_at(pos: int) -> tuple[str | None, str | None]:
        current = None
        title = None
        for m in part_matches:
            if m.start() > pos:
                break
            current = f"PART {m.group(1)}"
            after = corpus[m.end() : m.end() + 200]
            for line in after.splitlines():
                line = line.strip()
                if not line or line.startswith("<<<PAGE"):
                    continue
                if line.upper().startswith("PART "):
                    continue
                title = line
                break
        return current, title

    articles: list[dict] = []

    first_article_start = article_matches[0].start()
    preamble_text = corpus[:first_article_start].strip()

    if "WE, THE PEOPLE OF INDIA" in preamble_text:
        articles.append(
            {
                "chunk_id": "preamble",
                "part": None,
                "part_title": None,
                "article": None,
                "article_title": "Preamble",
                "text": strip_markers(preamble_text),
                "page_start": page_at(corpus, 0),
                "page_end": page_at(corpus, first_article_start - 1),
                "chunk_type": "preamble",
            }
        )

    for i, match in enumerate(article_matches):
        start = match.start()
        end = (
            article_matches[i + 1].start()
            if i + 1 < len(article_matches)
            else len(corpus)
        )
        span = corpus[start:end]
        article_no = match.group(1)
        article_title = " ".join(match.group(2).split())

        if is_footnote_match(article_title, span):
            continue

        part, part_title = part_at(start)
        page_start = page_at(corpus, start)
        page_end = page_at(corpus, end - 1)

        articles.append(
            {
                "chunk_id": f"article_{article_no}",
                "part": part,
                "part_title": part_title,
                "article": article_no,
                "article_title": article_title,
                "text": strip_markers(span),
                "page_start": page_start,
                "page_end": page_end,
                "chunk_type": "article",
            }
        )

    return dedupe_articles(articles)


def main() -> None:
    if not PAGES_JSON.exists():
        raise FileNotFoundError(
            f"Missing {PAGES_JSON}. Run: uv run python -m app.ingestion.parser"
        )

    pages = load_pages(PAGES_JSON)
    articles = split_articles(pages)
    print(f"Units found: {len(articles)}")

    unique = {a["article"] for a in articles if a.get("article")}
    print(f"Unique articles: {len(unique)}")

    wanted = {"14", "19", "21", "21A", "32", "25", "12", "368", "369"}
    found = wanted & unique
    print("Found key articles:", sorted(found))
    missing = wanted - found
    if missing:
        print("Missing:", sorted(missing))

    short = [a for a in articles if a.get("article") and len(a["text"]) < 100]
    print(f"Short chunks (<100 chars): {len(short)}")

    for a in articles:
        if a.get("article") == "21":
            print("\n--- Article 21 sample ---")
            print("part:", a["part"], "|", a["part_title"])
            print("pages:", a["page_start"], "-", a["page_end"])
            print(a["text"][:400])
            break

    for a in articles:
        if a.get("article") == "25":
            print("\n--- Article 25 title ---")
            print(a["article_title"])
            print("pages:", a["page_start"], "-", a["page_end"])
            break

    save_json(articles, OUT_JSON)
    print(f"\nWrote {OUT_JSON.resolve()}")


if __name__ == "__main__":
    main()
