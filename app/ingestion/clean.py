"""Light cleaning for Constitution page text."""


from __future__ import annotations

import re

HEADER_RE = re.compile(r"^\s*THE CONSTITUTION OF INDIA\s*$", re.IGNORECASE)
PART_FOOTER_RE = re.compile(r"^\s*\(Part\b.*\)\s*$", re.IGNORECASE)
PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$")


def clean_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r","\n")

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()

        if HEADER_RE.match(stripped):
            continue
        if PART_FOOTER_RE.match(stripped):
            continue
        if PAGE_NUM_RE.match(stripped):
            continue

        cleaned_lines.append(line.rstrip())

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
