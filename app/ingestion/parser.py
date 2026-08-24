import json
from pathlib import Path

import fitz

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def find_pdf() -> Path:
    pdfs = list(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {RAW_DIR.resolve()}")
    return pdfs[0]


def extract_pages(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append(
            {
                "page_number": page_num + 1,
                "text": text,
            }
        )
    
    doc.close()
    return pages


def save_pages(pages: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)


def main() -> None:
    pdf_path = find_pdf()
    print(f"File: {pdf_path.name}")

    pages = extract_pages(pdf_path)
    print(f"Number of pages extracted: {len(pages)}")
    print("-----page 1 preview-----")
    print(pages[0]["text"][:800])

    out = PROCESSED_DIR / pdf_path.with_suffix(".json").name
    save_pages(pages, out)
    print(f"Pages saved to {out.resolve()}")

if __name__ == "__main__":
    main()