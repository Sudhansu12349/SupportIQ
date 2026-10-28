
from pathlib import Path
import pymupdf


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text from a PDF while preserving page numbers."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "source": path.name,
                })

    return pages
