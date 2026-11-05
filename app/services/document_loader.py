
from pathlib import Path
import pymupdf
from langchain_core.documents import Document


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract PDF pages and create LangChain Document objects."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if text:
                document = Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_number - 1
                    }
                )

                pages.append({
                    "page_number": document.metadata["page"] + 1,
                    "text": document.page_content,
                    "source": document.metadata["source"]
                })

    return pages
