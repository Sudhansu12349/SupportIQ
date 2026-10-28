
from app.services.document_loader import extract_pages


def test_extract_pages_from_pdf():
    pdf_path = (
        r"C:\Users\HP\SupportIQ\app\data\policies"
        r"\Portronics- Warranty, Return & Refund Policy – Portronics.pdf"
    )

    pages = extract_pages(pdf_path)

    assert len(pages) > 0
    assert "text" in pages[0]
    assert "page_number" in pages[0]
    assert "source" in pages[0]
    assert pages[0]["page_number"] >= 1
    assert pages[0]["text"].strip()
