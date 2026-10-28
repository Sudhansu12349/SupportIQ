
from app.services.document_loader import extract_pages
from app.services.chunker import chunk_pages

pdf_path = r"C:\Users\HP\SupportIQ\app\data\policies\Portronics- Warranty, Return & Refund Policy – Portronics.pdf"

pages = extract_pages(pdf_path)
chunks = chunk_pages(pages, chunk_size=1000, overlap=150)

print("Pages extracted:", len(pages))
print("Chunks created:", len(chunks))

for chunk in chunks[:3]:
    print("\n--- Chunk ---")
    print("Source:", chunk["source"])
    print("Page:", chunk["page_number"])
    print("Content:", chunk["content"][:250])

