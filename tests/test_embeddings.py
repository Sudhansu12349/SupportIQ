
from app.services.document_loader import extract_pages
from app.services.chunker import chunk_pages
from app.services.embeddings import EmbeddingService


pdf_path = (
    r"C:\Users\HP\SupportIQ\app\data\policies"
    r"\Portronics- Warranty, Return & Refund Policy – Portronics.pdf"
)

# 1. Extract PDF pages and create chunks
pages = extract_pages(pdf_path)
chunks = chunk_pages(pages, chunk_size=1000, overlap=150)

# 2. Generate embeddings for the chunk contents
embedding_service = EmbeddingService()
texts = [chunk["content"] for chunk in chunks]
vectors = embedding_service.generate_embeddings(texts)

# 3. Verify the output
print("Chunks:", len(chunks))
print("Embeddings:", len(vectors))

if vectors:
    print("Vector dimensions:", len(vectors[0]))
    print("First chunk's source:", chunks[0]["source"])
    print("First chunk's page:", chunks[0]["page_number"])
    print("First 5 vector values:", vectors[0][:5])

    assert len(vectors) == len(chunks)
    assert all(len(vector) == 384 for vector in vectors)
    print("Embedding test passed!")
