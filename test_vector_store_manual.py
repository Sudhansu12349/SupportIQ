
from app.services.document_loader import extract_pages
from app.services.chunker import chunk_pages
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


PDF_PATH = (
    r"app\data\policies\Portronics- Warranty, Return & Refund Policy – Portronics.pdf"
)


# 1. Load PDF
pages = extract_pages(PDF_PATH)
print("Pages loaded:", len(pages))


# 2. Create chunks
chunks = chunk_pages(pages)
print("Chunks created:", len(chunks))


# 3. Initialize embedding model
embedding_service = EmbeddingService()


# 4. Initialize ChromaDB
vector_store = VectorStore(
    embedding_model=embedding_service.model
)


# 5. Store chunks
vector_store.add_chunks(chunks)
print("Documents stored in ChromaDB!")


# 6. Search
query = "What is the warranty period?"

results = vector_store.search(query, top_k=3)

print("\nRetrieved results:")

for result in results:
    print("\nContent:", result["content"][:200])
    print("Source:", result["source"])
    print("Page:", result["page_number"])
    print("Distance:", result["similarity_distance"])
