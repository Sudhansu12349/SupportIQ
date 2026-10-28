
from app.services.document_loader import extract_pages
from app.services.chunker import chunk_pages
from app.services.embeddings import EmbeddingService
from app.services.retriever import Retriever


PDF_PATH = (
    r"C:\Users\HP\SupportIQ\app\data\policies"
    r"\Portronics- Warranty, Return & Refund Policy – Portronics.pdf"
)


def main():
    question = input("Ask a question about the policy: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\nLoading policy...")

    pages = extract_pages(PDF_PATH)
    chunks = chunk_pages(pages, chunk_size=1000, overlap=150)

    if not chunks:
        print("No text chunks were extracted from the PDF.")
        return

    print(f"Extracted {len(pages)} pages and created {len(chunks)} chunks.")
    print("Generating embeddings...")

    embedding_service = EmbeddingService()

    # Generate an embedding for every policy chunk.
    chunk_texts = [chunk["content"] for chunk in chunks]
    chunk_embeddings = embedding_service.generate_embeddings(chunk_texts)

    # Generate an embedding for the user's question.
    query_embedding = embedding_service.generate_embeddings([question])[0]

    # Find the most relevant policy chunks.
    retriever = Retriever(chunks, chunk_embeddings)
    results = retriever.search(query_embedding, top_k=3)

    print("\nTop matching policy passages:")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page_number']}")
        print(f"Content:\n{result['content']}")


if __name__ == "__main__":
    main()
