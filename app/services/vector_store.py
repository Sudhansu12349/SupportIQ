
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStore:
    """Manage persistent document embeddings using ChromaDB."""

    def __init__(
        self,
        embedding_model,
        persist_directory: str = "chroma_db",
        collection_name: str = "supportiq_policies"
    ):
        self.persist_directory = str(Path(persist_directory))

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=self.persist_directory
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Convert chunks into LangChain Documents and store them."""

        if not chunks:
            return

        documents = []
        ids = []

        for chunk in chunks:
            document = Document(
                page_content=chunk["content"],
                metadata={
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"]
                }
            )

            documents.append(document)
            ids.append(chunk["chunk_id"])

        self.vector_store.add_documents(
            documents=documents,
            ids=ids
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search for relevant chunks using semantic similarity."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        results = self.vector_store.similarity_search_with_score(
            query,
            k=top_k
        )

        chunks = []

        for document, score in results:
            chunks.append({
                "content": document.page_content,
                "source": document.metadata["source"],
                "page_number": document.metadata["page_number"],
                "chunk_id": document.metadata["chunk_id"],
                "similarity_distance": float(score)
            })

        return chunks
