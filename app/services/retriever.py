
from app.services.vector_store import VectorStore


class Retriever:
    """Retrieve relevant policy chunks using ChromaDB."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search for relevant policy chunks."""

        return self.vector_store.search(
            query=query,
            top_k=top_k
        )
