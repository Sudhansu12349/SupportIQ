
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    """Generate embeddings using LangChain and HuggingFace."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for document chunks."""
        if not texts:
            return []

        return self.model.embed_documents(texts)

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a user query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.model.embed_query(query)
