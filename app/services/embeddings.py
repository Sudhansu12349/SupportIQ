
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate vector embeddings for SupportIQ document chunks."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of text strings into numerical vectors."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()
