
import numpy as np


class Retriever:
    """Find document chunks most relevant to a question."""

    def __init__(self, chunks: list[dict], embeddings: list[list[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding")

        self.chunks = chunks
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not self.chunks:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim != 1 or query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query embedding has the wrong dimensions")

        # Embeddings are normalized, so dot product = cosine similarity.
        scores = self.embeddings @ query

        # Sort scores from highest to lowest.
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for index in ranked_indices:
            result = self.chunks[int(index)].copy()
            result["similarity"] = float(scores[index])
            results.append(result)

        return results
