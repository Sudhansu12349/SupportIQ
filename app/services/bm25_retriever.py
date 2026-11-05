
import re
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Retrieve relevant document chunks using BM25 keyword search."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        self.tokenized_corpus = [
            self._tokenize(chunk["content"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Convert text into lowercase word tokens."""

        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the most relevant chunks for a query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            result = self.chunks[int(index)].copy()
            result["bm25_score"] = score

            results.append(result)

        return results
