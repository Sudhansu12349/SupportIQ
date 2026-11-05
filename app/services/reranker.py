from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(
            model_name,
            device="cpu"
        )

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3
    ) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not documents:
            return []

        pairs = [
            (query, document["content"])
            for document in documents
        ]

        scores = self.model.predict(pairs)

        reranked_documents = []

        for document, score in zip(documents, scores):
            result = document.copy()
            result["rerank_score"] = float(score)
            reranked_documents.append(result)

        reranked_documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_documents[:top_k]