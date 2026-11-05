class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever, rrf_k=60):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query, top_k=3, candidate_k=10):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be positive.")

        # Retrieve candidates independently
        vector_results = self.vector_retriever.search(
            query, top_k=candidate_k
        )

        bm25_results = self.bm25_retriever.search(
            query, top_k=candidate_k
        )

        fused_scores = {}
        documents = {}

        # Add vector ranking contributions
        for rank, document in enumerate(vector_results, start=1):
            doc_id = document["chunk_id"]

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0)
                + 1 / (self.rrf_k + rank)
            )

            documents[doc_id] = document

        # Add BM25 ranking contributions
        for rank, document in enumerate(bm25_results, start=1):
            doc_id = document["chunk_id"]

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0)
                + 1 / (self.rrf_k + rank)
            )

            documents[doc_id] = document

        # Sort by combined RRF score
        ranked_ids = sorted(
            fused_scores,
            key=lambda doc_id: fused_scores[doc_id],
            reverse=True
        )

        results = []

        for doc_id in ranked_ids[:top_k]:
            result = documents[doc_id].copy()
            result["rrf_score"] = fused_scores[doc_id]
            results.append(result)

        return results