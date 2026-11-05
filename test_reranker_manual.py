from app.services.reranker import Reranker

reranker = Reranker()

query = "What is the warranty period?"

documents = [
    {
        "chunk_id": "chunk_1",
        "content": "The product comes with a one-year warranty.",
        "source": "policy.pdf",
        "page_number": 1
    },
    {
        "chunk_id": "chunk_2",
        "content": "Customers can request a refund within seven days.",
        "source": "policy.pdf",
        "page_number": 2
    },
    {
        "chunk_id": "chunk_3",
        "content": "Warranty coverage applies to manufacturing defects.",
        "source": "policy.pdf",
        "page_number": 3
    }
]

results = reranker.rerank(query, documents, top_k=2)

for result in results:
    print(result["chunk_id"], result["rerank_score"])
    print(result["content"])