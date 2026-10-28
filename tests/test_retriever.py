
import pytest

from app.services.retriever import Retriever


def test_search_ranks_most_similar_chunk_first():
    chunks = [
        {
            "content": "The product warranty covers manufacturing defects.",
            "source": "policy.pdf",
            "page_number": 1,
        },
        {
            "content": "Refunds are available for eligible returns.",
            "source": "policy.pdf",
            "page_number": 2,
        },
    ]

    # Simple normalized vectors for testing retrieval logic.
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    retriever = Retriever(chunks, embeddings)

    # This query is identical to the first chunk's vector.
    results = retriever.search([1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0]["content"] == chunks[0]["content"]
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[0]["page_number"] == 1


def test_search_rejects_mismatched_embedding_count():
    chunks = [{"content": "Example", "source": "policy.pdf", "page_number": 1}]
    embeddings = []

    with pytest.raises(ValueError, match="one embedding"):
        Retriever(chunks, embeddings)


def test_search_rejects_wrong_query_dimensions():
    chunks = [{"content": "Example", "source": "policy.pdf", "page_number": 1}]
    embeddings = [[1.0, 0.0]]

    retriever = Retriever(chunks, embeddings)

    with pytest.raises(ValueError, match="dimensions"):
        retriever.search([1.0, 0.0, 0.0])
