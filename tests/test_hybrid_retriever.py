
from unittest.mock import Mock

import pytest

from app.services.hybrid_retriever import HybridRetriever


@pytest.fixture
def retriever():
    vector = Mock()
    bm25 = Mock()

    vector.search.return_value = [
        {"chunk_id": "A", "content": "Chunk A"},
        {"chunk_id": "B", "content": "Chunk B"}
    ]

    bm25.search.return_value = [
        {"chunk_id": "B", "content": "Chunk B"},
        {"chunk_id": "C", "content": "Chunk C"}
    ]

    return HybridRetriever(vector, bm25)


def test_hybrid_search_combines_rankings(retriever):
    results = retriever.search("replacement policy", top_k=3)

    assert len(results) == 3

    # B appears in both rankings, so it should rank first.
    assert results[0]["chunk_id"] == "B"

    assert "rrf_score" in results[0]


def test_hybrid_search_uses_both_retrievers(retriever):
    retriever.search("warranty", top_k=2)

    retriever.vector_retriever.search.assert_called_once()
    retriever.bm25_retriever.search.assert_called_once()


def test_hybrid_search_rejects_empty_query(retriever):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.search("")


def test_hybrid_search_rejects_invalid_top_k(retriever):
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("warranty", top_k=0)


def test_hybrid_search_rejects_invalid_rrf_k():
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(Mock(), Mock(), rrf_k=0)
