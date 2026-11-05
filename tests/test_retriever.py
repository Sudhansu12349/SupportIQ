
from unittest.mock import Mock

import pytest

from app.services.retriever import Retriever


def test_search_returns_relevant_chunks():
    mock_vector_store = Mock()

    expected_results = [
        {
            "content": "Warranty is valid for 12 months.",
            "source": "policy.pdf",
            "page_number": 1,
            "chunk_id": "chunk_1",
            "similarity_distance": 0.15
        }
    ]

    mock_vector_store.search.return_value = expected_results

    retriever = Retriever(mock_vector_store)

    results = retriever.search("What is the warranty period?", top_k=3)

    assert results == expected_results

    mock_vector_store.search.assert_called_once_with(
        query="What is the warranty period?",
        top_k=3
    )


def test_search_uses_default_top_k():
    mock_vector_store = Mock()
    mock_vector_store.search.return_value = []

    retriever = Retriever(mock_vector_store)

    results = retriever.search("Warranty information")

    assert results == []

    mock_vector_store.search.assert_called_once_with(
        query="Warranty information",
        top_k=3
    )
