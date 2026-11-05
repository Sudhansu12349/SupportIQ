
import pytest

from app.services.bm25_retriever import BM25Retriever


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "1",
            "content": "SoundDrum 1 replacement policy covers eligible defective products.",
            "source": "policy.pdf",
            "page_number": 1
        },
        {
            "chunk_id": "2",
            "content": "Warranty coverage is available for twelve months.",
            "source": "policy.pdf",
            "page_number": 2
        },
        {
            "chunk_id": "3",
            "content": "Refund requests are subject to return conditions.",
            "source": "policy.pdf",
            "page_number": 3
        }
    ]


def test_bm25_returns_relevant_chunk(sample_chunks):
    retriever = BM25Retriever(sample_chunks)

    results = retriever.search(
        "SoundDrum replacement policy",
        top_k=2
    )

    assert len(results) > 0
    assert results[0]["chunk_id"] == "1"


def test_bm25_rejects_empty_query(sample_chunks):
    retriever = BM25Retriever(sample_chunks)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.search("")


def test_bm25_rejects_invalid_top_k(sample_chunks):
    retriever = BM25Retriever(sample_chunks)

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.search("warranty", top_k=0)


def test_bm25_rejects_empty_chunks():
    with pytest.raises(ValueError, match="Chunks cannot be empty"):
        BM25Retriever([])
