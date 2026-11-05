
from functools import lru_cache
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.document_loader import extract_pages
from app.services.chunker import chunk_pages
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever
from app.services.answer_generator import AnswerGenerator
from app.services.bm25_retriever import BM25Retriever
from app.services.hybrid_retriever import HybridRetriever
from app.services.reranker import Reranker


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FastAPI application
app = FastAPI(
    title="SupportIQ",
    version="0.2.0"
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "policies"
    / "Portronics- Warranty, Return & Refund Policy – Portronics.pdf"
)

CHROMA_PATH = PROJECT_ROOT / "chroma_db"


# Request and response models
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    source: str
    page_number: int
    rerank_score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


# Initialize RAG services once
@lru_cache(maxsize=1)
def get_rag_services():

    logger.info("Initializing SupportIQ RAG services")

    # 1. Load PDF pages
    pages = extract_pages(str(PDF_PATH))
    logger.info("Loaded %d PDF pages", len(pages))

    # 2. Split pages into chunks
    chunks = chunk_pages(pages)
    logger.info("Created %d document chunks", len(chunks))

    # 3. Initialize embedding model
    embedding_service = EmbeddingService()

    # 4. Initialize ChromaDB
    vector_store = VectorStore(
        embedding_model=embedding_service.model,
        persist_directory=str(CHROMA_PATH)
    )

    vector_store.add_chunks(chunks)

    # 5. Initialize semantic retriever
    vector_retriever = Retriever(vector_store)

    # 6. Initialize BM25 retriever
    bm25_retriever = BM25Retriever(chunks)

    # 7. Combine both retrievers using RRF
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever
    )

    # 8. Initialize Cross-Encoder reranker
    reranker = Reranker()

    # 9. Initialize answer generator
    answer_generator = AnswerGenerator()

    logger.info("SupportIQ RAG services initialized successfully")

    return hybrid_retriever, reranker, answer_generator


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to SupportIQ",
        "status": "running"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or whitespace."
        )

    try:
        hybrid_retriever, reranker, answer_generator = get_rag_services()

        # 1. Hybrid retrieval using Vector Search + BM25 + RRF
        candidates = hybrid_retriever.search(
            query=question,
            top_k=10,
            candidate_k=10
        )

        # 2. Rerank retrieved candidates
        results = reranker.rerank(
            query=question,
            documents=candidates,
            top_k=3
        )

        # 3. Generate grounded answer
        answer = answer_generator.generate_answer(
            question,
            results
        )

        # 4. Prepare citations
        citations = [
            Citation(
                source=result["source"],
                page_number=result["page_number"],
                rerank_score=result["rerank_score"]
            )
            for result in results
        ]

        return ChatResponse(
            answer=answer,
            citations=citations
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Error while processing chat request")

        raise HTTPException(
            status_code=500,
            detail="SupportIQ could not process the question. Please try again."
        )
