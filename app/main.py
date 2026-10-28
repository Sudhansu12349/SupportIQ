
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.document_loader import extract_pages
from app.services.chunker import chunk_pages
from app.services.embeddings import EmbeddingService
from app.services.retriever import Retriever
from app.services.answer_generator import AnswerGenerator


app = FastAPI(title="SupportIQ", version="0.1.0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "policies"
    / "Portronics- Warranty, Return & Refund Policy – Portronics.pdf"
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    source: str
    page_number: int
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


@lru_cache(maxsize=1)
def get_rag_services():
    """Load the policy and models once per application process."""
    pages = extract_pages(str(PDF_PATH))
    chunks = chunk_pages(pages, chunk_size=1000, overlap=150)

    if not chunks:
        raise RuntimeError("No text could be extracted from the policy PDF.")

    embedding_service = EmbeddingService()
    chunk_texts = [chunk["content"] for chunk in chunks]
    chunk_embeddings = embedding_service.generate_embeddings(chunk_texts)

    retriever = Retriever(chunks, chunk_embeddings)
    answer_generator = AnswerGenerator()

    return embedding_service, retriever, answer_generator


@app.get("/")
def root():
    return {"message": "Welcome to SupportIQ", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or whitespace.",
        )

    try:
        embedding_service, retriever, answer_generator = get_rag_services()

        query_embedding = embedding_service.generate_embeddings([question])[0]
        results = retriever.search(query_embedding, top_k=3)
        answer = answer_generator.generate_answer(question, results)

        citations = [
            Citation(
                source=result["source"],
                page_number=result["page_number"],
                similarity=result["similarity"],
            )
            for result in results
        ]

        return ChatResponse(answer=answer, citations=citations)

    except HTTPException:
        raise
    except Exception:
        # Keep internal errors and credentials out of the API response.
        raise HTTPException(
            status_code=500,
            detail="SupportIQ could not process the question. Please try again.",
        )
