# SupportIQ — RAG-Powered Customer Support

SupportIQ is a customer-support question-answering system that uses Retrieval-Augmented Generation (RAG) to answer questions from policy documents. It retrieves relevant passages from a policy PDF and supplies them to a language model to generate evidence-based responses.

## Features

- **PDF ingestion:** Extracts policy text while preserving page numbers and source metadata.
- **Text chunking:** Splits documents into overlapping chunks for more targeted retrieval.
- **Semantic search:** Uses Sentence Transformers to generate vector embeddings and rank relevant passages by similarity.
- **Grounded answer generation:** Uses an LLM to answer questions using retrieved policy evidence, with an insufficient-evidence fallback.
- **Source references:** Returns the source document, page number, and similarity score for retrieved passages.
- **REST API:** Exposes the RAG pipeline through FastAPI, with interactive API documentation.
- **Testing:** Uses pytest to validate core components.

## Tech Stack

Python · FastAPI · Sentence Transformers · NumPy · PyMuPDF · OpenAI API · Pydantic · pytest

## How It Works

1. Extract text and page metadata from a policy PDF.
2. Split the extracted text into overlapping chunks.
3. Generate vector embeddings for the chunks.
4. Embed the customer's question and retrieve the most similar chunks.
5. Pass the question and retrieved evidence to the language model.
6. Return the generated answer and retrieval references through the API.

## Project Structure

- `app/main.py` — FastAPI application and chat endpoint
- `app/services/document_loader.py` — PDF text extraction
- `app/services/chunker.py` — Text chunking and metadata preservation
- `app/services/embeddings.py` — Embedding generation
- `app/services/retriever.py` — Similarity-based retrieval
- `app/services/answer_generator.py` — LLM answer generation
- `tests/` — Automated tests
- `search_policy.py` — Script for experimenting with retrieval

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Create a `.env` file based on `.env.example`.
5. Set your own `OPENAI_API_KEY` and, if needed, `OPENAI_MODEL`.
6. Place the policy PDF at the expected path under `app/data/policies/`.

Run the API from the project root:

`uvicorn app.main:app --reload`

Open `http://127.0.0.1:8000/docs` to interact with the API.

Run tests:

`python -m pytest tests -v`

## API Usage

Endpoint: `POST /chat`

Example request:

```json
{
  "question": "What does the warranty cover?"
}
```

The response contains an answer and a list of retrieved source references.

## Current Limitations

- Retrieval uses in-memory embeddings and does not yet use a persistent vector database.
- Only the configured policy document is indexed by the current MVP.
- Retrieval similarity is not a guarantee that every generated claim is supported by the cited evidence.
- Authentication, ticket management, production deployment, and comprehensive RAG evaluation are planned improvements.

## Roadmap

- Hybrid keyword and semantic retrieval
- Reranking and retrieval evaluation
- PostgreSQL-based storage
- Authentication and ticket management
- Redis caching and background jobs
- Automated evaluation, observability, and deployment
