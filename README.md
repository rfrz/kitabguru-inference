# KitabGuru Inference Engine

FastAPI backend for a cross-lingual RAG workflow: Arabic EPUB content is embedded into ChromaDB, while Indonesian queries are answered through a fallback LLM router.

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API docs.

## Main Endpoints

- `POST /api/documents/import` imports an EPUB, chunks Arabic text, embeds it, and stores document metadata.
- `GET /api/documents` lists imported books and whether their embeddings match the current `.env` embedding config.
- `DELETE /api/documents/{book_id}` removes SQLite metadata and Chroma vectors for a book.
- `POST /api/chat` retrieves matching Arabic chunks and generates an Indonesian answer.

## Embedding Provenance

Every book and vector chunk stores:

- embedding provider
- embedding model
- embedding dimension
- embedding fingerprint

If `.env` changes to a different embedding provider/model/dimension, `POST /api/chat` returns `409 Conflict` for stale documents. Delete and re-import the EPUB to re-embed it with the current model.

Set `EMBEDDING_DIMENSION` when using a model whose output dimension is not listed in the built-in defaults. This keeps the fingerprint strict and predictable.
