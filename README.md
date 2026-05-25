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

## RAG Accuracy Controls

The chat pipeline retrieves broad candidates, merges query variants with reciprocal rank fusion, expands neighboring chunks, and runs a completeness scan for numbered list questions. Re-import EPUB files after changing chunking or embedding settings so new metadata such as headings, neighbor ids, and text hashes is available.

Useful knobs:

- `RETRIEVAL_CANDIDATE_K`: broad first-pass retrieval count.
- `RETRIEVAL_FINAL_K`: final source count sent to the LLM.
- `RETRIEVAL_NEIGHBOR_WINDOW`: adjacent chunk expansion depth.
- `RAG_ENABLE_COMPLETENESS_SCAN`: include numbered headings from the same book for complete-list questions.
- `LLM_TEMPERATURE`: generation temperature; default is `0` for more deterministic grounded answers.

## Embedding Provenance

Every book and vector chunk stores:

- embedding provider
- embedding model
- embedding dimension
- embedding fingerprint

If `.env` changes to a different embedding provider/model/dimension, `POST /api/chat` returns `409 Conflict` for stale documents. Delete and re-import the EPUB to re-embed it with the current model.

Set `EMBEDDING_DIMENSION` when using a model whose output dimension is not listed in the built-in defaults. This keeps the fingerprint strict and predictable.
