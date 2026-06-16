---
title: KitabGuru Inference Engine
emoji: 📚
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# KitabGuru Inference Engine

FastAPI backend for a cross-lingual RAG workflow: Arabic EPUB content is embedded into Qdrant, while Indonesian queries are answered through a fallback LLM router.

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API docs.

## Streamlit Playground

Run the FastAPI backend first:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

In another terminal, run the temporary Streamlit playground:

```powershell
.\.venv\Scripts\streamlit.exe run playground_streamlit.py --server.port 8501
```

Open `http://127.0.0.1:8501` to upload EPUB files, manage imported documents, and test chat inference against the running backend. The playground uses `http://127.0.0.1:8000` as the default API base URL.

## Main Endpoints

- `GET /health`
  - Healthcheck endpoint to verify if the server is running.
  - **Response**: `{"status": "ok"}`

- `POST /api/documents/import`
  - Imports an EPUB, chunks Arabic text, embeds it, and stores document metadata.
  - **Request (Multipart Form Data)**:
    - `file`: The EPUB file to upload (required).
    - `title`: Title of the book (optional).
    - `author`: Author of the book (optional).
  - **Response**: JSON containing `book_id`, `title`, `author`, `total_chunks`, and `embedding` state.

- `GET /api/documents`
  - Lists imported books and whether their embeddings match the current `.env` embedding config.
  - **Response**: JSON Array of document metadata.

- `DELETE /api/documents/{book_id}`
  - Removes SQLite metadata and Qdrant vectors for a book.
  - **Response**: HTTP 204 No Content.

- `POST /api/chat`
  - Retrieves matching Arabic chunks and generates an Indonesian answer.
  - **Request (JSON)**:
    - `query` (string): The question to ask in Indonesian.
    - `book_filter` (string, optional): A specific `book_id` to restrict the search.
  - **Response**: JSON containing the generated `answer`, `provider_used`, list of `sources` (chunks), `answer_status`, and `citations`.

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

## Deployment to Hugging Face Spaces

This application is pre-configured to run on Hugging Face Spaces using the Docker SDK, as indicated by the YAML frontmatter at the top of this file (`sdk: docker`, `app_port: 7860`).

To deploy:
1. Create a new Space on Hugging Face and select **Docker** as the Blank template.
2. Push this repository directory to the Space.
3. Go to the Space's settings (Settings > Variables and secrets).
4. Add all required environment variables (e.g., `QDRANT_URL`, `QDRANT_API_KEY`, and LLM API Keys like `OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`) as **Secrets**.
5. Hugging Face will automatically build the Docker image using the `Dockerfile` in this directory and expose the app on port `7860`.
