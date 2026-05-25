from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.config import Settings
from app.database import get_session
from app.dependencies import get_app_settings, get_chroma_store, get_embedding_provider, get_llm_router
from app.models import Document
from app.providers.embeddings import EmbeddingProvider, build_embedding_profile
from app.providers.embeddings import EmbeddingProfile
from app.providers.llm import AllLLMProvidersFailed, LLMRouter
from app.schemas import ChatRequest, ChatResponse, Source
from app.services.chroma_store import ChromaStore

router = APIRouter(prefix="/chat", tags=["chat"])

PROMPT_TEMPLATE = """Kamu adalah asisten AI. Jawab pertanyaan pengguna dalam Bahasa Indonesia menggunakan konteks dokumen berbahasa Arab berikut.

Konteks:
{context}

Pertanyaan:
{query}
"""


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    chroma_store: ChromaStore = Depends(get_chroma_store),
    llm_router: LLMRouter = Depends(get_llm_router),
) -> ChatResponse:
    current_profile = build_embedding_profile(settings)
    _raise_if_stale_embeddings(session, request.book_filter, current_profile)

    query_embedding = embedding_provider.embed_query(request.query)
    where = _build_chroma_filter(
        book_id=request.book_filter,
        embedding_fingerprint=current_profile.fingerprint,
    )
    results = chroma_store.similarity_search(
        query_embedding=query_embedding,
        top_k=settings.retrieval_top_k,
        where=where,
    )

    sources = [
        Source(
            id=result.id,
            document=result.document,
            metadata=result.metadata,
            distance=result.distance,
        )
        for result in results
    ]
    if not results:
        return ChatResponse(
            answer="Saya tidak menemukan konteks dokumen yang relevan untuk menjawab pertanyaan tersebut.",
            provider_used="none",
            sources=[],
        )

    context = "\n\n---\n\n".join(result.document for result in results)
    prompt = PROMPT_TEMPLATE.format(context=context, query=request.query)
    try:
        generation = llm_router.generate(prompt)
    except AllLLMProvidersFailed as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Semua provider LLM gagal menghasilkan jawaban.",
                "failures": [failure.__dict__ for failure in exc.failures],
            },
        ) from exc

    return ChatResponse(
        answer=generation.answer,
        provider_used=generation.provider_used,
        sources=sources,
    )


def _raise_if_stale_embeddings(
    session: Session,
    book_filter: Optional[str],
    current_profile: EmbeddingProfile,
) -> None:
    if book_filter:
        document = session.get(Document, book_filter)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {book_filter} not found",
            )
        if document.embedding_fingerprint != current_profile.fingerprint:
            raise _stale_embedding_error([document], current_profile)
        return

    stale_documents = session.exec(
        select(Document).where(Document.embedding_fingerprint != current_profile.fingerprint)
    ).all()
    if stale_documents:
        raise _stale_embedding_error(stale_documents, current_profile)


def _stale_embedding_error(documents: list[Document], current_profile: EmbeddingProfile) -> HTTPException:
    first = documents[0]
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "message": (
                "Embedding dokumen lama tidak cocok dengan konfigurasi embedding saat ini. "
                "Hapus dokumen tersebut lalu import ulang EPUB agar diproses dengan model embedding baru."
            ),
            "affected_count": len(documents),
            "book_id": first.book_id,
            "stored_embedding": {
                "provider": first.embedding_provider,
                "model": first.embedding_model,
                "dimension": first.embedding_dimension,
                "fingerprint": first.embedding_fingerprint,
            },
            "current_embedding": {
                "provider": current_profile.provider,
                "model": current_profile.model,
                "dimension": current_profile.dimension,
                "fingerprint": current_profile.fingerprint,
            },
        },
    )


def _build_chroma_filter(book_id: Optional[str], embedding_fingerprint: str) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{"embedding_fingerprint": embedding_fingerprint}]
    if book_id:
        filters.append({"book_id": book_id})
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}
