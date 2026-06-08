from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.config import Settings
from app.database import get_session
from app.dependencies import get_app_settings, get_qdrant_store, get_embedding_provider, get_llm_router
from app.models import Document
from app.providers.embeddings import EmbeddingProvider, build_embedding_profile
from app.providers.embeddings import EmbeddingProfile
from app.providers.llm import AllLLMProvidersFailed, LLMRouter
from app.schemas import ChatRequest, ChatResponse, Source
from app.services.qdrant_store import QdrantStore
from app.services.retrieval import retrieve_context

router = APIRouter(prefix="/chat", tags=["chat"])

PROMPT_TEMPLATE = """Kamu adalah asisten AI untuk tanya jawab kitab.
Jawab dalam Bahasa Indonesia hanya berdasarkan konteks dokumen berbahasa Arab berikut.

Konteks:
{context}

Pertanyaan:
{query}

Aturan jawaban:
- Setiap klaim faktual harus didukung sumber dengan format [S1], [S2], dan seterusnya.
- Jika pertanyaan meminta daftar berjumlah tertentu, berikan setiap poin yang terbukti dari konteks.
- Jika konteks hanya membuktikan sebagian poin, jawab parsial dan sebutkan bahwa sisanya belum terbukti dari konteks yang diberikan.
- Jangan menyatakan bahwa dokumen asli tidak memuat daftar lengkap kecuali semua sumber yang diberikan memang membuktikan hal itu.
- Jangan menambahkan poin dari pengetahuan umum atau hafalan di luar konteks.
"""


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
    llm_router: LLMRouter = Depends(get_llm_router),
) -> ChatResponse:
    current_profile = build_embedding_profile(settings)
    _raise_if_stale_embeddings(session, request.book_filter, current_profile)

    retrieval = retrieve_context(
        query=request.query,
        book_filter=request.book_filter,
        embedding_fingerprint=current_profile.fingerprint,
        settings=settings,
        embedding_provider=embedding_provider,
        qdrant_store=qdrant_store,
    )
    results = retrieval.results

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
            answer_status="insufficient",
            retrieval_summary=retrieval.summary,
            citations=[],
        )

    context = _format_sources_for_prompt(results)
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

    citations = _extract_valid_citations(generation.answer, source_count=len(results))
    answer_status = _answer_status(
        answer=generation.answer,
        citations=citations,
        valid_citation_marker_count=_count_valid_citation_markers(generation.answer, source_count=len(results)),
        requested_count=retrieval.requested_count,
    )
    answer = _ensure_partial_notice(
        generation.answer,
        answer_status=answer_status,
        requested_count=retrieval.requested_count,
    )

    return ChatResponse(
        answer=answer,
        provider_used=generation.provider_used,
        sources=sources,
        answer_status=answer_status,
        retrieval_summary=retrieval.summary,
        citations=citations,
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


def _format_sources_for_prompt(results) -> str:
    formatted = []
    for index, result in enumerate(results, start=1):
        metadata = result.metadata
        heading = metadata.get("heading") or "-"
        formatted.append(
            "\n".join(
                [
                    f"[S{index}]",
                    f"book_id: {metadata.get('book_id', '-')}",
                    f"title: {metadata.get('title', '-')}",
                    f"chapter: {metadata.get('chapter', '-')}",
                    f"chunk_index: {metadata.get('chunk_index', '-')}",
                    f"heading: {heading}",
                    "text:",
                    result.document,
                ]
            )
        )
    return "\n\n---\n\n".join(formatted)


def _extract_valid_citations(answer: str, *, source_count: int) -> list[str]:
    citations = []
    for raw in re.findall(r"\[S([0-9]+)\]", answer):
        index = int(raw)
        if 1 <= index <= source_count:
            citations.append(f"S{index}")
    return list(dict.fromkeys(citations))


def _answer_status(
    *,
    answer: str,
    citations: list[str],
    valid_citation_marker_count: int,
    requested_count: Optional[int],
) -> str:
    lowered = answer.lower()
    if ("tidak menemukan konteks" in lowered or "tidak cukup" in lowered) and not citations:
        return "insufficient"

    if requested_count:
        listed_points = _count_listed_points(answer)
        if listed_points >= requested_count and valid_citation_marker_count >= requested_count:
            return "complete"
        if citations or listed_points:
            return "partial"
        return "insufficient"

    return "complete" if citations else "partial"


def _count_listed_points(answer: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[0-9]{1,2}[\).]|[-*])\s+", answer))


def _count_valid_citation_markers(answer: str, *, source_count: int) -> int:
    count = 0
    for raw in re.findall(r"\[S([0-9]+)\]", answer):
        index = int(raw)
        if 1 <= index <= source_count:
            count += 1
    return count


def _ensure_partial_notice(answer: str, *, answer_status: str, requested_count: Optional[int]) -> str:
    if answer_status != "partial" or "parsial" in answer.lower():
        return answer
    if requested_count:
        return (
            "Jawaban parsial: konteks yang ditemukan belum cukup untuk memverifikasi "
            f"seluruh {requested_count} poin yang diminta.\n\n{answer}"
        )
    return f"Jawaban parsial berdasarkan konteks yang ditemukan.\n\n{answer}"
