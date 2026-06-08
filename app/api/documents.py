from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlmodel import Session, select

from app.config import Settings
from app.database import get_session
from app.dependencies import get_app_settings, get_qdrant_store, get_embedding_provider
from app.models import Document
from app.providers.embeddings import EmbeddingProfile, EmbeddingProvider, build_embedding_profile
from app.schemas import DocumentImportResponse, DocumentRead, EmbeddingState
from app.services.qdrant_store import QdrantStore
from app.services.epub_ingestion import Chunk, chunk_chapters, extract_epub

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentRead])
def list_documents(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> list[DocumentRead]:
    current_profile = build_embedding_profile(settings)
    documents = session.exec(select(Document).order_by(Document.created_at.desc())).all()
    return [
        DocumentRead(
            **document.model_dump(),
            is_embedding_current=document.embedding_fingerprint == current_profile.fingerprint,
        )
        for document in documents
    ]


@router.post("/import", response_model=DocumentImportResponse, status_code=status.HTTP_201_CREATED)
async def import_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    author: str | None = Form(default=None),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
) -> DocumentImportResponse:
    if not file.filename or not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only EPUB files are supported")

    temp_path = await _write_temp_upload(file)
    book_id = str(uuid4())
    try:
        extracted = extract_epub(temp_path)
        chunks = chunk_chapters(
            extracted.chapters,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text chunks could be extracted from the EPUB",
            )

        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_provider.embed_documents(texts)
        actual_dimension = len(embeddings[0]) if embeddings else embedding_provider.profile.dimension
        profile = embedding_provider.profile

        resolved_title = title or extracted.title or Path(file.filename).stem
        resolved_author = author or extracted.author
        ids = [
            f"{book_id}_chapter_{chunk.chapter}_chunk_{chunk.chunk_index}"
            for chunk in chunks
        ]
        metadatas = _build_chunk_metadatas(
            chunks=chunks,
            ids=ids,
            book_id=book_id,
            title=resolved_title,
            profile=profile,
            actual_dimension=actual_dimension,
        )
        qdrant_store.add_chunks(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        document = Document(
            book_id=book_id,
            title=resolved_title,
            author=resolved_author,
            total_chunks=len(chunks),
            embedding_provider=profile.provider,
            embedding_model=profile.model,
            embedding_dimension=actual_dimension,
            embedding_fingerprint=profile.fingerprint,
        )
        try:
            session.add(document)
            session.commit()
            session.refresh(document)
        except Exception:
            session.rollback()
            qdrant_store.delete_book(book_id)
            raise

        return DocumentImportResponse(
            book_id=document.book_id,
            title=document.title,
            author=document.author,
            total_chunks=document.total_chunks,
            embedding=EmbeddingState(
                provider=document.embedding_provider,
                model=document.embedding_model,
                dimension=document.embedding_dimension,
                fingerprint=document.embedding_fingerprint,
            ),
        )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    book_id: str,
    session: Session = Depends(get_session),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
) -> None:
    document = session.get(Document, book_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {book_id} not found")

    qdrant_store.delete_book(book_id)
    session.delete(document)
    session.commit()


async def _write_temp_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "upload.epub").suffix or ".epub"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            temp_file.write(chunk)
        return temp_file.name


def _build_chunk_metadatas(
    *,
    chunks: list[Chunk],
    ids: list[str],
    book_id: str,
    title: str,
    profile: EmbeddingProfile,
    actual_dimension: int | None,
) -> list[dict]:
    return [
        {
            "book_id": book_id,
            "chunk_id": ids[index],
            "title": title,
            "chapter": chunk.chapter,
            "chunk_index": chunk.chunk_index,
            "heading": chunk.heading,
            "heading_number": chunk.heading_number or 0,
            "prev_id": ids[index - 1] if index > 0 else "",
            "next_id": ids[index + 1] if index + 1 < len(ids) else "",
            "text_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            "embedding_provider": profile.provider,
            "embedding_model": profile.model,
            "embedding_dimension": actual_dimension,
            "embedding_fingerprint": profile.fingerprint,
        }
        for index, chunk in enumerate(chunks)
    ]
