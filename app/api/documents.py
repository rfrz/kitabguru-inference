# Mengaktifkan evaluasi tipe data bertunda (postponed evaluation of annotations) agar kompatibel ke belakang
from __future__ import annotations

# Mengimpor modul hashlib untuk menghitung hash SHA-256 dari teks chunk
import hashlib
# Mengimpor modul os untuk melakukan operasi file system (seperti menghapus file temporer)
import os
# Mengimpor modul tempfile untuk membuat file temporer lokal saat proses upload EPUB
import tempfile
# Mengimpor kelas Path dari pathlib untuk operasi manipulasi path file cross-platform
from pathlib import Path
# Mengimpor fungsi uuid4 untuk menghasilkan ID unik acak (UUID v4)
from uuid import uuid4

# Mengimpor komponen FastAPI untuk router web, dependensi, upload file, form input, exception HTTP, status HTTP, dan background tasks
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
# Mengimpor kelas Session (koneksi database) dan fungsi query select dari SQLModel
from sqlmodel import Session, select, col

# Mengimpor kelas Settings untuk membaca konfigurasi aplikasi
from app.config import Settings
# Mengimpor get_session dan engine untuk mendapatkan koneksi database
from app.database import engine, get_session
# Mengimpor fungsi penyedia dependensi aplikasi (settings, qdrant, embeddings)
from app.dependencies import get_app_settings, get_qdrant_store, get_embedding_provider
# Mengimpor model tabel Document dan DocumentTask database
from app.models import Document, DocumentTask
# Mengimpor profil embedding, provider embedding, dan fungsi pembangun profil model aktif
from app.providers.embeddings import EmbeddingProfile, EmbeddingProvider, build_embedding_profile
# Mengimpor skema data untuk respons import dokumen, detail dokumen, status model embedding, dll
from app.schemas import (
    DocumentImportResponse,
    DocumentRead,
    DocumentTaskResponse,
    DocumentUpdate,
    EmbeddingState,
)
# Mengimpor kelas penyimpanan pencarian vektor Qdrant
from app.services.qdrant_store import QdrantStore
# Mengimpor struktur data Chunk, fungsi pembagi chapter, dan fungsi pengekstraksi file EPUB
from app.services.epub_ingestion import Chunk, chunk_chapters, extract_epub

# Menginisialisasi router untuk rute '/documents' dengan tag dokumentasi 'documents'
router = APIRouter(prefix="/documents", tags=["documents"])


# Menentukan rute GET '/' untuk menampilkan daftar seluruh dokumen buku yang diimpor ke sistem
@router.get("", response_model=list[DocumentRead])
# Fungsi untuk menampilkan seluruh daftar dokumen
def list_documents(
    # Pagination skip
    skip: int = Query(0, ge=0),
    # Pagination limit
    limit: int = Query(100, ge=1, le=1000),
    # Parameter pencarian opsional (judul/penulis)
    search: str | None = Query(None),
    # Sesi database
    session: Session = Depends(get_session),
    # Konfigurasi aplikasi
    settings: Settings = Depends(get_app_settings),
) -> list[DocumentRead]:
    # Membangun profil model embedding aktif berdasarkan konfigurasi saat ini
    current_profile = build_embedding_profile(settings)
    
    # Menyusun kueri pencarian dasar
    query = select(Document).order_by(Document.created_at.desc())
    
    # Menambahkan filter pencarian jika ada
    if search:
        search_term = f"%{search}%"
        query = query.where(
            (col(Document.title).like(search_term)) | 
            (col(Document.author).like(search_term))
        )
        
    # Mengaplikasikan paginasi
    query = query.offset(skip).limit(limit)
    
    # Eksekusi kueri
    documents = session.exec(query).all()
    
    # Mengembalikan format detail dokumen dilengkapi verifikasi apakah sidik jari model embeddingnya cocok
    return [
        DocumentRead(
            **document.model_dump(),
            is_embedding_current=document.embedding_fingerprint == current_profile.fingerprint,
        )
        for document in documents
    ]


# Menentukan rute POST '/import' untuk mengunggah dan memproses dokumen buku format EPUB secara asinkron
@router.post("/import", response_model=DocumentImportResponse, status_code=status.HTTP_202_ACCEPTED)
async def import_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    author: str | None = Form(default=None),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> DocumentImportResponse:
    if not file.filename or not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only EPUB files are supported")

    # Tulis ke file temporer
    temp_path = await _write_temp_upload(file)
    task_id = str(uuid4())

    # Buat record tugas di database
    task_record = DocumentTask(
        task_id=task_id,
        status="PENDING",
        title=title or Path(file.filename).stem,
    )
    session.add(task_record)
    session.commit()

    # Kirimkan proses ke latar belakang (background tasks)
    background_tasks.add_task(
        _process_epub_task,
        task_id=task_id,
        temp_path=temp_path,
        file_name=file.filename,
        title=title,
        author=author,
        settings=settings,
    )

    return DocumentImportResponse(task_id=task_id, status="PENDING")

# Menentukan rute GET '/tasks/{task_id}' untuk mengecek status import
@router.get("/tasks/{task_id}", response_model=DocumentTaskResponse)
def get_task_status(
    task_id: str,
    session: Session = Depends(get_session),
) -> DocumentTaskResponse:
    task = session.get(DocumentTask, task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")
    return DocumentTaskResponse(**task.model_dump())

# Menentukan rute PATCH '/{book_id}' untuk update metadata
@router.patch("/{book_id}", response_model=DocumentRead)
def update_document(
    book_id: str,
    update_data: DocumentUpdate,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
) -> DocumentRead:
    document = session.get(Document, book_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {book_id} not found")

    # Update metadata di SQLite
    if update_data.title is not None:
        document.title = update_data.title
    if update_data.author is not None:
        document.author = update_data.author

    session.add(document)
    session.commit()
    session.refresh(document)

    # Update metadata Qdrant payload
    qdrant_store.update_book_metadata(
        book_id=book_id,
        new_title=update_data.title,
        new_author=update_data.author,
    )

    current_profile = build_embedding_profile(settings)
    return DocumentRead(
        **document.model_dump(),
        is_embedding_current=document.embedding_fingerprint == current_profile.fingerprint,
    )


# Menentukan rute DELETE '/{book_id}' untuk menghapus dokumen buku beserta seluruh data vektornya
@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
# Fungsi untuk menghapus dokumen buku
def delete_document(
    # ID buku target
    book_id: str,
    # Sesi database
    session: Session = Depends(get_session),
    # Client Qdrant
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
) -> None:
    # Mengambil baris data Document dari database berdasarkan ID buku
    document = session.get(Document, book_id)
    # Jika dokumen tidak ditemukan di database
    if document is None:
        # Melemparkan HTTP Exception 404 Not Found
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {book_id} not found")

    # Menghapus seluruh potongan vektor terkait book_id dari penyimpanan Qdrant
    qdrant_store.delete_book(book_id)
    # Menghapus baris data Document dari database relasional
    session.delete(document)
    # Melakukan komit untuk meresmikan penghapusan
    session.commit()


# Fungsi asinkron pembantu untuk menulis data biner unggahan file ke file temporer lokal
async def _write_temp_upload(file: UploadFile) -> str:
    # Mengambil ekstensi file asli (default: '.epub')
    suffix = Path(file.filename or "upload.epub").suffix or ".epub"
    # Membuat file temporer baru dengan akhiran ekstensi yang sesuai, jangan dihapus otomatis saat close
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        # Loop terus-menerus untuk membaca data biner per-blok 1MB
        while True:
            # Membaca data biner secara asinkron
            chunk = await file.read(1024 * 1024)
            # Jika tidak ada data tersisa, keluar dari loop
            if not chunk:
                break
            # Menulis blok biner ke file temporer
            temp_file.write(chunk)
        # Mengembalikan string path lokasi absolut file temporer yang berhasil ditulis
        return temp_file.name


# Fungsi pembantu untuk merakit list kamus metadata untuk setiap potongan chunk vektor
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


# Fungsi pemrosesan latar belakang untuk mengimpor file EPUB dan mengirimnya ke model embedding
def _process_epub_task(
    task_id: str,
    temp_path: str,
    file_name: str,
    title: str | None,
    author: str | None,
    settings: Settings,
) -> None:
    # Membuka sesi koneksi database sendiri karena ini background task
    with Session(engine) as session:
        # Ambil record tugas
        task = session.get(DocumentTask, task_id)
        if not task:
            os.unlink(temp_path)
            return

        book_id = str(uuid4())
        task.book_id = book_id
        task.status = "PROCESSING"
        session.add(task)
        session.commit()

        qdrant_store = QdrantStore(settings)
        embedding_profile = build_embedding_profile(settings)
        embedding_provider = get_provider(embedding_profile, settings.model_dump())

        try:
            # Ekstraksi EPUB
            extracted = extract_epub(temp_path)
            
            # Chunking
            chunks = chunk_chapters(
                extracted.chapters,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            if not chunks:
                raise ValueError("No text chunks could be extracted from the EPUB")

            task.total_chunks = len(chunks)
            session.add(task)
            session.commit()

            resolved_title = title or extracted.title or Path(file_name).stem
            resolved_author = author or extracted.author

            texts = [chunk.text for chunk in chunks]
            ids = [f"{book_id}_chapter_{chunk.chapter}_chunk_{chunk.chunk_index}" for chunk in chunks]
            
            # Batching embedding agar tidak over-limit memori dan dapat memantau progres (contoh per batch=50)
            batch_size = 50
            total = len(texts)
            actual_dimension = embedding_provider.profile.dimension
            
            for i in range(0, total, batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_chunks = chunks[i:i+batch_size]
                
                # Proses Embedding
                batch_embeddings = embedding_provider.embed_documents(batch_texts)
                if i == 0 and batch_embeddings:
                    actual_dimension = len(batch_embeddings[0])

                batch_metadatas = _build_chunk_metadatas(
                    chunks=batch_chunks,
                    ids=batch_ids,
                    book_id=book_id,
                    title=resolved_title,
                    profile=embedding_provider.profile,
                    actual_dimension=actual_dimension,
                )

                # Simpan ke Qdrant
                qdrant_store.add_chunks(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )

                # Update progress
                task.progress += len(batch_texts)
                session.add(task)
                session.commit()

            # Selesai, catat dokumen ke tabel relational
            document = Document(
                book_id=book_id,
                title=resolved_title,
                author=resolved_author,
                total_chunks=total,
                embedding_provider=embedding_provider.profile.provider,
                embedding_model=embedding_provider.profile.model,
                embedding_dimension=actual_dimension,
                embedding_fingerprint=embedding_provider.profile.fingerprint,
            )
            session.add(document)
            
            # Mark task success
            task.title = resolved_title
            task.status = "COMPLETED"
            session.add(task)
            session.commit()

        except Exception as e:
            # Tangkap kegagalan
            session.rollback()
            task.status = "FAILED"
            task.error_message = str(e)
            session.add(task)
            session.commit()
            
            # Cleanup Qdrant jika ada
            try:
                qdrant_store.delete_book(book_id)
            except Exception:
                pass
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
