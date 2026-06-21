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

# Mengimpor komponen FastAPI untuk router web, dependensi, upload file, form input, exception HTTP, dan status HTTP
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
# Mengimpor kelas Session (koneksi database) dan fungsi query select dari SQLModel
from sqlmodel import Session, select

# Mengimpor kelas Settings untuk membaca konfigurasi aplikasi
from app.config import Settings
# Mengimpor get_session untuk mendapatkan koneksi database aktif
from app.database import get_session
# Mengimpor fungsi penyedia dependensi aplikasi (settings, qdrant, embeddings)
from app.dependencies import get_app_settings, get_qdrant_store, get_embedding_provider
# Mengimpor model tabel Document database
from app.models import Document
# Mengimpor profil embedding, provider embedding, dan fungsi pembangun profil model aktif
from app.providers.embeddings import EmbeddingProfile, EmbeddingProvider, build_embedding_profile
# Mengimpor skema data untuk respons import dokumen, detail dokumen, dan status model embedding
from app.schemas import DocumentImportResponse, DocumentRead, EmbeddingState
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
    # Sesi database
    session: Session = Depends(get_session),
    # Konfigurasi aplikasi
    settings: Settings = Depends(get_app_settings),
) -> list[DocumentRead]:
    # Membangun profil model embedding aktif berdasarkan konfigurasi saat ini
    current_profile = build_embedding_profile(settings)
    # Mencari semua baris data Document dari database diurutkan berdasarkan tanggal pendaftaran terbaru
    documents = session.exec(select(Document).order_by(Document.created_at.desc())).all()
    # Mengembalikan format detail dokumen dilengkapi verifikasi apakah sidik jari model embeddingnya cocok
    return [
        DocumentRead(
            # Memasukkan seluruh kolom model database ke skema
            **document.model_dump(),
            # Membandingkan sidik jari dokumen dengan sidik jari model aktif saat ini
            is_embedding_current=document.embedding_fingerprint == current_profile.fingerprint,
        )
        # Iterasi setiap dokumen
        for document in documents
    ]


# Menentukan rute POST '/import' untuk mengunggah dan memproses dokumen buku format EPUB menjadi vektor (RAG)
@router.post("/import", response_model=DocumentImportResponse, status_code=status.HTTP_201_CREATED)
# Fungsi asinkron untuk menangani proses impor buku EPUB
async def import_document(
    # File biner EPUB wajib diunggah
    file: UploadFile = File(...),
    # Judul buku (opsional, jika kosong akan dibaca dari metadata EPUB)
    title: str | None = Form(default=None),
    # Nama penulis buku (opsional, jika kosong dibaca dari metadata EPUB)
    author: str | None = Form(default=None),
    # Sesi database
    session: Session = Depends(get_session),
    # Konfigurasi aplikasi
    settings: Settings = Depends(get_app_settings),
    # Provider embedding aktif
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    # Client penyimpanan Qdrant aktif
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
) -> DocumentImportResponse:
    # Memeriksa apakah file yang diunggah tidak memiliki nama atau tidak berakhiran '.epub' (case-insensitive)
    if not file.filename or not file.filename.lower().endswith(".epub"):
        # Melemparkan HTTP Exception 400 Bad Request
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only EPUB files are supported")

    # Menulis file unggahan biner tersebut ke berkas temporer lokal di server
    temp_path = await _write_temp_upload(file)
    # Menghasilkan ID buku unik menggunakan UUID v4
    book_id = str(uuid4())
    try:
        # Pengekstrakan EPUB: Membaca teks, bab, dan metadata dari file EPUB temporer
        extracted = extract_epub(temp_path)
        # Pemotongan Dokumen (Chunking): Membagi bab-bab buku menjadi potongan teks (chunks) kecil
        chunks = chunk_chapters(
            # Daftar bab terektraksi
            extracted.chapters,
            # Batas ukuran karakter tiap potongan (dari settings)
            chunk_size=settings.chunk_size,
            # Jumlah karakter tumpang tindih antar potongan (dari settings)
            chunk_overlap=settings.chunk_overlap,
        )
        # Jika tidak ada potongan teks yang berhasil diekstraksi
        if not chunks:
            # Melemparkan HTTP Exception 400 Bad Request
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text chunks could be extracted from the EPUB",
            )

        # Mengumpulkan teks isi dokumen dari seluruh objek potongan teks
        texts = [chunk.text for chunk in chunks]
        # Vektorisasi (Embedding): Mengonversi seluruh teks potongan menjadi representasi vektor numerik via API embedding
        embeddings = embedding_provider.embed_documents(texts)
        # Menghitung dimensi dimensi vektor hasil konversi (atau default ke dimensi profil jika kosong)
        actual_dimension = len(embeddings[0]) if embeddings else embedding_provider.profile.dimension
        # Mendapatkan objek profil model embedding
        profile = embedding_provider.profile

        # Menentukan judul akhir buku (mengutamakan parameter input, lalu metadata EPUB, terakhir nama file asli)
        resolved_title = title or extracted.title or Path(file.filename).stem
        # Menentukan nama penulis akhir buku (mengutamakan parameter input, lalu metadata EPUB)
        resolved_author = author or extracted.author
        # Menyusun ID unik terformat untuk setiap potongan vektor di Qdrant (berupa gabungan book_id dan nomor bab/chunk)
        ids = [
            f"{book_id}_chapter_{chunk.chapter}_chunk_{chunk.chunk_index}"
            for chunk in chunks
        ]
        # Menyusun payload metadata lengkap untuk setiap potongan dokumen
        metadatas = _build_chunk_metadatas(
            chunks=chunks,
            ids=ids,
            book_id=book_id,
            title=resolved_title,
            profile=profile,
            actual_dimension=actual_dimension,
        )
        # Menyimpan seluruh ID, teks, vektor, dan metadata ke database Qdrant
        qdrant_store.add_chunks(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Membuat rekam baris baru model Document untuk dicatat di database relasional
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
            # Memasukkan data dokumen baru ke antrean database session
            session.add(document)
            # Melakukan komit transaksi ke database
            session.commit()
            # Memuat ulang data dari database
            session.refresh(document)
        # Menangkap kegagalan pencatatan di database relasional
        except Exception:
            # Batalkan seluruh transaksi penambahan data database (rollback)
            session.rollback()
            # Hapus data vektor buku yang terlanjur terkirim ke Qdrant demi menjaga integritas data (cleanup)
            qdrant_store.delete_book(book_id)
            # Lempar kembali exception ke atas
            raise

        # Mengembalikan informasi detail buku yang sukses diimpor
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
    # Blok finally akan selalu dijalankan untuk membersihkan sisa pemrosesan file temporer
    finally:
        try:
            # Menghapus file EPUB temporer dari media penyimpanan server
            os.unlink(temp_path)
        # Abaikan error jika file temporer tidak ditemukan / gagal dihapus
        except OSError:
            pass


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
    # Daftar potongan dokumen Chunk
    chunks: list[Chunk],
    # Daftar string ID unik chunk
    ids: list[str],
    # ID unik buku
    book_id: str,
    # Judul buku
    title: str,
    # Profil model embedding
    profile: EmbeddingProfile,
    # Dimensi vektor aktual
    actual_dimension: int | None,
) -> list[dict]:
    # Memetakan daftar chunk menjadi data metadata JSON terstruktur
    return [
        {
            # ID buku induk
            "book_id": book_id,
            # ID unik potongan chunk vektor
            "chunk_id": ids[index],
            # Judul buku
            "title": title,
            # Nomor urut bab
            "chapter": chunk.chapter,
            # Nomor urut chunk
            "chunk_index": chunk.chunk_index,
            # Nama sub-bab / heading
            "heading": chunk.heading,
            # Angka indeks heading (default 0 jika tidak ada)
            "heading_number": chunk.heading_number or 0,
            # Menyambungkan pointer ID chunk sebelumnya (kosong jika chunk pertama)
            "prev_id": ids[index - 1] if index > 0 else "",
            # Menyambungkan pointer ID chunk berikutnya (kosong jika chunk terakhir)
            "next_id": ids[index + 1] if index + 1 < len(ids) else "",
            # Menyimpan hash SHA-256 isi teks chunk untuk mendeteksi perubahan konten
            "text_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            # Nama provider model embedding
            "embedding_provider": profile.provider,
            # Nama model embedding
            "embedding_model": profile.model,
            # Dimensi vektor
            "embedding_dimension": actual_dimension,
            # Sidik jari konfigurasi model embedding
            "embedding_fingerprint": profile.fingerprint,
        }
        # Iterasi seluruh data potongan chunk
        for index, chunk in enumerate(chunks)
    ]
