# Mengimpor modul datetime untuk format waktu response
from datetime import datetime
# Mengimpor Any dan Optional dari typing untuk kebebasan tipe data dinamis dan opsi None
from typing import Any, Optional

# Mengimpor BaseModel dan Field dari Pydantic untuk data validation skema request/response
from pydantic import BaseModel, Field


# Skema ChatRequest memvalidasi request body saat mengirim pertanyaan RAG
class ChatRequest(BaseModel):
    # Menyatakan teks pertanyaan wajib diisi (minimal panjang 1 karakter)
    query: str = Field(min_length=1)
    # Menyatakan filter ID buku opsional jika ingin membatasi pencarian sumber hanya pada buku tertentu
    book_filter: Optional[str] = None


# Skema Source mewakili informasi satu potongan teks sumber referensi yang diambil dari Qdrant
class Source(BaseModel):
    # ID unik potongan teks
    id: str
    # Isi teks (fragmen potongan buku)
    document: str
    # Metadata dokumen (seperti nomor halaman, judul buku, dll)
    metadata: dict[str, Any]
    # Nilai jarak/skor kemiripan kosinus vektor pencarian (semakin kecil, semakin mirip)
    distance: Optional[float] = None


# Skema ChatResponse mewakili data respon balasan dari mesin inferensi RAG
class ChatResponse(BaseModel):
    # Teks respon jawaban dari AI
    answer: str
    # Nama provider LLM yang akhirnya sukses menghasilkan respon jawaban
    provider_used: str
    # Daftar fragmen teks referensi yang digunakan LLM untuk menyusun jawaban
    sources: list[Source]
    # Status kelengkapan jawaban dari hasil evaluasi RAG (misal: complete)
    answer_status: Optional[str] = None
    # Rangkuman informasi pencarian data (retrieval summary)
    retrieval_summary: Optional[dict[str, Any]] = None
    # Daftar kutipan rujukan (citations) halaman/buku
    citations: list[str] = Field(default_factory=list)


# Skema EmbeddingState memvalidasi status konfigurasi embedding dari dokumen
class EmbeddingState(BaseModel):
    # Nama provider embedding (misal: huggingface)
    provider: str
    # Nama model embedding
    model: str
    # Dimensi vektor embedding
    dimension: Optional[int] = None
    # Sidik jari hash konfigurasi embedding
    fingerprint: str


# Skema DocumentRead memvalidasi format respon saat membaca data detail buku dari database SQLite
class DocumentRead(BaseModel):
    # ID buku
    book_id: str
    # Judul buku
    title: str
    # Penulis buku (opsional)
    author: Optional[str] = None
    # Jumlah potongan teks buku
    total_chunks: int
    # Tanggal buku diimpor
    created_at: datetime
    # Provider embedding yang tercatat
    embedding_provider: str
    # Nama model embedding
    embedding_model: str
    # Dimensi vektor embedding
    embedding_dimension: Optional[int] = None
    # Sidik jari hash konfigurasi embedding
    embedding_fingerprint: str
    # Menandai apakah konfigurasi model embedding buku sama dengan model aktif saat ini di server (stale check)
    is_embedding_current: bool


# Skema DocumentImportResponse merespon status keberhasilan impor buku epub baru
class DocumentImportResponse(BaseModel):
    # ID buku
    book_id: str
    # Judul buku yang dideteksi
    title: str
    # Penulis buku
    author: Optional[str] = None
    # Jumlah total potongan teks yang di-ingest ke Qdrant
    total_chunks: int
    # Detail konfigurasi status embedding yang dihasilkan
    embedding: EmbeddingState


# Skema ProviderFailure mencatat logs kesalahan ketika sebuah provider LLM dalam antrean fallback mengalami error
class ProviderFailure(BaseModel):
    # Nama provider LLM yang gagal
    provider: str
    # Detail isi pesan kesalahan/error API
    error: str
