# Mengimpor datetime, timezone untuk pengelolaan tanggal-waktu pendaftaran dokumen
from datetime import datetime, timezone
# Mengimpor Optional dari typing untuk menandai tipe data yang boleh kosong (None)
from typing import Optional

# Mengimpor Field dan SQLModel dari sqlmodel untuk deklarasi tabel database
from sqlmodel import Field, SQLModel


# Fungsi helper untuk mendapatkan tanggal dan waktu UTC saat ini
def utc_now() -> datetime:
    """Mengembalikan objek datetime UTC saat ini."""
    return datetime.now(timezone.utc)


# Kelas model Document mendefinisikan struktur tabel metadata dokumen/buku di database SQLite
class Document(SQLModel, table=True):
    # Menetapkan nama tabel fisik database SQLite
    __tablename__ = "documents"

    # Kolom book_id sebagai primary key bertipe string untuk menyimpan ID buku unik (biasanya bersumber dari nama file)
    book_id: str = Field(primary_key=True, index=True)
    # Kolom title menyimpan judul asli dokumen/buku
    title: str
    # Kolom author menyimpan nama penulis buku (opsional)
    author: Optional[str] = None
    # Kolom total_chunks mencatat jumlah total fragmen potongan teks hasil pembagian buku tersebut
    total_chunks: int
    # Kolom created_at mencatat tanggal impor buku ke sistem (default waktu UTC sekarang)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)

    # Kolom embedding_provider mencatat nama penyedia model embedding (huggingface / gemini) yang dipakai saat ingest
    embedding_provider: str
    # Kolom embedding_model mencatat nama model embedding yang dipakai
    embedding_model: str
    # Kolom embedding_dimension mencatat ukuran dimensi vektor dari model embedding
    embedding_dimension: Optional[int] = None
    # Kolom embedding_fingerprint menyimpan hash sidik jari konfigurasi embedding untuk melacak keaslian model
    embedding_fingerprint: str = Field(index=True)


# Kelas model DocumentTask mendefinisikan struktur tabel untuk melacak status proses latar belakang import EPUB
class DocumentTask(SQLModel, table=True):
    __tablename__ = "document_tasks"

    # ID tugas yang unik
    task_id: str = Field(primary_key=True, index=True)
    # ID buku terkait (bisa kosong jika gagal di tahap awal ekstraksi)
    book_id: Optional[str] = None
    # Judul buku sementara yang dideteksi
    title: Optional[str] = None
    # Status pemrosesan ("PENDING", "PROCESSING", "COMPLETED", "FAILED")
    status: str = Field(default="PENDING", index=True)
    # Jumlah chunk yang telah berhasil diproses
    progress: int = Field(default=0)
    # Total keseluruhan chunk yang perlu diproses
    total_chunks: int = Field(default=0)
    # Pesan kesalahan jika terjadi kegagalan (opsional)
    error_message: Optional[str] = None
    # Tanggal dan waktu tugas dibuat
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
