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
