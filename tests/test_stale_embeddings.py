# Mengimpor pytest untuk menangkap exception pengujian
import pytest
# Mengimpor HTTPException dari FastAPI untuk verifikasi error respons API
from fastapi import HTTPException
# Mengimpor Session, SQLModel, dan create_engine dari SQLModel untuk inisialisasi database uji
from sqlmodel import Session, SQLModel, create_engine

# Mengimpor fungsi penilai status embedding dari chat API
from app.api.chat import _raise_if_stale_embeddings
# Mengimpor model tabel Document
from app.models import Document
# Mengimpor profil model embedding
from app.providers.embeddings import EmbeddingProfile


# Unit test: Memverifikasi apakah fungsi _raise_if_stale_embeddings melemparkan HTTP 409 Conflict berisi detail embedding
def test_stale_book_filter_returns_409_with_current_embedding_details():
    # Membuat engine database SQLite in-memory yang bersih
    engine = create_engine("sqlite:///:memory:")
    # Membuat tabel database uji
    SQLModel.metadata.create_all(engine)
    # Menyiapkan profil model embedding aktif saat ini (e5-base)
    current = EmbeddingProfile(
        provider="huggingface",
        model="intfloat/multilingual-e5-base",
        dimension=768,
        behavior="e5-query-passage-prefix",
    )
    # Menyiapkan profil model embedding lama yang disimulasikan tersimpan di database (e5-large)
    stored = EmbeddingProfile(
        provider="huggingface",
        model="intfloat/multilingual-e5-large",
        dimension=1024,
        behavior="e5-query-passage-prefix",
    )

    # Membuka sesi database uji
    with Session(engine) as session:
        # Menambahkan data dokumen dengan model large ke database uji
        session.add(
            Document(
                book_id="book-1",
                title="Kitab",
                total_chunks=1,
                embedding_provider=stored.provider,
                embedding_model=stored.model,
                embedding_dimension=stored.dimension,
                embedding_fingerprint=stored.fingerprint,
            )
        )
        # Melakukan komit database
        session.commit()

        # Memastikan pemanggilan fungsi validator memicu exception HTTPException
        with pytest.raises(HTTPException) as exc:
            # Membandingkan dokumen "book-1" dengan profil model aktif (current)
            _raise_if_stale_embeddings(session, "book-1", current)

    # Memverifikasi status kode HTTP bernilai 409 Conflict
    assert exc.value.status_code == 409
    # Memverifikasi detail data book_id bernilai "book-1"
    assert exc.value.detail["book_id"] == "book-1"
    # Memverifikasi nama model tersimpan adalah e5-large
    assert exc.value.detail["stored_embedding"]["model"] == "intfloat/multilingual-e5-large"
    # Memverifikasi nama model aktif saat ini adalah e5-base
    assert exc.value.detail["current_embedding"]["model"] == "intfloat/multilingual-e5-base"
