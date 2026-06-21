# Mengimpor fungsi pembantu perakit metadata chunk dari modul documents API
from app.api.documents import _build_chunk_metadatas
# Mengimpor profil model embedding
from app.providers.embeddings import EmbeddingProfile
# Mengimpor model data potongan Chunk
from app.services.epub_ingestion import Chunk


# Unit test untuk menguji perakitan metadata chunk terhubung tetangga (prev_id & next_id) serta kalkulasi hash teks
def test_build_chunk_metadatas_links_neighbor_ids_and_hashes_text():
    # Menyiapkan profil model embedding kustom
    profile = EmbeddingProfile(
        provider="huggingface",
        model="intfloat/multilingual-e5-large",
        dimension=1024,
        behavior="e5-query-passage-prefix",
    )
    # Membuat daftar potongan chunk dummy sebanyak 2 data
    chunks = [
        Chunk(chapter=1, chunk_index=0, text="first", heading="one", heading_number=1),
        Chunk(chapter=1, chunk_index=1, text="second", heading="two", heading_number=2),
    ]
    # Menyiapkan list ID unik buatan untuk setiap chunk
    ids = ["book-1_chapter_1_chunk_0", "book-1_chapter_1_chunk_1"]

    # Mengeksekusi fungsi perakitan metadata chunk
    metadatas = _build_chunk_metadatas(
        chunks=chunks,
        ids=ids,
        book_id="book-1",
        title="Kitab",
        profile=profile,
        actual_dimension=1024,
    )

    # Memverifikasi ID chunk pertama sesuai
    assert metadatas[0]["chunk_id"] == ids[0]
    # Memverifikasi chunk pertama tidak memiliki tetangga sebelum (karena data paling awal)
    assert metadatas[0]["prev_id"] == ""
    # Memverifikasi chunk pertama memiliki tetangga sesudah mengarah ke ID chunk kedua
    assert metadatas[0]["next_id"] == ids[1]
    # Memverifikasi chunk kedua memiliki tetangga sebelum mengarah ke ID chunk pertama
    assert metadatas[1]["prev_id"] == ids[0]
    # Memverifikasi chunk kedua tidak memiliki tetangga sesudah (karena data paling akhir)
    assert metadatas[1]["next_id"] == ""
    # Memverifikasi nomor urut heading chunk kedua diset bernilai 2
    assert metadatas[1]["heading_number"] == 2
    # Memastikan hash SHA-256 teks chunk pertama berbeda dengan hash teks chunk kedua
    assert metadatas[0]["text_hash"] != metadatas[1]["text_hash"]
