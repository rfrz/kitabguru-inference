from app.api.documents import _build_chunk_metadatas
from app.providers.embeddings import EmbeddingProfile
from app.services.epub_ingestion import Chunk


def test_build_chunk_metadatas_links_neighbor_ids_and_hashes_text():
    profile = EmbeddingProfile(
        provider="huggingface",
        model="intfloat/multilingual-e5-large",
        dimension=1024,
        behavior="e5-query-passage-prefix",
    )
    chunks = [
        Chunk(chapter=1, chunk_index=0, text="first", heading="one", heading_number=1),
        Chunk(chapter=1, chunk_index=1, text="second", heading="two", heading_number=2),
    ]
    ids = ["book-1_chapter_1_chunk_0", "book-1_chapter_1_chunk_1"]

    metadatas = _build_chunk_metadatas(
        chunks=chunks,
        ids=ids,
        book_id="book-1",
        title="Kitab",
        profile=profile,
        actual_dimension=1024,
    )

    assert metadatas[0]["chunk_id"] == ids[0]
    assert metadatas[0]["prev_id"] == ""
    assert metadatas[0]["next_id"] == ids[1]
    assert metadatas[1]["prev_id"] == ids[0]
    assert metadatas[1]["next_id"] == ""
    assert metadatas[1]["heading_number"] == 2
    assert metadatas[0]["text_hash"] != metadatas[1]["text_hash"]
