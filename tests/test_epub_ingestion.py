from app.services.epub_ingestion import EpubChapter, chunk_chapters


def test_chunk_chapters_respects_overlap():
    chunks = chunk_chapters(
        [EpubChapter(chapter=1, text="abcdefghij")],
        chunk_size=6,
        chunk_overlap=2,
    )

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
