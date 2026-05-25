from app.services.epub_ingestion import EpubChapter, chunk_chapters


def test_chunk_chapters_respects_overlap():
    chunks = chunk_chapters(
        [EpubChapter(chapter=1, text="abcdefghij")],
        chunk_size=6,
        chunk_overlap=2,
    )

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_chunk_chapters_preserves_numbered_arabic_headings():
    text = "\n".join(
        [
            "١- كف الأذى وبذل الندى",
            "شرح النقطة الأولى.",
            "٢- البدء بالسلام",
            "شرح النقطة الثانية.",
        ]
    )

    chunks = chunk_chapters(
        [EpubChapter(chapter=1, text=text)],
        chunk_size=100,
        chunk_overlap=10,
    )

    assert [chunk.heading_number for chunk in chunks] == [1, 2]
    assert chunks[0].heading == "كف الأذى وبذل الندى"
    assert chunks[1].heading == "البدء بالسلام"
    assert "٢- البدء بالسلام" not in chunks[0].text
