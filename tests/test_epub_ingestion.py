# Mengimpor model EpubChapter dan fungsi chunk_chapters dari modul epub_ingestion
from app.services.epub_ingestion import EpubChapter, chunk_chapters


# Unit test 1: Menguji apakah pembagi chunk mematuhi aturan tumpang tindih (chunk_overlap)
def test_chunk_chapters_respects_overlap():
    # Memotong teks bab "abcdefghij" dengan ukuran 6 karakter dan overlap 2 karakter
    chunks = chunk_chapters(
        [EpubChapter(chapter=1, text="abcdefghij")],
        chunk_size=6,
        chunk_overlap=2,
    )

    # Memastikan teks chunk pertama adalah "abcdef" dan chunk kedua adalah "efghij" (karakter 'ef' tumpang tindih)
    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    # Memastikan indeks urutan chunk adalah 0 dan 1
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


# Unit test 2: Menguji apakah pembagi chunk berhasil mengidentifikasi heading bernomor bahasa Arab (seperti ١ dan ٢)
def test_chunk_chapters_preserves_numbered_arabic_headings():
    # Menyusun teks bab berisi dua heading bernomor Arab
    text = "\n".join(
        [
            "١- كف الأذى وبذل الندى",
            "شرح النقطة الأولى.",
            "٢- البدء بالسلام",
            "شرح النقطة الثانية.",
        ]
    )

    # Memotong teks bab dengan ukuran batas chunk 100 karakter
    chunks = chunk_chapters(
        [EpubChapter(chapter=1, text=text)],
        chunk_size=100,
        chunk_overlap=10,
    )

    # Memverifikasi angka heading yang terdeteksi untuk chunk pertama adalah 1 dan chunk kedua adalah 2
    assert [chunk.heading_number for chunk in chunks] == [1, 2]
    # Memverifikasi teks judul heading pertama terekstraksi bersih tanpa angka awalan
    assert chunks[0].heading == "كف الأذى وبذل الندى"
    # Memverifikasi teks judul heading kedua terekstraksi bersih tanpa angka
    assert chunks[1].heading == "البدء بالسلام"
    # Memastikan teks heading kedua tidak bocor masuk ke dalam konten teks chunk pertama
    assert "٢- البدء بالسلام" not in chunks[0].text
