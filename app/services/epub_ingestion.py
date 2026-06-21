# Mengaktifkan evaluasi tipe data bertunda (postponed evaluation of annotations) agar kompatibel ke belakang
from __future__ import annotations

# Mengimpor modul regular expression (re) untuk mencocokkan pola heading bernomor
import re
# Mengimpor dataclass untuk mempermudah struktur model data bab, buku, dan chunk
from dataclasses import dataclass
# Mengimpor Path untuk pengolahan direktori file secara lintas platform
from pathlib import Path
# Mengimpor Optional untuk type-hinting data yang boleh bernilai None
from typing import Optional


# Kelas data untuk menampung teks mentah per bab (chapter) dari file EPUB
@dataclass
class EpubChapter:
    # Urutan nomor bab
    chapter: int
    # Konten isi teks bab tersebut
    text: str


# Kelas data untuk menampung hasil ekstraksi buku EPUB (judul, penulis, dan daftar bab)
@dataclass
class ExtractedEpub:
    # Judul buku (opsional)
    title: Optional[str]
    # Nama penulis/pengarang buku (opsional)
    author: Optional[str]
    # Daftar bab yang berhasil diekstraksi
    chapters: list[EpubChapter]


# Kelas data untuk mewakili potongan teks (chunk) yang siap ditransfer ke database vektor
@dataclass
class Chunk:
    # Nomor bab asal potongan teks
    chapter: int
    # Indeks urutan potongan teks di dalam bab bersangkutan
    chunk_index: int
    # Isi potongan teks
    text: str
    # Judul sub-bab (heading) chunk ini
    heading: str = ""
    # Angka urutan heading (misalnya poin ke-1, ke-2, dll)
    heading_number: Optional[int] = None


# Kamus terjemahan angka ordinal bahasa Arab menjadi nilai integer
ARABIC_ORDINALS: dict[str, int] = {
    "الأول": 1,
    "الأولى": 1,
    "الاول": 1,
    "الاولى": 1,
    "الثاني": 2,
    "الثانية": 2,
    "الثالث": 3,
    "الثالثة": 3,
    "الرابع": 4,
    "الرابعة": 4,
    "الخامس": 5,
    "الخامسة": 5,
    "السادس": 6,
    "السادسة": 6,
    "السابع": 7,
    "السابعة": 7,
    "الثامن": 8,
    "الثامنة": 8,
    "التاسع": 9,
    "التاسعة": 9,
    "العاشر": 10,
    "العاشرة": 10,
}

# Regex untuk mendeteksi heading berangka desimal/arab (contoh: "1. Mukadimah" atau "١) Bahasan")
NUMBERED_HEADING_RE = re.compile(
    r"^\s*[\(\[]?\s*([0-9٠-٩۰-۹]{1,3})\s*[\)\].:\-–—/،]\s*(.+?)\s*$"
)
# Regex untuk mendeteksi heading bertuliskan huruf angka ordinal Arab (contoh: "الأمر الأول : العلم")
ORDINAL_HEADING_RE = re.compile(
    r"^\s*(?:الطريقة|الوسيلة|الأمر|السبب)?\s*"
    r"(الأول|الأولى|الاول|الاولى|الثاني|الثانية|الثالث|الثالثة|الرابع|الرابعة|"
    r"الخامس|الخامسة|السادس|السadسة|السابع|السابعة|الثامن|الثامنة|التاسع|التاسعة|"
    r"العاشر|العاشرة)\s*[:\-–—/،]\s*(.+?)\s*$"
)


# Fungsi untuk membaca dan mengekstrak isi buku dari file format EPUB
def extract_epub(path: str | Path) -> ExtractedEpub:
    # Impor modul ebooklib secara lokal
    import ebooklib
    # Impor BeautifulSoup untuk parsing dokumen HTML di dalam EPUB
    from bs4 import BeautifulSoup
    # Impor modul epub dari ebooklib
    from ebooklib import epub

    # Membaca file EPUB dari path yang diberikan
    book = epub.read_epub(str(path))
    # Mengambil metadata judul buku pertama
    title = _metadata_first(book, "title")
    # Mengambil metadata pencipta/penulis buku pertama
    author = _metadata_first(book, "creator")

    # Membuat mapping ID item ke item dokumen EPUB untuk mempercepat pembacaan spine
    id_to_item = {item.get_id(): item for item in book.get_items()}
    # Menyiapkan list untuk menampung item dokumen sesuai urutan bacaan (spine)
    ordered_items = []
    # Iterasi data spine buku EPUB
    for spine_entry in book.spine:
        # Mengambil idref dari spine entry
        idref = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
        # Mengambil objek item berdasarkan idref
        item = id_to_item.get(idref)
        # Jika item terdaftar
        if item is not None:
            # Masukkan ke list ordered_items
            ordered_items.append(item)

    # Jika daftar spine kosong, ambil semua item bertipe dokumen HTML secara berurutan
    if not ordered_items:
        ordered_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    # Inisialisasi list penampung bab
    chapters: list[EpubChapter] = []
    # Iterasi seluruh dokumen HTML berurutan
    for index, item in enumerate(ordered_items, start=1):
        # Abaikan jika tipe item bukan dokumen teks/HTML
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        # Parsing konten biner HTML menggunakan BeautifulSoup
        soup = BeautifulSoup(item.get_content(), "html.parser")
        # Mengambil teks polos dari HTML dibatasi tanda baris baru
        text = soup.get_text("\n", strip=True)
        # Jika teks bab tidak kosong
        if text:
            # Tambahkan objek EpubChapter ke list bab
            chapters.append(EpubChapter(chapter=index, text=text))

    # Mengembalikan hasil ekstraksi buku
    return ExtractedEpub(title=title, author=author, chapters=chapters)


# Fungsi untuk membagi bab-bab buku menjadi potongan teks (chunks) kecil
def chunk_chapters(
    # Daftar bab buku
    chapters: list[EpubChapter],
    # Parameter ukuran chunk maksimal
    chunk_size: int,
    # Parameter tumpang tindih antar chunk
    chunk_overlap: int,
) -> list[Chunk]:
    # Memastikan ukuran chunk valid lebih dari nol
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    # Memastikan overlap tidak negatif
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    # Memastikan overlap lebih kecil dari ukuran chunk
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Inisialisasi list chunks
    chunks: list[Chunk] = []
    # Iterasi setiap bab
    for chapter in chapters:
        # Memotong teks bab menggunakan helper _chunk_chapter_text
        chapter_chunks = _chunk_chapter_text(
            chapter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # Menambahkan hasil potongan bab ke list utama chunks
        chunks.extend(chapter_chunks)
    # Mengembalikan daftar potongan chunk final
    return chunks


# Fungsi internal untuk memotong teks satu bab berdasarkan pemisahan heading bernomor
def _chunk_chapter_text(
    chapter: EpubChapter,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    # Memecah teks bab berdasarkan baris dan membuang baris kosong
    lines = [line.strip() for line in chapter.text.splitlines() if line.strip()]
    # Membagi baris-baris teks menjadi beberapa bagian (sections) berdasarkan heading bernomor
    sections = _split_numbered_sections(lines)

    # Inisialisasi list chunk dan indeks urutan
    chunks: list[Chunk] = []
    chunk_index = 0
    # Iterasi setiap bagian (section)
    for section in sections:
        # Memotong teks di dalam bagian section menggunakan helper _split_section_text
        for text in _split_section_text(
            section["lines"],
            heading=section["heading"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            # Memasukkan objek Chunk ke list
            chunks.append(
                Chunk(
                    chapter=chapter.chapter,
                    chunk_index=chunk_index,
                    text=text,
                    heading=section["heading"],
                    heading_number=section["heading_number"],
                )
            )
            # Menambahkan indeks chunk
            chunk_index += 1
    # Mengembalikan kumpulan chunk dari bab ini
    return chunks


# Memisahkan baris-baris teks bab menjadi dictionary section berdasarkan penemuan heading bernomor
def _split_numbered_sections(lines: list[str]) -> list[dict[str, object]]:
    # Inisialisasi list penampung section
    sections: list[dict[str, object]] = []
    # Menyiapkan list untuk menampung baris saat ini
    current_lines: list[str] = []
    # Inisialisasi heading saat ini
    current_heading = ""
    # Inisialisasi angka heading saat ini
    current_heading_number: Optional[int] = None

    # Iterasi setiap baris
    for line in lines:
        # Mencoba memparsing baris apakah berupa heading bernomor
        heading_number, heading = parse_numbered_heading(line)
        # Jika terdeteksi ada heading baru dan list baris sebelumnya tidak kosong
        if heading_number is not None and current_lines:
            # Simpan baris-baris sebelumnya sebagai satu section utuh
            sections.append(
                {
                    "lines": current_lines,
                    "heading": current_heading,
                    "heading_number": current_heading_number,
                }
            )
            # Reset penampung baris
            current_lines = []

        # Jika baris ini merupakan heading baru, simpan info headingnya
        if heading_number is not None:
            current_heading = heading
            current_heading_number = heading_number
        # Masukkan baris ini ke list baris saat ini
        current_lines.append(line)

    # Jika masih ada sisa baris di akhir dokumen, simpan sebagai section terakhir
    if current_lines:
        sections.append(
            {
                "lines": current_lines,
                "heading": current_heading,
                "heading_number": current_heading_number,
            }
        )
    # Mengembalikan daftar section
    return sections


# Membagi teks section baris-perbaris agar ukurannya tidak melebihi batas chunk_size
def _split_section_text(
    lines: list[str],
    *,
    heading: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    # Inisialisasi list chunk teks
    chunks: list[str] = []
    # Menampung baris-baris chunk saat ini
    current: list[str] = []

    # Iterasi setiap baris
    for line in lines:
        # Membuat usulan teks gabungan dengan baris baru
        proposed = "\n".join([*current, line]).strip()
        # Jika usulan panjang teks melebihi batas chunk_size
        if current and len(proposed) > chunk_size:
            # Simpan teks chunk saat ini ke list chunks
            chunks.append("\n".join(current).strip())
            # Membuat seed awal kelanjutan chunk berikutnya menggunakan sisa baris tumpang tindih (overlap)
            current = _continuation_seed(current, heading, chunk_overlap)

        # Jika satu baris tersendiri saja sudah melebihi batas maksimal chunk_size
        if len(line) > chunk_size:
            # Jika ada teks di penampung, simpan dulu sebagai chunk
            if current:
                chunks.append("\n".join(current).strip())
                current = _continuation_seed(current, heading, chunk_overlap)
            # Memotong paksa baris yang sangat panjang tersebut menjadi beberapa potongan chunk
            chunks.extend(_split_long_line(line, heading=heading, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
            # Kosongkan penampung current
            current = []
            continue

        # Masukkan baris ke penampung saat ini
        current.append(line)

    # Simpan sisa baris terakhir di penampung jika ada
    if current:
        chunks.append("\n".join(current).strip())

    # Mengembalikan daftar teks potongan chunk yang tidak kosong
    return [chunk for chunk in chunks if chunk]


# Mempersiapkan baris seed awal untuk chunk berikutnya menggunakan sisa baris overlap dan menyematkan heading bab
def _continuation_seed(lines: list[str], heading: str, chunk_overlap: int) -> list[str]:
    # Jika overlap diset 0, kembalikan hanya heading (jika ada)
    if chunk_overlap <= 0:
        return [heading] if heading else []

    # Inisialisasi list seed
    seed: list[str] = []
    # Menghitung panjang karakter
    total = 0
    # Membaca baris dari urutan terbalik (paling akhir) untuk memenuhi batas tumpang tindih overlap
    for line in reversed(lines):
        # Abaikan jika baris sama dengan heading
        if heading and line == heading:
            continue
        # Menghitung usulan panjang karakter
        next_total = total + len(line)
        # Jika total karakter sudah melebihi batas overlap, hentikan pengambilan baris
        if seed and next_total > chunk_overlap:
            break
        # Sisipkan baris di urutan depan seed
        seed.insert(0, line)
        # Perbarui total panjang karakter
        total = next_total

    # Menyematkan teks heading di urutan pertama list seed jika belum ada
    if heading and (not seed or seed[0] != heading):
        seed.insert(0, heading)
    # Mengembalikan list seed
    return seed


# Memotong satu baris teks yang sangat panjang menjadi beberapa potongan chunk berukuran chunk_size
def _split_long_line(
    line: str,
    *,
    heading: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    # Inisialisasi chunks
    chunks: list[str] = []
    # Tentukan teks awalan prefix (berupa teks heading jika baris tidak dimulai dengan heading tersebut)
    prefix = f"{heading}\n" if heading and not line.startswith(heading) else ""
    # Menghitung kapasitas karakter yang tersedia di sisa chunk setelah dikurangi panjang prefix
    available = max(1, chunk_size - len(prefix))
    # Indeks mulai pemotongan
    start = 0
    # Lakukan loop pemotongan sepanjang teks baris
    while start < len(line):
        # Tentukan indeks batas akhir pemotongan
        end = min(start + available, len(line))
        # Masukkan potongan teks berserta prefix ke list chunks
        chunks.append(f"{prefix}{line[start:end]}".strip())
        # Keluar dari loop jika sudah mencapai akhir baris
        if end >= len(line):
            break
        # Perbarui indeks mulai berdasarkan overlap agar teks tetap berkesinambungan
        start = max(start + 1, end - chunk_overlap)
    # Mengembalikan list potongan baris panjang
    return chunks


# Memparsing baris teks untuk mendeteksi apakah berupa heading berangka desimal atau ordinal arab
def parse_numbered_heading(line: str) -> tuple[Optional[int], str]:
    # Mencoba mencocokkan dengan regex angka desimal/Arab
    numbered = NUMBERED_HEADING_RE.match(line)
    # Jika cocok
    if numbered:
        # Mengembalikan angka terjemahan integer dan teks judul heading
        return _parse_int(numbered.group(1)), numbered.group(2).strip()

    # Mencoba mencocokkan dengan regex ordinal bahasa Arab
    ordinal = ORDINAL_HEADING_RE.match(line)
    # Jika cocok
    if ordinal:
        # Mengambil nilai angka dari kamus ARABIC_ORDINALS dan teks judul heading
        return ARABIC_ORDINALS[ordinal.group(1)], ordinal.group(2).strip()

    # Jika tidak cocok pola heading mana pun, kembalikan None dan string kosong
    return None, ""


# Menerjemahkan angka karakter Arab/Persia menjadi angka integer standar
def _parse_int(value: str) -> int:
    # Membuat tabel transliterasi karakter angka Arab/Persia ke desimal
    translation = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    # Menerjemahkan dan mengubahnya menjadi tipe data integer
    return int(value.translate(translation))


# Fungsi pembantu untuk mengambil metadata Dublin Core (DC) pertama dari buku EPUB
def _metadata_first(book, key: str) -> Optional[str]:
    # Meminta metadata Dublin Core (DC) berdasarkan key
    metadata = book.get_metadata("DC", key)
    # Jika metadata kosong
    if not metadata:
        return None
    # Mengambil nilai elemen pertama metadata
    value = metadata[0][0]
    # Mengembalikan string bersih dari nilai tersebut (atau None jika kosong)
    return str(value).strip() or None
