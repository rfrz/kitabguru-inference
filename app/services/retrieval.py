# Mengaktifkan evaluasi tipe data bertunda (postponed evaluation of annotations) agar kompatibel ke belakang
from __future__ import annotations

# Mengimpor modul regular expression (re) untuk mencocokkan pola jumlah poin dan sitasi
import re
# Mengimpor dataclass untuk representasi terstruktur hasil pencarian
from dataclasses import dataclass
# Mengimpor Any dan Optional untuk type-hinting python
from typing import Any, Optional

# Mengimpor kelas Settings untuk membaca konfigurasi pencarian RAG
from app.config import Settings
# Mengimpor tipe antarmuka EmbeddingProvider
from app.providers.embeddings import EmbeddingProvider
# Mengimpor router utama LLM
from app.providers.llm import LLMRouter
# Mengimpor QdrantStore dan model data SearchResult dari modul qdrant
from app.services.qdrant_store import QdrantStore, SearchResult


# Konstanta penentu pembobotan peringkat RRF (Reciprocal Rank Fusion)
RRF_RANK_CONSTANT = 60

# Kamus terjemahan kata angka dalam beberapa bahasa (Indonesian, English, Japanese) ke nilai integer
NUMBER_WORDS: dict[str, int] = {
    # Bahasa Indonesia
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10,
    # Bahasa Inggris
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    # Bahasa Jepang (Kanji)
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

# Kamus petunjuk kueri bahasa Arab untuk menerjemahkan kata kunci bahasa Indonesia ke padanannya di kitab Arab
ARABIC_QUERY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tetangga", ("الجار", "الجيران")),
    ("bertetangga", ("الجار", "الجيران")),
    ("memikat", ("كسب", "طرق لكسب")),
    ("menarik hati", ("كسب", "طرق لكسب")),
    ("hati", ("كسب",)),
    ("cara", ("طرق", "وسائل")),
    ("adab", ("آداب",)),
    ("salam", ("السلام",)),
    ("wajah", ("طلاقة الوجه",)),
    ("ceria", ("طلاقة الوجه",)),
    ("privasi", ("احترام الخصوصيات",)),
    ("nasihat", ("النصح",)),
    ("aib", ("الستر",)),
    ("kunjung", ("الزيارة",)),
    ("hadiah", ("هدية", "المجاملة")),
)


# Template prompt evaluator untuk menyaring chunk hasil pencarian vektor yang tidak berguna (useless)
CHUNK_EVALUATION_PROMPT_TEMPLATE = """Kamu adalah evaluator sistem RAG.
Berikut adalah cuplikan teks konteks (chunk) dari hasil pencarian.
Tentukan daftar `chunk_id` yang BENAR-BENAR 100% USELESS dan TIDAK ADA KAITANNYA SAMA SEKALI dengan pertanyaan.
PENTING: Jika ada teks yang memiliki sedikit saja kaitan atau relevansi dengan pertanyaan, JANGAN masukkan ke daftar useless.
Jawab HANYA dengan JSON valid.
Format: {{"useless_chunk_ids": ["chunk_id_1", "chunk_id_2"]}}

Konteks:
{context}

Pertanyaan:
{query}
"""


# Kelas data untuk membungkus hasil pencarian RAG lengkap beserta metadata prosesnya
@dataclass
class RetrievedContext:
    # Daftar akhir chunk dokumen yang terpilih
    results: list[SearchResult]
    # Variasi teks kueri pencarian yang digunakan
    query_variants: list[str]
    # Jumlah poin yang diminta (jika terdeteksi)
    requested_count: Optional[int]
    # Jumlah chunk pelengkap kelayakan informasi yang ditemukan
    completeness_found_count: int
    # Jumlah kandidat dokumen awal yang berhasil ditarik dari Qdrant
    candidate_count: int

    # Properti untuk mengembalikan ringkasan statistik pencarian RAG
    @property
    def summary(self) -> dict[str, Any]:
        return {
            "query_variants": self.query_variants,
            "requested_count": self.requested_count,
            "completeness_found_count": self.completeness_found_count,
            "candidate_count": self.candidate_count,
            "final_count": len(self.results),
        }


# Fungsi utama untuk melakukan penarikan (retrieval) konteks relevan dari Qdrant
def retrieve_context(
    *,
    # Teks pertanyaan pengguna
    query: str,
    # ID buku untuk pemfilteran kueri (opsional)
    book_filter: Optional[str],
    # Sidik jari model embedding aktif saat ini
    embedding_fingerprint: str,
    # Konfigurasi aplikasi
    settings: Settings,
    # Provider embedding
    embedding_provider: EmbeddingProvider,
    # Database Qdrant
    qdrant_store: QdrantStore,
    # Router LLM evaluator (opsional) untuk menyaring dokumen tidak relevan
    evaluator_llm_router: Optional[LLMRouter] = None,
) -> RetrievedContext:
    # Membangun variasi teks kueri pencarian (termasuk kata kunci terjemahan bahasa Arab)
    query_variants = build_query_variants(query)
    # Membangun filter kueri berdasarkan ID buku dan sidik jari model embedding
    where = build_vector_filter(book_id=book_filter, embedding_fingerprint=embedding_fingerprint)
    # Menarik daftar kandidat awal dari Qdrant menggunakan metode RRF di seluruh variasi kueri
    candidates = _retrieve_candidates(
        query_variants=query_variants,
        where=where,
        settings=settings,
        embedding_provider=embedding_provider,
        qdrant_store=qdrant_store,
    )

    # Inisialisasi set penampung ID chunk yang ditolak oleh model evaluator
    rejected_chunk_ids: set[str] = set()
    # Jika objek LLM evaluator disediakan
    if evaluator_llm_router:
        # Melakukan impor lokal untuk memformat teks sumber
        from app.api.chat import _format_sources_for_prompt
        # Memformat teks kandidat terpilih
        context = _format_sources_for_prompt(candidates)
        # Menyusun prompt filter chunk sampah
        prompt = CHUNK_EVALUATION_PROMPT_TEMPLATE.format(context=context, query=query)
        try:
            # Meminta model evaluator menyaring dokumen yang 100% tidak berguna
            eval_result = evaluator_llm_router.generate_json(prompt)
            useless_ids = eval_result.get("useless_chunk_ids", [])
            # Jika ditemukan ada ID dokumen yang sampah
            if useless_ids and isinstance(useless_ids, list):
                # Catat ID yang ditolak
                rejected_chunk_ids = set(str(uid) for uid in useless_ids)
                # Bersihkan dokumen yang ditolak dari daftar kandidat
                candidates = [c for c in candidates if str(c.id) not in rejected_chunk_ids]
        # Abaikan error penyaringan jika model gagal merespon
        except Exception:
            pass

    # Mengekstrak jumlah poin informasi yang diminta oleh pertanyaan (jika ada)
    requested_count = extract_requested_count(query)
    # Membatasi kandidat dokumen teratas berdasarkan parameter konfigurasi final K
    selected = candidates[: settings.retrieval_final_k]
    # Ekspansi Konteks: Mengambil chunk tetangga (sebelum & sesudah) untuk memperluas alur bacaan
    expanded = expand_with_neighbors(
        selected,
        qdrant_store=qdrant_store,
        # Rentang window pencarian tetangga
        window=settings.retrieval_neighbor_window,
        # Daftar ID dokumen yang diblacklist
        rejected_ids=rejected_chunk_ids,
    )

    # Inisialisasi list penampung chunk pelengkap kelengkapan informasi
    completeness_chunks: list[SearchResult] = []
    # Jika fitur pemindaian kelayakan informasi diaktifkan di settings dan terdapat permintaan jumlah poin spesifik
    if settings.rag_enable_completeness_scan and requested_count:
        # Memindai seluruh chunk di Qdrant untuk melengkapi nomor heading poin informasi yang masih kurang
        completeness_chunks = find_completeness_chunks(
            requested_count=requested_count,
            book_filter=book_filter,
            embedding_fingerprint=embedding_fingerprint,
            seed_results=expanded or selected,
            qdrant_store=qdrant_store,
            rejected_ids=rejected_chunk_ids,
        )

    # Menggabungkan seluruh hasil ekspansi dan hasil pemindaian kelayakan, lalu buang duplikatnya
    merged = dedupe_results([*expanded, *completeness_chunks])
    # Mengemas dan mengurutkan hasil akhir secara fisik berdasarkan chapter dan urutan chunk index buku
    final_results = repack_results(merged)

    # Mengembalikan objek detail RetrievedContext
    return RetrievedContext(
        results=final_results,
        query_variants=query_variants,
        requested_count=requested_count,
        completeness_found_count=len(completeness_chunks),
        candidate_count=len(candidates),
    )


# Fungsi internal untuk menarik kandidat dokumen teratas dari Qdrant menggunakan penggabungan nilai RRF
def _retrieve_candidates(
    *,
    # Daftar variasi kueri
    query_variants: list[str],
    # Filter metadata
    where: dict[str, Any],
    # Pengaturan aplikasi
    settings: Settings,
    # Provider embedding
    embedding_provider: EmbeddingProvider,
    # Database Qdrant
    qdrant_store: QdrantStore,
) -> list[SearchResult]:
    # Kamus penampung objek SearchResult berdasarkan ID
    by_id: dict[str, SearchResult] = {}
    # Kamus pencatat skor RRF kumulatif
    scores: dict[str, float] = {}

    # Iterasi setiap variasi teks kueri
    for variant in query_variants:
        # Mengonversi kueri variasi menjadi vektor embedding
        query_embedding = embedding_provider.embed_query(variant)
        # Melakukan pencarian vektor kemiripan di database Qdrant
        results = qdrant_store.similarity_search(
            query_embedding=query_embedding,
            # Menggunakan batas candidate K dari settings
            top_k=settings.retrieval_candidate_k,
            where=where,
        )
        # Menghitung peringkat skor RRF di setiap hasil pencarian kueri variasi
        for rank, result in enumerate(results, start=1):
            # Daftarkan objek dokumen jika belum ada di dictionary
            by_id.setdefault(result.id, result)
            # Menghitung skor RRF: skor_lama + 1 / (60 + peringkat)
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (RRF_RANK_CONSTANT + rank)

    # Mengembalikan hasil terurut berdasarkan akumulasi skor RRF tertinggi
    return sorted(by_id.values(), key=lambda result: scores.get(result.id, 0.0), reverse=True)


# Membangun daftar variasi teks kueri untuk mengoptimalkan pencarian buku berbahasa Arab
def build_query_variants(query: str) -> list[str]:
    # Memasukkan teks pertanyaan asli pengguna sebagai variasi pertama
    variants = [query.strip()]
    # Konversi teks pertanyaan ke huruf kecil
    lowered = query.lower()
    # Menampung kata kunci padanan bahasa Arab
    arabic_terms: list[str] = []

    # Memeriksa pemicu terjemahan bahasa Arab
    for trigger, hints in ARABIC_QUERY_HINTS:
        # Jika kata pemicu ditemukan di dalam pertanyaan pengguna
        if trigger in lowered:
            # Masukkan kata kunci padanan bahasa Arab ke list
            arabic_terms.extend(hints)

    # Mengambil nomor poin yang diminta (jika ada)
    requested_count = extract_requested_count(query)
    # Jika angka poin terdeteksi
    if requested_count:
        # Masukkan angka desimal dan versi tulisan Arabnya (khusus angka 10) ke daftar kueri
        arabic_terms.extend([str(requested_count), "عشرة" if requested_count == 10 else ""])

    # Menggabungkan kata kunci Arab unik menjadi satu string padat
    compact_terms = " ".join(term for term in dict.fromkeys(arabic_terms) if term)
    # Jika string terjemahan tidak kosong
    if compact_terms:
        # Masukkan sebagai variasi kueri baru
        variants.append(compact_terms)
    # Kasus penanganan kustom jika bertanya adab/cara berteman/tetangga
    if "tetangga" in lowered and "cara" in lowered:
        variants.append("طرق لكسب الجيران")
        variants.append("عشر طرق لكسب الجيران")

    # Mengembalikan daftar kueri variasi unik yang tidak kosong
    return list(dict.fromkeys(variant for variant in variants if variant.strip()))


# Mengekstrak jumlah angka poin informasi yang diminta oleh kueri pertanyaan
def extract_requested_count(query: str) -> Optional[int]:
    import re
    # Menghapus teks di dalam tanda kutip untuk menghindari deteksi nomor hadis/pasal/bab yang dikutip
    query_no_quotes = re.sub(r"['\"].*?['\"]", "", query)
    # Konversi ke huruf kecil
    lowered = query_no_quotes.lower()
    
    # Kata kunci penunjuk permintaan daftar poin informasi
    trigger_keywords = (
        "sebutkan", "sebut", "sebutin", "jelaskan", "jelasin", "berikan", 
        "tuliskan", "papar", "rincikan", "apa saja", "seluruh", "semua", 
        "berapa", "poin", "macam", "jenis", "adab", "syarat", "rukun", "cara",
        "explain", "list", "what are", "all", "types", "conditions", "pillars", "ways", "how many",
        "説明", "教えて", "リスト", "すべて", "種類", "条件", "方法", "いくつ", "何"
    )
    
    # Jika pertanyaan tidak mengandung satu pun kata kunci pemicu daftar
    if not any(trigger in lowered for trigger in trigger_keywords):
        # Kembalikan None (tidak meminta daftar poin spesifik)
        return None

    # Daftar awalan kata yang harus diabaikan angkanya
    ignore_prefixes = (
        "ke-", "hadis ", "bab ", "ayat ", "pasal ", "halaman ", "surah ", "surat ", "nomor ", "no ",
        "chapter ", "page ", "number ",
        "第"
    )
    # Daftar akhiran kata yang harus diabaikan angkanya
    ignore_suffixes = (
        "st", "nd", "rd", "th",
        "目", "番", "章", "ページ"
    )

    # Fungsi pembantu untuk memverifikasi apakah angka yang ditemukan berada di dekat kata abaikan
    def is_ignored(start_idx: int, end_idx: int) -> bool:
        # Mengambil 10 karakter sebelum angka
        context_before = lowered[max(0, start_idx - 10):start_idx]
        # Mengambil 10 karakter setelah angka
        context_after = lowered[end_idx:min(len(lowered), end_idx + 10)]
        # Jika teks sebelum angka diakhiri dengan prefix abaikan
        if any(context_before.endswith(prefix) for prefix in ignore_prefixes):
            # Tandai untuk diabaikan
            return True
        # Bersihkan spasi kosong di depan teks sesudah angka
        context_after_stripped = context_after.lstrip()
        # Jika teks sesudah angka diawali dengan suffix abaikan
        if any(context_after_stripped.startswith(suffix) for suffix in ignore_suffixes):
            # Tandai untuk diabaikan
            return True
        # Tidak diabaikan
        return False

    # Mencari pola angka desimal 1-2 digit di dalam teks
    for match in re.finditer(r"(?<!ke-)\b([0-9]{1,2})\b", lowered):
        # Jika angka tersebut tidak diabaikan
        if not is_ignored(match.start(), match.end()):
            # Langsung kembalikan nilai integer angka tersebut
            return int(match.group(1))

    # Mencari kata penunjuk angka dari kamus NUMBER_WORDS
    for word, number in NUMBER_WORDS.items():
        # Menentukan apakah kata merupakan huruf Kanji Jepang
        is_kanji = word in ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
        # Menyusun regex pencocokan kata (menggunakan batas kata \b jika alfabet)
        pattern = rf"{word}" if is_kanji else rf"\b{word}\b"
        
        # Mencari kecocokan kata angka di teks kueri
        for match in re.finditer(pattern, lowered):
            # Abaikan kata sandang 'ke' jika terdeteksi
            if not is_kanji and word == "ke":
                continue
            # Jika kata angka tersebut tidak berdekatan dengan kata abaikan
            if not is_ignored(match.start(), match.end()):
                # Mengembalikan nilai integer angka
                return number

    # Jika tidak ada angka yang terdeteksi
    return None


# Menarik chunk tetangga sebelum (prev_id) dan sesudah (next_id) dari Qdrant
def expand_with_neighbors(
    # Daftar dokumen awal
    results: list[SearchResult],
    *,
    # Client Qdrant
    qdrant_store: QdrantStore,
    # Jumlah tingkat perluasan (window size)
    window: int,
    # Daftar ID dokumen yang diblacklist
    rejected_ids: set[str] = frozenset(),
) -> list[SearchResult]:
    # Jika window bernilai 0 atau list hasil kosong, langsung kembalikan duplikasi hasil bersih
    if window <= 0 or not results:
        return dedupe_results(results)

    # Inisialisasi kamus penampung hasil gabungan
    by_id = {result.id: result for result in results}
    # Daftar antrean chunk terluar yang akan dicari tetangganya
    frontier = list(results)
    # Lakukan loop ekspansi sebanyak ukuran window
    for _ in range(window):
        # List penampung ID tetangga yang akan ditarik
        neighbor_ids: list[str] = []
        # Iterasi setiap dokumen di antrean terluar
        for result in frontier:
            # Ambil prev_id (sebelum) dan next_id (sesudah) dari metadata dokumen
            for key in ("prev_id", "next_id"):
                neighbor_id = str(result.metadata.get(key) or "")
                # Jika ID tetangga ada, belum tersimpan di kamus, dan tidak diblacklist
                if neighbor_id and neighbor_id not in by_id and neighbor_id not in rejected_ids:
                    # Masukkan ke list penampung ID tetangga
                    neighbor_ids.append(neighbor_id)

        # Menarik data detail seluruh ID tetangga secara pararel dari Qdrant
        fetched = qdrant_store.get_by_ids(list(dict.fromkeys(neighbor_ids)))
        # Reset antrean terluar
        frontier = []
        # Iterasi hasil penarikan data tetangga
        for result in fetched:
            # Jika ID tetangga belum pernah disimpan dan tidak diblacklist
            if result.id not in by_id and str(result.id) not in rejected_ids:
                # Simpan ke kamus hasil gabungan
                by_id[result.id] = result
                # Masukkan ke antrean terluar untuk diiterasi di loop window berikutnya
                frontier.append(result)

    # Mengembalikan daftar hasil gabungan utuh
    return list(by_id.values())


# Memindai database Qdrant untuk melengkapi nomor heading poin informasi yang diminta pertanyaan (RAG completeness)
def find_completeness_chunks(
    *,
    # Jumlah poin yang diminta
    requested_count: int,
    # ID buku filter
    book_filter: Optional[str],
    # Sidik jari model
    embedding_fingerprint: str,
    # Dokumen hasil pencarian awal
    seed_results: list[SearchResult],
    # Client Qdrant
    qdrant_store: QdrantStore,
    # ID diblacklist
    rejected_ids: set[str] = frozenset(),
) -> list[SearchResult]:
    # Mendapatkan ID buku: gunakan filter buku jika ada, atau tebak dari hasil pencarian awal
    book_ids = [book_filter] if book_filter else _seed_book_ids(seed_results)
    # List penampung hasil pelengkap
    completeness: list[SearchResult] = []

    # Iterasi setiap ID buku target
    for book_id in book_ids:
        # Membangun filter
        where = build_vector_filter(book_id=book_id, embedding_fingerprint=embedding_fingerprint)
        # Menarik seluruh potongan chunk buku tersebut dari Qdrant (scroll data)
        chunks = qdrant_store.get_chunks(where=where)
        # Menyaring chunk yang memiliki nomor urut heading yang valid antara 1 hingga requested_count
        numbered = [
            chunk
            for chunk in chunks
            # Memastikan nomor heading chunk valid dan tidak diblacklist
            if _heading_number(chunk) is not None and 1 <= _heading_number(chunk) <= requested_count and str(chunk.id) not in rejected_ids
        ]
        # Membuat set berisi daftar nomor heading yang berhasil ditemukan
        found_numbers = {_heading_number(chunk) for chunk in numbered}
        # Jika jumlah nomor heading unik yang terkumpul memadai untuk melengkapi pencarian
        if len(found_numbers) >= min(requested_count, len(numbered)):
            # Masukkan seluruh chunk pelengkap tersebut ke list completeness
            completeness.extend(numbered)

    # Mengembalikan daftar chunk pelengkap unik tanpa duplikasi
    return dedupe_results(completeness)


# Mengemas dan mengurutkan hasil akhir secara fisik berdasarkan struktur buku asli
def repack_results(results: list[SearchResult]) -> list[SearchResult]:
    return sorted(
        # Bersihkan duplikat terlebih dahulu
        dedupe_results(results),
        # Diurutkan berdasarkan: 1. ID Buku, 2. Nomor Bab, 3. Indeks Chunk Bab
        key=lambda result: (
            str(result.metadata.get("book_id") or ""),
            int(result.metadata.get("chapter") or 0),
            int(result.metadata.get("chunk_index") or 0),
        ),
    )


# Membuang elemen dokumen duplikat dari list SearchResult berdasarkan keunikan ID chunk-nya
def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    # Dictionary penampung ter-deduplikasi
    deduped: dict[str, SearchResult] = {}
    for result in results:
        # setdefault akan mempertahankan objek pertama yang masuk dan mengabaikan ID yang kembar
        deduped.setdefault(result.id, result)
    # Mengembalikan list values dari dictionary
    return list(deduped.values())


# Membangun kamus filter kueri berdasarkan ID buku dan sidik jari model embedding
def build_vector_filter(book_id: Optional[str], embedding_fingerprint: str) -> dict[str, Any]:
    # Menyusun kriteria filter sidik jari model embedding
    filters: list[dict[str, Any]] = [{"embedding_fingerprint": embedding_fingerprint}]
    # Jika penyaringan ID buku ditentukan
    if book_id:
        # Masukkan ke kriteria filter
        filters.append({"book_id": book_id})
    # Jika kriteria filter hanya satu, langsung kembalikan dictionary tersebut
    if len(filters) == 1:
        return filters[0]
    # Gabungkan kriteria filter menggunakan operator logika '$and'
    return {"$and": filters}


# Mengambil daftar ID buku yang unik dari sekumpulan data SearchResult
def _seed_book_ids(results: list[SearchResult]) -> list[str]:
    # Mengekstrak book_id dari metadata tiap hasil
    book_ids = [str(result.metadata.get("book_id") or "") for result in results]
    # Mengembalikan daftar ID buku unik yang tidak kosong
    return [book_id for book_id in dict.fromkeys(book_ids) if book_id]


# Mengambil nilai angka integer dari kolom heading_number di metadata chunk
def _heading_number(result: SearchResult) -> Optional[int]:
    # Membaca nilai heading_number
    raw = result.metadata.get("heading_number")
    try:
        # Mengonversi nilai mentah menjadi integer
        value = int(raw)
    # Jika gagal konversi (nilai kosong/bukan angka)
    except (TypeError, ValueError):
        # Kembalikan None
        return None
    # Mengembalikan nilai angka (atau None jika angka bernilai 0)
    return value or None
