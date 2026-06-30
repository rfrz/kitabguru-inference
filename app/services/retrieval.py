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
    # Jumlah kandidat dokumen awal yang berhasil ditarik dari Qdrant
    candidate_count: int

    # Properti untuk mengembalikan ringkasan statistik pencarian RAG
    @property
    def summary(self) -> dict[str, Any]:
        return {
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
    # Membangun filter kueri berdasarkan ID buku dan sidik jari model embedding
    where = build_vector_filter(book_id=book_filter, embedding_fingerprint=embedding_fingerprint)
    
    # 1. Pencarian vektor tunggal (langsung menggunakan query asli, tanpa RRF)
    query_embedding = embedding_provider.embed_query(query.strip())
    candidates = qdrant_store.similarity_search(
        query_embedding=query_embedding,
        top_k=settings.retrieval_candidate_k,
        where=where,
    )

    # Inisialisasi set penampung ID chunk yang ditolak oleh model evaluator
    rejected_chunk_ids: set[str] = set()
    
    # 2. Evaluator LLM Filter (Untuk kandidat awal)
    if evaluator_llm_router and candidates:
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
                for uid in useless_ids:
                    rejected_chunk_ids.add(str(uid))
                # Bersihkan dokumen yang ditolak dari daftar kandidat
                candidates = [c for c in candidates if str(c.id) not in rejected_chunk_ids]
        # Abaikan error penyaringan jika model gagal merespon
        except Exception:
            pass

    # 3. Ekspansi Konteks (Neighbor Expansion) dengan Evaluator LLM di setiap iterasi
    expanded = expand_with_neighbors(
        candidates,
        qdrant_store=qdrant_store,
        window=settings.retrieval_neighbor_window,
        rejected_ids=rejected_chunk_ids,
        evaluator_llm_router=evaluator_llm_router,
        query=query,
    )

    # Menggabungkan seluruh hasil kandidat awal dengan hasil ekspansi, lalu buang duplikatnya
    merged = dedupe_results([*candidates, *expanded])
    # Mengemas dan mengurutkan hasil akhir secara fisik berdasarkan chapter dan urutan chunk index buku
    final_results = repack_results(merged)

    # Mengembalikan objek detail RetrievedContext
    return RetrievedContext(
        results=final_results,
        candidate_count=len(candidates),
    )


# Menarik chunk tetangga sebelum (prev_id) dan sesudah (next_id) dari Qdrant
def expand_with_neighbors(
    # Daftar dokumen awal
    results: list[SearchResult],
    *,
    # Client Qdrant
    qdrant_store: QdrantStore,
    # Jumlah tingkat perluasan (window size)
    window: int,
    # Daftar ID dokumen yang diblacklist (diperbarui in-place)
    rejected_ids: set[str],
    # Router LLM evaluator
    evaluator_llm_router: Optional[LLMRouter],
    # Pertanyaan asli
    query: str,
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

        if not neighbor_ids:
            break

        # Menarik data detail seluruh ID tetangga secara pararel dari Qdrant
        fetched = qdrant_store.get_by_ids(list(dict.fromkeys(neighbor_ids)))
        
        # --- EVALUATOR LLM FILTER (Per-Batch Iterasi) ---
        if evaluator_llm_router and fetched:
            from app.api.chat import _format_sources_for_prompt
            # Memformat teks kandidat terpilih hanya dari batch tetangga yang baru ditarik
            context = _format_sources_for_prompt(fetched)
            prompt = CHUNK_EVALUATION_PROMPT_TEMPLATE.format(context=context, query=query)
            try:
                eval_result = evaluator_llm_router.generate_json(prompt)
                useless_ids = eval_result.get("useless_chunk_ids", [])
                if useless_ids and isinstance(useless_ids, list):
                    for uid in useless_ids:
                        rejected_ids.add(str(uid))
                    # Bersihkan batch fetched dari chunk yang ditolak
                    fetched = [f for f in fetched if str(f.id) not in rejected_ids]
            except Exception:
                pass
        # ------------------------------------------------

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
