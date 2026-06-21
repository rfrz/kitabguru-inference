# Mengaktifkan evaluasi tipe data bertunda (postponed evaluation of annotations) agar kompatibel ke belakang
from __future__ import annotations

# Mengimpor modul regex (regular expression) untuk ekstraksi tanda sitasi
import re
# Mengimpor Optional untuk tipe data parameter opsional (boleh None)
from typing import Optional

# Mengimpor modul FastAPI untuk router, dependensi, exception HTTP, dan status kode
from fastapi import APIRouter, Depends, HTTPException, status
# Mengimpor Session (koneksi database) dan fungsi select untuk kueri SQLModel
from sqlmodel import Session, select

# Mengimpor kelas Settings untuk konfigurasi aplikasi
from app.config import Settings
# Mengimpor fungsi pembuka sesi database asinkron
from app.database import get_session
# Mengimpor fungsi penyedia dependensi aplikasi (settings, qdrant, embeddings, llm router, evaluator)
from app.dependencies import get_app_settings, get_qdrant_store, get_embedding_provider, get_llm_router, get_evaluator_llm_router
# Mengimpor model tabel Document
from app.models import Document
# Mengimpor provider embedding dan fungsi profil pembuat sidik jari model embedding
from app.providers.embeddings import EmbeddingProvider, build_embedding_profile
# Mengimpor tipe data profil embedding
from app.providers.embeddings import EmbeddingProfile
# Mengimpor exception kegagalan LLM dan router utama pemanggil LLM
from app.providers.llm import AllLLMProvidersFailed, LLMRouter
# Mengimpor skema data request, response, dan detail sumber kutipan
from app.schemas import ChatRequest, ChatResponse, Source
# Mengimpor modul penyimpanan pencarian vektor Qdrant
from app.services.qdrant_store import QdrantStore
# Mengimpor fungsi pembantu pencarian RAG, ekspansi tetangga (windowing), dan pembersihan duplikat chunk
from app.services.retrieval import retrieve_context, expand_with_neighbors, dedupe_results

# Inisialisasi router FastAPI untuk rute '/chat' di bawah tag dokumentasi 'chat'
router = APIRouter(prefix="/chat", tags=["chat"])

# Template teks prompt utama yang dikirim ke LLM asisten tanya jawab RAG
PROMPT_TEMPLATE = """Kamu adalah asisten AI untuk tanya jawab kitab.
CRITICAL INSTRUCTION: You MUST reply in the EXACT SAME LANGUAGE as the user's query below. If the user's query is in English, reply in English. If Japanese, reply in Japanese. If Arabic, reply in Arabic. This is mandatory.
Jawab hanya berdasarkan konteks dokumen berikut.

Konteks:
{context}

Pertanyaan:
{query}

Aturan jawaban:
- Setiap klaim faktual harus didukung sumber dengan format [S1], [S2], dan seterusnya.
- Jika pertanyaan meminta daftar berjumlah tertentu, berikan setiap poin yang terbukti dari konteks.
- Jika konteks hanya membuktikan sebagian poin, jawab parsial dan sebutkan bahwa sisanya belum terbukti dari konteks yang diberikan.
- Jangan menyatakan bahwa dokumen asli tidak memuat daftar lengkap kecuali semua sumber yang diberikan memang membuktikan hal itu.
- Jangan menambahkan poin dari pengetahuan umum atau hafalan di luar konteks.
"""

# Template teks prompt untuk model evaluator guna memeriksa kelengkapan jumlah poin jawaban
EVAL_PROMPT_TEMPLATE = """Kamu adalah evaluator sistem RAG.
Baca konteks berikut dan tentukan apakah konteks tersebut sudah memuat cukup poin untuk menjawab pertanyaan.
Jawab HANYA dengan objek JSON valid, tanpa markdown tambahan.
Format wajib: {{"is_complete": boolean, "found_count": integer}}

Konteks:
{context}

Pertanyaan (butuh {requested_count} poin):
{query}
"""


# Rute POST '/chat' untuk melayani pencarian RAG dan tanya jawab buku
@router.post("", response_model=ChatResponse)
# Fungsi asinkron/sinkron untuk melayani request chat RAG
def chat(
    # Body request chat berisi pertanyaan dan opsional filter buku
    request: ChatRequest,
    # Mengambil dependensi sesi database
    session: Session = Depends(get_session),
    # Mengambil konfigurasi aplikasi
    settings: Settings = Depends(get_app_settings),
    # Mengambil provider embedding aktif
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    # Mengambil client Qdrant vector store aktif
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
    # Mengambil router LLM utama untuk penjawab
    llm_router: LLMRouter = Depends(get_llm_router),
    # Mengambil router LLM evaluator untuk memeriksa kecukupan informasi
    evaluator_llm_router: LLMRouter = Depends(get_evaluator_llm_router),
) -> ChatResponse:
    # Membangun profil model embedding yang sedang aktif berdasarkan konfigurasi saat ini
    current_profile = build_embedding_profile(settings)
    # Memastikan dokumen yang di-query memiliki sidik jari embedding yang cocok (tidak kedaluwarsa)
    _raise_if_stale_embeddings(session, request.book_filter, current_profile)

    # Mengeksekusi pencarian vektor awal ke Qdrant dan mendeteksi apakah pertanyaan meminta daftar poin tertentu
    retrieval = retrieve_context(
        # Pertanyaan pengguna
        query=request.query,
        # Filter ID buku (jika ada)
        book_filter=request.book_filter,
        # Sidik jari model embedding saat ini
        embedding_fingerprint=current_profile.fingerprint,
        # Konfigurasi aplikasi
        settings=settings,
        # Provider embedding
        embedding_provider=embedding_provider,
        # Client penyimpanan Qdrant
        qdrant_store=qdrant_store,
        # Router LLM evaluator
        evaluator_llm_router=evaluator_llm_router,
    )
    # Menyimpan daftar hasil chunk pencarian awal
    results = retrieval.results

    # Evaluasi Mandiri (Self-Correction Loop RAG): Jika pertanyaan terdeteksi meminta poin berjumlah tertentu
    if retrieval.requested_count and retrieval.requested_count > 0:
        # Melakukan iterasi berulang hingga batas maksimal retry evaluasi yang ditentukan di settings
        for _ in range(settings.rag_max_eval_retries):
            # Jika tidak ada chunk hasil pencarian yang ditemukan, hentikan loop
            if not results:
                break
            
            # Memformat chunk hasil pencarian saat ini menjadi teks konteks [S1], [S2]...
            context = _format_sources_for_prompt(results)
            # Menyusun prompt evaluasi kelayakan informasi
            eval_prompt = EVAL_PROMPT_TEMPLATE.format(
                context=context,
                requested_count=retrieval.requested_count,
                query=request.query
            )
            
            # Meminta model evaluator memberikan penilaian kelengkapan dalam bentuk JSON
            try:
                eval_result = evaluator_llm_router.generate_json(eval_prompt)
            # Jika panggilan model evaluator gagal, segera hentikan loop evaluasi mandiri
            except Exception:
                break
            
            # Mengambil nilai boolean apakah informasi sudah lengkap dari respons JSON
            is_complete = eval_result.get("is_complete", False)
            # Mengambil jumlah poin informasi yang berhasil ditemukan dari dokumen saat ini
            found_count = eval_result.get("found_count", 0)
            
            # Jika informasi dinilai lengkap atau jumlah poin yang ditemukan memenuhi syarat yang diminta
            if is_complete or found_count >= retrieval.requested_count:
                # Segera keluar dari loop karena informasi dinilai sudah memadai
                break
                
            # Ekspansi Konteks (Context Expansion): Jika informasi belum lengkap, ambil chunk tetangga (sebelum/sesudah) dari Qdrant
            new_results = expand_with_neighbors(results, qdrant_store=qdrant_store, window=settings.retrieval_neighbor_window)
            # Menggabungkan hasil lama dengan chunk tetangga baru dan membuang yang terduplikasi
            merged = dedupe_results(results + new_results)
            # Jika setelah digabung jumlah dokumen tidak bertambah, hentikan loop untuk mencegah perulangan tak berujung
            if len(merged) <= len(results):
                break
            # Perbarui variabel results dengan hasil ekspansi terbaru untuk dinilai kembali pada iterasi berikutnya
            results = merged

    # Memetakan hasil akhir chunk pencarian menjadi objek Source untuk respons API
    sources = [
        Source(
            id=result.id,
            document=result.document,
            metadata=result.metadata,
            distance=result.distance,
        )
        # Iterasi setiap hasil akhir chunk
        for result in results
    ]
    # Jika tidak ada dokumen relevan sama sekali yang ditemukan di database vektor
    if not results:
        # Kembalikan segera jawaban default bahwa konteks tidak ditemukan
        return ChatResponse(
            answer="Saya tidak menemukan konteks dokumen yang relevan untuk menjawab pertanyaan tersebut.",
            provider_used="none",
            sources=[],
            answer_status="insufficient",
            retrieval_summary=retrieval.summary,
            citations=[],
        )

    # Memformat chunk hasil akhir menjadi teks konteks yang rapi untuk prompt LLM penjawab
    context = _format_sources_for_prompt(results)
    # Menyusun prompt final tanya jawab RAG
    prompt = PROMPT_TEMPLATE.format(context=context, query=request.query)
    try:
        # Mengirim prompt ke router LLM penjawab untuk menghasilkan teks tanggapan
        generation = llm_router.generate(prompt)
    # Menangkap error jika seluruh provider LLM yang terdaftar gagal merespon
    except AllLLMProvidersFailed as exc:
        # Melemparkan exception HTTP 500 berisi daftar detail kegagalan provider LLM
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Semua provider LLM gagal menghasilkan jawaban.",
                "failures": [failure.__dict__ for failure in exc.failures],
            },
        ) from exc

    # Mengekstrak daftar sitasi sumber yang valid dan benar-benar dirujuk di teks jawaban (misalnya: S1, S2)
    citations = _extract_valid_citations(generation.answer, source_count=len(results))
    # Menentukan klasifikasi kelengkapan status jawaban (complete / partial / insufficient)
    answer_status = _answer_status(
        answer=generation.answer,
        citations=citations,
        valid_citation_marker_count=_count_valid_citation_markers(generation.answer, source_count=len(results)),
        requested_count=retrieval.requested_count,
    )
    # Menambahkan catatan kalimat penjelas di depan jawaban jika statusnya dinilai parsial
    answer = _ensure_partial_notice(
        generation.answer,
        answer_status=answer_status,
        requested_count=retrieval.requested_count,
    )

    # Mengembalikan objek ChatResponse final
    return ChatResponse(
        answer=answer,
        provider_used=generation.provider_used,
        sources=sources,
        answer_status=answer_status,
        retrieval_summary=retrieval.summary,
        citations=citations,
    )


# Fungsi pembantu untuk memvalidasi sidik jari model embedding aktif dengan sidik jari dokumen di database
def _raise_if_stale_embeddings(
    # Sesi database
    session: Session,
    # Filter buku opsional
    book_filter: Optional[str],
    # Profil model embedding aktif saat ini
    current_profile: EmbeddingProfile,
) -> None:
    # Jika query menyaring ID buku spesifik
    if book_filter:
        # Mengambil baris data Document berdasarkan ID buku
        document = session.get(Document, book_filter)
        # Jika dokumen ID buku tersebut tidak ditemukan di database
        if document is None:
            # Melemparkan HTTP Exception 404
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {book_filter} not found",
            )
        # Jika sidik jari model embedding dokumen berbeda dengan model aktif saat ini
        if document.embedding_fingerprint != current_profile.fingerprint:
            # Segera lemparkan exception konflik embedding kedaluwarsa (stale)
            raise _stale_embedding_error([document], current_profile)
        # Selesai validasi dokumen tunggal
        return

    # Jika query mencari di seluruh buku, kueri semua Document yang memiliki sidik jari tidak cocok
    stale_documents = session.exec(
        select(Document).where(Document.embedding_fingerprint != current_profile.fingerprint)
    ).all()
    # Jika ditemukan ada dokumen yang sidik jarinya kedaluwarsa
    if stale_documents:
        # Melemparkan exception konflik embedding kedaluwarsa
        raise _stale_embedding_error(stale_documents, current_profile)


# Fungsi pembantu untuk membungkus data error konflik embedding kedaluwarsa menjadi HTTP Exception
def _stale_embedding_error(documents: list[Document], current_profile: EmbeddingProfile) -> HTTPException:
    # Mengambil dokumen pertama yang terkena dampak sebagai contoh detail
    first = documents[0]
    # Mengembalikan objek HTTPException 409 Conflict berisi informasi detil perbandingan model lama vs baru
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "message": (
                "Embedding dokumen lama tidak cocok dengan konfigurasi embedding saat ini. "
                "Hapus dokumen tersebut lalu import ulang EPUB agar diproses dengan model embedding baru."
            ),
            # Jumlah buku yang terkena dampak ketidakcocokan model
            "affected_count": len(documents),
            # Contoh ID buku pertama yang tidak cocok
            "book_id": first.book_id,
            # Struktur informasi model embedding yang tersimpan di database saat ini
            "stored_embedding": {
                "provider": first.embedding_provider,
                "model": first.embedding_model,
                "dimension": first.embedding_dimension,
                "fingerprint": first.embedding_fingerprint,
            },
            # Struktur informasi model embedding baru yang dikonfigurasikan di settings backend
            "current_embedding": {
                "provider": current_profile.provider,
                "model": current_profile.model,
                "dimension": current_profile.dimension,
                "fingerprint": current_profile.fingerprint,
            },
        },
    )


# Mengubah dan memformat daftar chunk hasil pencarian menjadi teks string berseri untuk prompt LLM
def _format_sources_for_prompt(results) -> str:
    # Inisialisasi list kosong untuk menampung teks terformat tiap chunk
    formatted = []
    # Iterasi setiap hasil chunk pencarian
    for index, result in enumerate(results, start=1):
        # Mengambil metadata chunk
        metadata = result.metadata
        # Mengambil judul sub-bab atau judul halaman (default '-' jika kosong)
        heading = metadata.get("heading") or "-"
        # Memformat metadata dan teks isi dokumen menjadi struktur standar
        formatted.append(
            "\n".join(
                [
                    # Penanda indeks sumber (misal: [S1], [S2])
                    f"[S{index}]",
                    # ID unik chunk vektor
                    f"chunk_id: {result.id}",
                    # ID Buku
                    f"book_id: {metadata.get('book_id', '-')}",
                    # Judul Buku
                    f"title: {metadata.get('title', '-')}",
                    # Nama Chapter/Bab
                    f"chapter: {metadata.get('chapter', '-')}",
                    # Indeks urutan chunk dalam buku
                    f"chunk_index: {metadata.get('chunk_index', '-')}",
                    # Judul sub-bab
                    f"heading: {heading}",
                    # Label isi teks
                    "text:",
                    # Konten isi teks kutipan dokumen
                    result.document,
                ]
            )
        )
    # Menggabungkan seluruh teks chunk terformat dipisahkan pembatas '---'
    return "\n\n---\n\n".join(formatted)


# Mengekstrak daftar sitasi yang valid dari teks jawaban asisten (contoh: teks '[S1] dan [S3]' menghasilkan ['S1', 'S3'])
def _extract_valid_citations(answer: str, *, source_count: int) -> list[str]:
    # Inisialisasi list penampung sitasi
    citations = []
    # Mencari pola sitasi seperti [S1], [S2] menggunakan pencarian regular expression (regex)
    for raw in re.findall(r"\[S([0-9]+)\]", answer):
        # Mengubah string angka menjadi tipe data integer
        index = int(raw)
        # Memastikan nomor indeks sitasi berada dalam rentang jumlah chunk sumber yang dikirimkan ke LLM
        if 1 <= index <= source_count:
            # Memasukkan kode sitasi terformat ke list
            citations.append(f"S{index}")
    # Mengembalikan list sitasi unik (menghilangkan duplikasi) dengan mempertahankan urutan aslinya
    return list(dict.fromkeys(citations))


# Fungsi pembantu untuk menentukan klasifikasi kualitas kelengkapan jawaban RAG
def _answer_status(
    *,
    # Teks jawaban asisten
    answer: str,
    # Daftar kode sitasi unik yang valid
    citations: list[str],
    # Jumlah total tanda penanda sitasi di teks jawaban
    valid_citation_marker_count: int,
    # Jumlah poin informasi yang diminta oleh pertanyaan (jika ada)
    requested_count: Optional[int],
) -> str:
    # Mengubah teks jawaban ke huruf kecil semua agar pencocokan teks konsisten
    lowered = answer.lower()
    # Jika jawaban mengandung kata penolakan konteks dan tidak menyertakan rujukan sitasi sumber
    if ("tidak menemukan konteks" in lowered or "tidak cukup" in lowered) and not citations:
        # Klasifikasikan jawaban sebagai tidak mencukupi (insufficient)
        return "insufficient"

    # Jika pertanyaan meminta daftar dengan jumlah poin tertentu
    if requested_count:
        # Menghitung jumlah poin bullet-points yang berhasil ditulis di dalam teks jawaban
        listed_points = _count_listed_points(answer)
        # Jika jumlah poin yang ditulis memenuhi syarat dan jumlah tanda sitasi valid juga memenuhi syarat
        if listed_points >= requested_count and valid_citation_marker_count >= requested_count:
            # Klasifikasikan jawaban sebagai lengkap (complete)
            return "complete"
        # Jika minimal ada rujukan sitasi atau ada poin yang sempat ditulis
        if citations or listed_points:
            # Klasifikasikan jawaban sebagai parsial (terjawab sebagian)
            return "partial"
        # Jika tidak ada sitasi dan tidak ada poin yang berhasil ditulis
        return "insufficient"

    # Jika tidak meminta poin spesifik, kembalikan 'complete' jika ada sitasi, atau 'partial' jika tidak ada sitasi
    return "complete" if citations else "partial"


# Menghitung jumlah baris poin daftar (bullet list) seperti "1. ", "2) ", "- ", atau "* " di dalam teks jawaban
def _count_listed_points(answer: str) -> int:
    # Menggunakan regex mode multiline untuk mencocokkan format bullet point di awal baris baru
    return len(re.findall(r"(?m)^\s*(?:[0-9]{1,2}[\).]|[-*])\s+", answer))


# Menghitung jumlah penanda sitasi valid yang ditulis di dalam teks jawaban (memperbolehkan rujukan berulang)
def _count_valid_citation_markers(answer: str, *, source_count: int) -> int:
    # Inisialisasi hitungan
    count = 0
    # Mencari seluruh tanda sitasi angka di teks jawaban
    for raw in re.findall(r"\[S([0-9]+)\]", answer):
        # Konversi ke integer
        index = int(raw)
        # Jika indeks sitasi valid (ada di dalam daftar sumber)
        if 1 <= index <= source_count:
            # Tambahkan hitungan
            count += 1
    # Mengembalikan total jumlah tanda sitasi
    return count


# Memastikan jawaban asisten diawali dengan teks informasi parsial jika status jawabannya adalah parsial
def _ensure_partial_notice(answer: str, *, answer_status: str, requested_count: Optional[int]) -> str:
    # Jika status jawaban bukan parsial atau di dalam teks jawaban sudah memuat penjelasan parsial
    if answer_status != "partial" or "parsial" in answer.lower():
        # Kembalikan teks jawaban apa adanya
        return answer
    # Jika ada permintaan jumlah poin spesifik
    if requested_count:
        # Sisipkan kalimat penjelas parsial yang merinci target requested_count di depan teks jawaban asli
        return (
            "Jawaban parsial: konteks yang ditemukan belum cukup untuk memverifikasi "
            f"seluruh {requested_count} poin yang diminta.\n\n{answer}"
        )
    # Sisipkan kalimat penjelas parsial umum di depan teks jawaban asli
    return f"Jawaban parsial berdasarkan konteks yang ditemukan.\n\n{answer}"
