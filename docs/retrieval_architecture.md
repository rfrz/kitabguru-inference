# Arsitektur & Alur Kerja High-Level Retrieval KitabGuru

Dokumen ini mencatat bagaimana sistem *Retrieval-Augmented Generation* (RAG) pada aplikasi KitabGuru bekerja secara *high-level*, mulai dari *query* masuk hingga siap dijawab oleh LLM utama.

## 1. Query Expansion & Variasi (Pre-Retrieval)
Ketika *user* mengirimkan pertanyaan:
- Sistem memanggil `build_query_variants()`.
- Teks di-*scan* untuk mencari kata kunci (seperti "tetangga", "adab"). Jika cocok, sistem menambahkan padanan kata dalam bahasa Arab ke dalam *query variants*.
- Sistem juga mengekstrak angka (misalnya "Sebutkan 10...") menggunakan regex, untuk mengaktifkan *mode enumerasi/completeness*.

## 2. Vector Candidate Search (Pencarian Awal)
- Semua *query variants* diubah menjadi *vector embeddings*.
- Sistem memanggil Qdrant `similarity_search` untuk mengambil `retrieval_candidate_k` (misal: 30) *chunk* terbaik.
- Jika ada banyak variasi *query*, hasilnya di-merger menggunakan algoritma *Reciprocal Rank Fusion* (RRF) agar mendapatkan *ranking* gabungan terbaik.

## 3. Multi-Document Routing Layer (Book Selection)
Untuk mencegah RAG "terjebak" di buku yang salah ketika konteksnya mirip:
- Sistem mengekstrak semua `book_id` unik dari 30 kandidat awal tersebut.
- Jika kandidat berasal dari lebih dari 1 buku (dan *user* tidak memfilter 1 buku secara spesifik di UI), **Evaluator LLM** akan dipanggil secara rahasia.
- Evaluator membaca cuplikan teks kandidat dan menjawab dalam bentuk JSON (e.g. `{"relevant_book_ids": ["buku_B"]}`).
- Kandidat awal kemudian **difilter** hingga hanya menyisakan *chunk* dari buku-buku yang terbukti relevan tersebut.

## 4. Pemotongan Top-K & Neighbor Expansion
- Kandidat yang sudah bersih/terfilter kemudian dipotong mengambil `retrieval_final_k` (misal: 12) *chunk* paling relevan.
- Sistem memanggil `expand_with_neighbors()`. Algoritma ini akan melihat ID `prev_id` dan `next_id` dari setiap *chunk*, lalu menarik paragraf sebelum dan sesudahnya dari *database*.
- Ini memberikan *context-window* alami agar kalimat tidak terpotong di tengah jalan.

## 5. Agentic RAG Loop (Iterative Completeness)
Khusus untuk *query* yang menanyakan jumlah tertentu (contoh: "10 kebaikan"):
- **Evaluator LLM** kembali beraksi! Ia disuruh mengevaluasi apakah teks yang sudah diekspansi benar-benar memuat *10 poin*.
- Jika belum genap (berdasarkan JSON *output* `found_count`), sistem akan masuk ke dalam *loop iterasi* (hingga `rag_max_eval_retries`).
- Di dalam *loop* ini, `expand_with_neighbors` dipanggil lagi secara progresif untuk memperlebar cakupan teks, HANYA PADA buku yang sudah diverifikasi di langkah 3.
- Proses ini berulang hingga LLM menyatakan *"is_complete: true"* atau mencapai batas percobaan.

## 6. Deduplication & Repacking
- Semua *chunk* hasil ekspansi dan iterasi digabung.
- Sistem membuang duplikat (`dedupe_results`).
- Sistem mengurutkan ulang *chunk* tersebut berdasarkan letak aslinya dalam buku (`chapter`, `chunk_index`) agar LLM utama membacanya layaknya membaca buku berurutan (`repack_results`).

## 7. Generasi Jawaban Akhir (LLM Utama)
- Teks yang sudah dirakit sempurna ini dikirim ke **LLM Utama** (misal Gemini Flash Lite).
- LLM merangkai jawaban bahasa Indonesia dan menempelkan *citation markers* (misal `[S1]`, `[S2]`) sesuai referensi yang diberikan.
- Sistem terakhir akan memeriksa *citation* tersebut dan memberikan label `answer_status` (lengkap, parsial, atau *insufficient*).

---
*Catatan: Konfigurasi model Evaluator dapat diatur secara mandiri dan sepenuhnya terpisah dari model Utama via variabel environment (`EVALUATOR_LLM_FALLBACK_ORDER`, dll).*
