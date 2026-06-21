# Mengimpor lru_cache untuk caching konfigurasi agar tidak membaca file env berulang kali
from functools import lru_cache
# Mengimpor Path untuk penanganan folder lokal
from pathlib import Path
# Mengimpor Optional untuk type hinting nilai yang boleh None (kosong)
from typing import Optional

# Mengimpor BaseSettings dan SettingsConfigDict untuk mengelola konfigurasi aplikasi via environment variables
from pydantic_settings import BaseSettings, SettingsConfigDict


# Kelas Settings menampung seluruh konfigurasi mesin inferensi RAG
class Settings(BaseSettings):
    # Konfigurasi Pydantic Settings untuk membaca file .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # ─── Pengaturan Aplikasi (App Settings) ───────────────────────────────
    # Nama aplikasi mesin inferensi RAG KitabGuru
    app_name: str = "KitabGuru Inference Engine"
    # Prefix path routing URL untuk API
    api_prefix: str = "/api"
    # Mengizinkan seluruh domain luar untuk mengakses API (CORS)
    cors_origins: str = "*"

    # ─── Pengaturan database & Mesin RAG (Database & Vector Store) ────────
    # URL database SQLite lokal tempat menyimpan metadata buku
    database_url: str = "sqlite:///./data/app.db"
    # Path folder tempat penyimpanan database vektor Qdrant lokal
    qdrant_location: str = "./data/qdrant"
    # Kunci API Qdrant (opsional, jika menggunakan Qdrant Cloud)
    qdrant_api_key: Optional[str] = None
    # Nama koleksi (tabel vektor) utama di Qdrant
    qdrant_collection: str = "epub_collection"
    # Jumlah kandidat fragmen teks awal (k) yang diambil oleh pencarian vektor (retrieval)
    retrieval_candidate_k: int = 30
    # Jumlah fragmen teks final (k) teratas yang akan dikirim ke LLM sebagai konteks RAG
    retrieval_final_k: int = 12
    # Jumlah fragmen teks tetangga (sibling chunk) kiri-kanan yang ikut diambil untuk memperkaya konteks RAG
    retrieval_neighbor_window: int = 1
    # Mengaktifkan evaluasi otomatis kelengkapan jawaban AI oleh model evaluator RAG
    rag_enable_completeness_scan: bool = True
    # Batas maksimal pengulangan evaluasi RAG jika jawaban dinilai belum lengkap oleh evaluator
    rag_max_eval_retries: int = 3
    # Ukuran kapasitas jumlah karakter per potongan teks buku saat di-ingest (chunk size)
    chunk_size: int = 1200
    # Ukuran tumpang-tindih karakter antar potongan teks agar tidak kehilangan makna (overlap)
    chunk_overlap: int = 160

    # ─── Pengaturan Model Embedding (Vector Embeddings) ───────────────────
    # Nama provider embedding (huggingface / gemini)
    embedding_provider: str = "huggingface"
    # Dimensi vektor embedding (opsional)
    embedding_dimension: Optional[int] = None
    # Kunci API Hugging Face (opsional)
    hf_api_key: Optional[str] = None
    # Nama model embedding Hugging Face default (multilingual E5 large)
    hf_embedding_model: str = "intfloat/multilingual-e5-large"
    # Kunci API Google Gemini untuk embedding
    gemini_api_key: Optional[str] = None
    # Nama model embedding dari Google Gemini
    gemini_embedding_model: str = "text-embedding-004"

    # ─── Pengaturan Model Bahasa (LLM Inference Fallback) ─────────────────
    # Urutan fallback provider LLM utama (jika provider pertama gagal, coba provider berikutnya)
    llm_fallback_order: str = "gemini,groq,openrouter,openai_compatible"
    # Nilai temperatur LLM (0.0 membuat jawaban bernilai faktual/fokus dan minim halusinasi)
    llm_temperature: float = 0.0
    # Nama model Gemini yang digunakan untuk inferensi RAG
    gemini_llm_model: str = "gemini-3.1-flash-lite"
    # Kunci API Groq
    groq_api_key: Optional[str] = None
    # Nama model Llama yang dijalankan di Groq
    groq_llm_model: str = "llama3-70b-8192"
    # Kunci API OpenRouter
    openrouter_api_key: Optional[str] = None
    # Nama model LLM yang dipanggil lewat OpenRouter
    openrouter_llm_model: str = "meta-llama/llama-3-70b-instruct"
    # Kunci API provider alternatif yang kompatibel dengan format OpenAI
    openai_compatible_api_key: Optional[str] = None
    # URL endpoint dasar provider alternatif OpenAI
    openai_compatible_base_url: Optional[str] = None
    # Nama model provider alternatif OpenAI
    openai_compatible_model: Optional[str] = None

    # ─── Pengaturan Model Evaluator RAG (RAG Evaluator Fallback) ──────────
    # Urutan fallback provider LLM khusus untuk tugas mengevaluasi kelengkapan jawaban
    evaluator_llm_fallback_order: str = "gemini,groq,openrouter,openai_compatible"
    # Kunci API Gemini untuk model evaluator
    evaluator_gemini_api_key: Optional[str] = None
    # Nama model Gemini evaluator
    evaluator_gemini_llm_model: Optional[str] = None
    # Kunci API Groq untuk model evaluator
    evaluator_groq_api_key: Optional[str] = None
    # Nama model Groq evaluator
    evaluator_groq_llm_model: Optional[str] = None
    # Kunci API OpenRouter untuk model evaluator
    evaluator_openrouter_api_key: Optional[str] = None
    # Nama model OpenRouter evaluator
    evaluator_openrouter_llm_model: Optional[str] = None
    # Kunci API OpenAI compatible untuk model evaluator
    evaluator_openai_compatible_api_key: Optional[str] = None
    # URL dasar OpenAI compatible untuk model evaluator
    evaluator_openai_compatible_base_url: Optional[str] = None
    # Nama model OpenAI compatible untuk model evaluator
    evaluator_openai_compatible_model: Optional[str] = None

    # Mengembalikan daftar origin CORS dalam bentuk list Python
    @property
    def cors_origin_list(self) -> list[str]:
        # Jika diatur ke '*', izinkan semua domain
        if self.cors_origins.strip() == "*":
            return ["*"]
        # Memisahkan string CORS dengan koma dan membuang spasi kosong
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    # Mengembalikan list urutan provider LLM utama
    @property
    def llm_provider_order(self) -> list[str]:
        return [
            provider.strip().lower()
            for provider in self.llm_fallback_order.split(",")
            if provider.strip()
        ]

    # Mengembalikan list urutan provider LLM evaluator
    @property
    def evaluator_llm_provider_order(self) -> list[str]:
        return [
            provider.strip().lower()
            for provider in self.evaluator_llm_fallback_order.split(",")
            if provider.strip()
        ]

    # Mengembalikan nama provider embedding dalam format huruf kecil bersih
    @property
    def normalized_embedding_provider(self) -> str:
        return self.embedding_provider.strip().lower()

    # Mengembalikan nama model embedding aktif berdasarkan provider pilihan
    @property
    def active_embedding_model(self) -> str:
        provider = self.normalized_embedding_provider
        # Jika menggunakan Hugging Face
        if provider == "huggingface":
            return self.hf_embedding_model
        # Jika menggunakan Google Gemini
        if provider == "gemini":
            return self.gemini_embedding_model
        # Kosong jika tidak dikenali
        return ""

    # Memastikan folder Qdrant lokal dan folder database SQLite lokal telah dibuat di filesystem
    def ensure_local_directories(self) -> None:
        # Jika Qdrant diset ke folder lokal (bukan URL http/https cloud)
        if not self.qdrant_location.startswith(("http://", "https://")):
            # Buat folder penyimpanan Qdrant jika belum ada
            Path(self.qdrant_location).mkdir(parents=True, exist_ok=True)
        # Jika database menggunakan skema SQLite file lokal
        if self.database_url.startswith("sqlite:///"):
            # Mengekstrak path file database
            db_path = self.database_url.removeprefix("sqlite:///")
            # Memastikan database bukan SQLite in-memory
            if db_path and db_path != ":memory:":
                # Membuat folder induk tempat menyimpan database SQLite jika belum dibuat
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)


# Menyimpan instance konfigurasi dalam cache memori server
@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Memastikan folder database dan vektor lokal sudah siap digunakan
    settings.ensure_local_directories()
    return settings
