# Mengaktifkan evaluasi tipe data bertunda (postponed evaluation of annotations) agar kompatibel ke belakang
from __future__ import annotations

# Mengimpor modul hashlib untuk menghitung hash SHA-256 sidik jari profil model
import hashlib
# Mengimpor modul json untuk melakukan encoding konfigurasi profil ke string JSON
import json
# Mengimpor dataclass dan helper replace dari dataclasses untuk representasi struktur data konstan
from dataclasses import dataclass, replace
# Mengimpor tipe pembantu Iterable, Optional, dan Protocol untuk type-hinting antarmuka (interface)
from typing import Iterable, Optional, Protocol

# Mengimpor kelas Settings untuk membaca konfigurasi model embedding
from app.config import Settings


# Versi internal untuk pembuatan sidik jari (fingerprint) model embedding
FINGERPRINT_VERSION = 1


# Kelas Exception khusus untuk menangani konfigurasi provider yang tidak didukung atau salah
class ProviderConfigurationError(RuntimeError):
    """Melemparkan error ketika provider yang dikonfigurasi tidak dapat digunakan."""
    pass


# Struktur data konstan untuk menampung konfigurasi profil model embedding aktif
@dataclass(frozen=True)
class EmbeddingProfile:
    # Nama provider (huggingface / gemini)
    provider: str
    # Nama model spesifik (seperti 'intfloat/multilingual-e5-large')
    model: str
    # Jumlah dimensi vektor (misalnya 768 atau 1024)
    dimension: Optional[int]
    # Penanda perilaku khusus pemrosesan teks
    behavior: str

    # Properti asinkron untuk menghasilkan sidik jari unik hash SHA-256 dari konfigurasi profil
    @property
    def fingerprint(self) -> str:
        # Menyusun dictionary berisi parameter pembeda model
        payload = {
            "version": FINGERPRINT_VERSION,
            "provider": self.provider.strip().lower(),
            "model": self.model.strip(),
            "dimension": self.dimension,
            "behavior": self.behavior,
        }
        # Mengubah dictionary menjadi string JSON dengan kunci terurut secara konsisten
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        # Menghasilkan hash SHA-256 heksadesimal dari string JSON tersebut
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # Membuat duplikat profil dengan memperbarui nilai dimensi vektornya
    def with_dimension(self, dimension: Optional[int]) -> "EmbeddingProfile":
        # Menggunakan helper replace untuk menyalin instans konstan dengan dimensi baru
        return replace(self, dimension=dimension)


# Mendefinisikan antarmuka (Protocol/Interface) wajib untuk class provider embedding
class EmbeddingProvider(Protocol):
    # Properti untuk membaca profil konfigurasi
    @property
    def profile(self) -> EmbeddingProfile:
        ...

    # Metode untuk merubah satu teks pertanyaan (query) menjadi vektor float
    def embed_query(self, text: str) -> list[float]:
        ...

    # Metode untuk merubah daftar teks potongan dokumen (passage) menjadi daftar vektor float
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


# Kamus pemetaan dimensi default untuk model-model embedding populer yang umum dipakai
KNOWN_DIMENSIONS: dict[tuple[str, str], int] = {
    ("huggingface", "intfloat/multilingual-e5-large"): 1024,
    ("huggingface", "intfloat/multilingual-e5-base"): 768,
    ("huggingface", "intfloat/multilingual-e5-small"): 384,
    ("gemini", "text-embedding-004"): 768,
    ("gemini", "models/text-embedding-004"): 768,
    ("gemini", "gemini-embedding-001"): 3072,
    ("gemini", "gemini-embedding-2"): 3072,
}


# Memeriksa apakah nama model yang diset merupakan kelompok model e5 (butuh prefix query/passage)
def is_e5_model(model: str) -> bool:
    # Mengembalikan True jika teks 'e5' terkandung di dalam nama model
    return "e5" in model.lower()


# Menentukan tipe perilaku format teks berdasarkan provider dan model yang diset
def embedding_behavior(provider: str, model: str) -> str:
    # Jika menggunakan provider Hugging Face dan model bertipe e5
    if provider == "huggingface" and is_e5_model(model):
        # Terapkan format penambahan awalan teks 'query: ' atau 'passage: '
        return "e5-query-passage-prefix"
    # Jika menggunakan provider Google Gemini
    if provider == "gemini":
        # Terapkan format pencarian semantik similarity bawaan Gemini
        return "gemini-semantic-similarity"
    # Jika menggunakan provider lokal dengan model e5
    if provider == "local" and is_e5_model(model):
        return "e5-query-passage-prefix"
    # Selain dari itu, gunakan perilaku default normal
    return "default"


# Mencari tahu dimensi vektor berdasarkan pengaturan manual atau kamus default model
def resolve_embedding_dimension(settings: Settings) -> Optional[int]:
    # Jika pengguna menetapkan dimensi secara manual di file env, langsung gunakan nilai tersebut
    if settings.embedding_dimension is not None:
        # Mengembalikan angka dimensi manual
        return settings.embedding_dimension
    # Mengambil nama provider ternormalisasi
    provider = settings.normalized_embedding_provider
    # Mengambil nama model yang aktif
    model = settings.active_embedding_model
    # Mengambil nilai dimensi bawaan dari kamus KNOWN_DIMENSIONS
    return KNOWN_DIMENSIONS.get((provider, model))


# Merakit profil konfigurasi model embedding aktif berdasarkan object settings
def build_embedding_profile(settings: Settings) -> EmbeddingProfile:
    # Mengambil provider ternormalisasi
    provider = settings.normalized_embedding_provider
    # Mengambil model aktif
    model = settings.active_embedding_model
    # Memvalidasi provider yang didukung
    if provider not in {"huggingface", "gemini", "local"}:
        raise ProviderConfigurationError(f"Unsupported embedding provider: {provider}")
    # Memastikan nama model telah diisi
    if not model:
        raise ProviderConfigurationError(f"No embedding model configured for provider: {provider}")
    # Mengembalikan instans profile baru
    return EmbeddingProfile(
        provider=provider,
        model=model,
        dimension=resolve_embedding_dimension(settings),
        behavior=embedding_behavior(provider, model),
    )


# Pabrik instans: membuat objek instans provider embedding yang sesuai dengan setelan di config
def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    # Mengambil provider ternormalisasi
    provider = settings.normalized_embedding_provider
    # Jika setelannya huggingface, buat instans HuggingFaceEmbeddingProvider
    if provider == "huggingface":
        return HuggingFaceEmbeddingProvider(settings)
    # Jika setelannya gemini, buat instans GeminiEmbeddingProvider
    if provider == "gemini":
        return GeminiEmbeddingProvider(settings)
    # Jika setelannya lokal, buat instans LocalEmbeddingProvider
    if provider == "local":
        return LocalEmbeddingProvider(settings)
    # Lempar error jika tidak ada provider yang cocok
    raise ProviderConfigurationError(f"Unsupported embedding provider: {provider}")


# Menyematkan teks awalan 'query: ' atau 'passage: ' jika menggunakan model jenis e5
def prefixed_for_e5(model: str, text: str, *, is_query: bool) -> str:
    # Jika model bukan e5, kembalikan teks aslinya tanpa modifikasi
    if not is_e5_model(model):
        return text
    # Tentukan awalan: 'query: ' untuk pencarian, 'passage: ' untuk dokumen buku
    prefix = "query: " if is_query else "passage: "
    # Sisipkan awalan jika teks belum diawali dengan kata tersebut
    return text if text.lower().startswith(prefix) else f"{prefix}{text}"


# Memaksa tipe respons keluaran API embedding agar menjadi list float satu dimensi
def coerce_vector(value) -> list[float]:
    # Jika objek memiliki metode 'tolist' (seperti array numpy), ubah ke list Python
    if hasattr(value, "tolist"):
        value = value.tolist()
    # Jika berupa tuple, ubah menjadi list
    if isinstance(value, tuple):
        value = list(value)
    # Memastikan hasil akhir berupa tipe list Python
    if not isinstance(value, list):
        raise ValueError("Embedding response is not a vector")
    # Jika berupa list di dalam list (dua dimensi)
    if value and isinstance(value[0], list):
        # Jika isinya hanya satu baris
        if len(value) == 1:
            # Panggil fungsi ini kembali secara rekursif untuk membongkar baris pertama
            return coerce_vector(value[0])
        # Hitung panjang kolom vektor
        width = len(value[0])
        # Jika kosong, kembalikan list kosong
        if width == 0:
            # Kembalikan list kosong
            return []
        # Melakukan rata-rata nilai kolom di seluruh baris jika menerima list multi-dimensi (pooling)
        return [
            float(sum(float(row[index]) for row in value) / len(value))
            for index in range(width)
        ]
    # Memastikan setiap angka di dalam list bertipe float
    return [float(item) for item in value]


# Mengonversi kumpulan iterable respons embedding menjadi daftar vektor float dua dimensi
def coerce_vectors(values: Iterable) -> list[list[float]]:
    return [coerce_vector(value) for value in values]


# Kelas implementasi provider embedding menggunakan API Hugging Face Inference
class HuggingFaceEmbeddingProvider:
    # Inisialisasi token API dan mengimpor pustaka client asinkron huggingface-hub
    def __init__(self, settings: Settings):
        # Memastikan tersedianya token API Hugging Face
        if not settings.hf_api_key:
            raise ProviderConfigurationError("HF_API_KEY is required for Hugging Face embeddings")
        # Menyimpan nama model embedding target
        self.model = settings.hf_embedding_model
        # Membuat profil konfigurasi model
        self._profile = build_embedding_profile(settings)
        try:
            # Mencoba mengimpor InferenceClient dari pustaka resmi huggingface_hub
            from huggingface_hub import InferenceClient
        # Lempar error konfigurasi jika pustaka belum terpasang
        except ImportError as exc:
            raise ProviderConfigurationError("Install huggingface-hub to use Hugging Face embeddings") from exc

        try:
            # Membuat instans InferenceClient asinkron dengan konfigurasi provider hf-inference
            self.client = InferenceClient(provider="hf-inference", api_key=settings.hf_api_key)
        # Menangani fallback tipe parameter instans jika menggunakan versi library lama
        except TypeError:
            self.client = InferenceClient(token=settings.hf_api_key)

    # Membaca profil model embedding
    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    # Mengonversi teks query pertanyaan menjadi vektor dengan menyisipkan awalan e5 query
    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(prefixed_for_e5(self.model, text, is_query=True))

    # Mengonversi daftar dokumen buku menjadi daftar vektor dengan menyisipkan awalan e5 passage
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            self._embed_one(prefixed_for_e5(self.model, text, is_query=False))
            for text in texts
        ]

    # Mengirim request ekstraksi fitur vektor ke API Hugging Face
    def _embed_one(self, text: str) -> list[float]:
        # Memanggil metode feature_extraction secara sinkron ke server Hugging Face
        result = self.client.feature_extraction(text, model=self.model)
        # Memaksa format hasil ekstraksi menjadi list float satu dimensi
        return coerce_vector(result)


# Kelas implementasi provider embedding menggunakan API Google Gemini GenAI SDK
class GeminiEmbeddingProvider:
    # Inisialisasi kunci API dan mengimpor pustaka google-genai
    def __init__(self, settings: Settings):
        # Memastikan tersedianya kunci API Google Gemini
        if not settings.gemini_api_key:
            raise ProviderConfigurationError("GEMINI_API_KEY is required for Gemini embeddings")
        # Menyimpan model embedding yang digunakan
        self.model = settings.gemini_embedding_model
        # Menyimpan ukuran dimensi vektor
        self.dimension = settings.embedding_dimension
        # Membuat profil konfigurasi model
        self._profile = build_embedding_profile(settings)
        try:
            # Mencoba mengimpor modul genai dan types dari SDK google
            from google import genai
            from google.genai import types
        # Lempar error jika library google-genai belum dipasang
        except ImportError as exc:
            raise ProviderConfigurationError("Install google-genai to use Gemini embeddings") from exc

        # Membuat instans Client SDK Google GenAI
        self.client = genai.Client(api_key=settings.gemini_api_key)
        # Menyimpan rujukan types untuk konfigurasi API
        self.types = types

    # Membaca profil model embedding
    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    # Mengonversi satu teks query menjadi vektor via pemanggilan API Gemini
    def embed_query(self, text: str) -> list[float]:
        return self._embed_many([text])[0]

    # Mengonversi daftar dokumen menjadi daftar vektor via pemanggilan API Gemini
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_many(texts)

    # Mengirimkan request konversi daftar teks menjadi vektor ke Google Gemini API
    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        # Inisialisasi argumen konfigurasi tambahan
        config_kwargs = {}
        # Jika dimensi diset secara spesifik, masukkan ke parameter output_dimensionality
        if self.dimension:
            config_kwargs["output_dimensionality"] = self.dimension
        # Jika menggunakan model text-embedding bawaan Gemini, set tipe tugas ke SEMANTIC_SIMILARITY
        if self.model in {"text-embedding-004", "models/text-embedding-004", "gemini-embedding-001"}:
            config_kwargs["task_type"] = "SEMANTIC_SIMILARITY"
        # Membungkus opsi menjadi objek EmbedContentConfig
        config = self.types.EmbedContentConfig(**config_kwargs) if config_kwargs else None
        # Memanggil endpoint embed_content dari SDK Google GenAI secara sinkron
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=config,
        )
        # Memaksa list hasil embedding menjadi daftar vektor float dua dimensi
        return coerce_vectors([embedding.values for embedding in result.embeddings])


# Kelas implementasi provider embedding yang berjalan secara lokal dengan fastembed
class LocalEmbeddingProvider:
    # Inisialisasi provider lokal
    def __init__(self, settings: Settings):
        # Menyimpan model embedding yang digunakan
        self.model = settings.local_embedding_model
        # Membuat profil konfigurasi model
        self._profile = build_embedding_profile(settings)
        try:
            # Mengimpor fastembed
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ProviderConfigurationError("Install fastembed to use local embeddings") from exc

        # Inisialisasi model lokal (akan otomatis mengunduh jika belum ada)
        self.client = TextEmbedding(model_name=self.model)

    # Membaca profil model embedding
    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    # Mengonversi teks query pertanyaan menjadi vektor dengan menyisipkan awalan e5 query (jika e5)
    def embed_query(self, text: str) -> list[float]:
        processed_text = prefixed_for_e5(self.model, text, is_query=True)
        # fastembed.embed mengembalikan generator, kita ambil yang pertama
        result_generator = self.client.embed([processed_text])
        return coerce_vector(next(result_generator))

    # Mengonversi daftar dokumen buku menjadi daftar vektor
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        processed_texts = [
            prefixed_for_e5(self.model, text, is_query=False)
            for text in texts
        ]
        # fastembed.embed mengembalikan generator berisikan array numpy
        result_generator = self.client.embed(processed_texts)
        return coerce_vectors(list(result_generator))
