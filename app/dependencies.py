# Mengimpor lru_cache untuk mencache objek store/klien agar tidak dibuat berulang kali
from functools import lru_cache

# Mengimpor HTTPException dan status HTTP dari FastAPI
from fastapi import HTTPException, status

# Mengimpor Settings and get_settings
from app.config import Settings, get_settings
# Mengimpor error exception dan pembuat client embedding
from app.providers.embeddings import ProviderConfigurationError, create_embedding_provider
# Mengimpor router LLM utama
from app.providers.llm import LLMRouter
# Mengimpor client database vektor Qdrant
from app.services.qdrant_store import QdrantStore


# Fungsi dependensi untuk mendapatkan objek model embedding aktif
def get_embedding_provider():
    """Mengembalikan objek provider model embedding yang valid."""
    try:
        # Mencoba instansiasi provider embedding berdasarkan settings
        return create_embedding_provider(get_settings())
    # Menangkap error jika parameter API Key atau nama model embedding salah
    except ProviderConfigurationError as exc:
        # Lempar error HTTP 503 Service Unavailable
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


# Menyimpan instance QdrantStore dalam cache memori agar client Qdrant digunakan kembali di seluruh request
@lru_cache
def get_qdrant_store_cached() -> QdrantStore:
    """Membuat dan mencache instance QdrantStore."""
    return QdrantStore(get_settings())


# Fungsi dependensi FastAPI untuk mengambil client database vektor Qdrant
def get_qdrant_store() -> QdrantStore:
    """Mendapatkan client database vektor Qdrant."""
    return get_qdrant_store_cached()


# Fungsi penutup koneksi client database vektor Qdrant saat server shutdown
def close_qdrant_store() -> None:
    """Menutup koneksi client Qdrant secara bersih saat membebaskan memori."""
    # Memeriksa apakah cache instance QdrantStore terisi
    if get_qdrant_store_cached.cache_info().currsize > 0:
        # Mengambil instance QdrantStore ter-cache
        store = get_qdrant_store_cached()
        # Memastikan objek store memiliki client Qdrant aktif
        if hasattr(store, "client") and store.client:
            # Menutup koneksi client Qdrant
            store.client.close()


# Fungsi dependensi untuk mendapatkan objek router LLM utama RAG
def get_llm_router() -> LLMRouter:
    """Mendapatkan router LLM utama untuk inferensi RAG."""
    return LLMRouter.from_settings(get_settings())


# Fungsi dependensi untuk mendapatkan objek router LLM khusus evaluator RAG
def get_evaluator_llm_router() -> LLMRouter:
    """Mendapatkan router LLM khusus untuk tugas evaluator kelengkapan jawaban."""
    return LLMRouter.from_settings(get_settings(), is_evaluator=True)


# Fungsi dependensi FastAPI untuk mengambil konfigurasi settings global
def get_app_settings() -> Settings:
    """Mendapatkan objek konfigurasi settings global."""
    return get_settings()
