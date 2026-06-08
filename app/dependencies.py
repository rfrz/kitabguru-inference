from functools import lru_cache

from fastapi import HTTPException, status

from app.config import Settings, get_settings
from app.providers.embeddings import ProviderConfigurationError, create_embedding_provider
from app.providers.llm import LLMRouter
from app.services.qdrant_store import QdrantStore


def get_embedding_provider():
    try:
        return create_embedding_provider(get_settings())
    except ProviderConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@lru_cache
def get_qdrant_store_cached() -> QdrantStore:
    return QdrantStore(get_settings())


def get_qdrant_store() -> QdrantStore:
    return get_qdrant_store_cached()


def close_qdrant_store() -> None:
    if get_qdrant_store_cached.cache_info().currsize > 0:
        store = get_qdrant_store_cached()
        if hasattr(store, "client") and store.client:
            store.client.close()


def get_llm_router() -> LLMRouter:
    return LLMRouter.from_settings(get_settings())


def get_app_settings() -> Settings:
    return get_settings()
