from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Protocol

from app.config import Settings


FINGERPRINT_VERSION = 1


class ProviderConfigurationError(RuntimeError):
    """Raised when a configured provider cannot be used."""


@dataclass(frozen=True)
class EmbeddingProfile:
    provider: str
    model: str
    dimension: Optional[int]
    behavior: str

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": FINGERPRINT_VERSION,
            "provider": self.provider.strip().lower(),
            "model": self.model.strip(),
            "dimension": self.dimension,
            "behavior": self.behavior,
        }
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def with_dimension(self, dimension: Optional[int]) -> "EmbeddingProfile":
        return replace(self, dimension=dimension)


class EmbeddingProvider(Protocol):
    @property
    def profile(self) -> EmbeddingProfile:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


KNOWN_DIMENSIONS: dict[tuple[str, str], int] = {
    ("huggingface", "intfloat/multilingual-e5-large"): 1024,
    ("huggingface", "intfloat/multilingual-e5-base"): 768,
    ("huggingface", "intfloat/multilingual-e5-small"): 384,
    ("gemini", "text-embedding-004"): 768,
    ("gemini", "models/text-embedding-004"): 768,
    ("gemini", "gemini-embedding-001"): 3072,
    ("gemini", "gemini-embedding-2"): 3072,
}


def is_e5_model(model: str) -> bool:
    return "e5" in model.lower()


def embedding_behavior(provider: str, model: str) -> str:
    if provider == "huggingface" and is_e5_model(model):
        return "e5-query-passage-prefix"
    if provider == "gemini":
        return "gemini-semantic-similarity"
    return "default"


def resolve_embedding_dimension(settings: Settings) -> Optional[int]:
    if settings.embedding_dimension is not None:
        return settings.embedding_dimension
    provider = settings.normalized_embedding_provider
    model = settings.active_embedding_model
    return KNOWN_DIMENSIONS.get((provider, model))


def build_embedding_profile(settings: Settings) -> EmbeddingProfile:
    provider = settings.normalized_embedding_provider
    model = settings.active_embedding_model
    if provider not in {"huggingface", "gemini"}:
        raise ProviderConfigurationError(f"Unsupported embedding provider: {provider}")
    if not model:
        raise ProviderConfigurationError(f"No embedding model configured for provider: {provider}")
    return EmbeddingProfile(
        provider=provider,
        model=model,
        dimension=resolve_embedding_dimension(settings),
        behavior=embedding_behavior(provider, model),
    )


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.normalized_embedding_provider
    if provider == "huggingface":
        return HuggingFaceEmbeddingProvider(settings)
    if provider == "gemini":
        return GeminiEmbeddingProvider(settings)
    raise ProviderConfigurationError(f"Unsupported embedding provider: {provider}")


def prefixed_for_e5(model: str, text: str, *, is_query: bool) -> str:
    if not is_e5_model(model):
        return text
    prefix = "query: " if is_query else "passage: "
    return text if text.lower().startswith(prefix) else f"{prefix}{text}"


def coerce_vector(value) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError("Embedding response is not a vector")
    if value and isinstance(value[0], list):
        if len(value) == 1:
            return coerce_vector(value[0])
        width = len(value[0])
        if width == 0:
            return []
        return [
            float(sum(float(row[index]) for row in value) / len(value))
            for index in range(width)
        ]
    return [float(item) for item in value]


def coerce_vectors(values: Iterable) -> list[list[float]]:
    return [coerce_vector(value) for value in values]


class HuggingFaceEmbeddingProvider:
    def __init__(self, settings: Settings):
        if not settings.hf_api_key:
            raise ProviderConfigurationError("HF_API_KEY is required for Hugging Face embeddings")
        self.model = settings.hf_embedding_model
        self._profile = build_embedding_profile(settings)
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ProviderConfigurationError("Install huggingface-hub to use Hugging Face embeddings") from exc

        try:
            self.client = InferenceClient(provider="hf-inference", api_key=settings.hf_api_key)
        except TypeError:
            self.client = InferenceClient(token=settings.hf_api_key)

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(prefixed_for_e5(self.model, text, is_query=True))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            self._embed_one(prefixed_for_e5(self.model, text, is_query=False))
            for text in texts
        ]

    def _embed_one(self, text: str) -> list[float]:
        result = self.client.feature_extraction(text, model=self.model)
        return coerce_vector(result)


class GeminiEmbeddingProvider:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ProviderConfigurationError("GEMINI_API_KEY is required for Gemini embeddings")
        self.model = settings.gemini_embedding_model
        self.dimension = settings.embedding_dimension
        self._profile = build_embedding_profile(settings)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ProviderConfigurationError("Install google-genai to use Gemini embeddings") from exc

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.types = types

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def embed_query(self, text: str) -> list[float]:
        return self._embed_many([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_many(texts)

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        config_kwargs = {}
        if self.dimension:
            config_kwargs["output_dimensionality"] = self.dimension
        if self.model in {"text-embedding-004", "models/text-embedding-004", "gemini-embedding-001"}:
            config_kwargs["task_type"] = "SEMANTIC_SIMILARITY"
        config = self.types.EmbedContentConfig(**config_kwargs) if config_kwargs else None
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=config,
        )
        return coerce_vectors([embedding.values for embedding in result.embeddings])
