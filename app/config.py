from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    app_name: str = "KitabGuru Inference Engine"
    api_prefix: str = "/api"
    cors_origins: str = "*"

    database_url: str = "sqlite:///./data/app.db"
    qdrant_location: str = "./data/qdrant"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "epub_collection"
    retrieval_candidate_k: int = 30
    retrieval_final_k: int = 12
    retrieval_neighbor_window: int = 1
    rag_enable_completeness_scan: bool = True
    rag_max_eval_retries: int = 3
    chunk_size: int = 1200
    chunk_overlap: int = 160

    embedding_provider: str = "huggingface"
    embedding_dimension: Optional[int] = None
    hf_api_key: Optional[str] = None
    hf_embedding_model: str = "intfloat/multilingual-e5-large"
    gemini_api_key: Optional[str] = None
    gemini_embedding_model: str = "text-embedding-004"

    llm_fallback_order: str = "gemini,groq,openrouter,openai_compatible"
    llm_temperature: float = 0.0
    gemini_llm_model: str = "gemini-3.1-flash-lite"
    groq_api_key: Optional[str] = None
    groq_llm_model: str = "llama3-70b-8192"
    openrouter_api_key: Optional[str] = None
    openrouter_llm_model: str = "meta-llama/llama-3-70b-instruct"
    openai_compatible_api_key: Optional[str] = None
    openai_compatible_base_url: Optional[str] = None
    openai_compatible_model: Optional[str] = None

    evaluator_llm_fallback_order: str = "gemini,groq,openrouter,openai_compatible"
    evaluator_gemini_api_key: Optional[str] = None
    evaluator_gemini_llm_model: Optional[str] = None
    evaluator_groq_api_key: Optional[str] = None
    evaluator_groq_llm_model: Optional[str] = None
    evaluator_openrouter_api_key: Optional[str] = None
    evaluator_openrouter_llm_model: Optional[str] = None
    evaluator_openai_compatible_api_key: Optional[str] = None
    evaluator_openai_compatible_base_url: Optional[str] = None
    evaluator_openai_compatible_model: Optional[str] = None

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def llm_provider_order(self) -> list[str]:
        return [
            provider.strip().lower()
            for provider in self.llm_fallback_order.split(",")
            if provider.strip()
        ]

    @property
    def evaluator_llm_provider_order(self) -> list[str]:
        return [
            provider.strip().lower()
            for provider in self.evaluator_llm_fallback_order.split(",")
            if provider.strip()
        ]

    @property
    def normalized_embedding_provider(self) -> str:
        return self.embedding_provider.strip().lower()

    @property
    def active_embedding_model(self) -> str:
        provider = self.normalized_embedding_provider
        if provider == "huggingface":
            return self.hf_embedding_model
        if provider == "gemini":
            return self.gemini_embedding_model
        return ""

    def ensure_local_directories(self) -> None:
        if not self.qdrant_location.startswith(("http://", "https://")):
            Path(self.qdrant_location).mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite:///"):
            db_path = self.database_url.removeprefix("sqlite:///")
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_local_directories()
    return settings
