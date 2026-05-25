from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    book_filter: Optional[str] = None


class Source(BaseModel):
    id: str
    document: str
    metadata: dict[str, Any]
    distance: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    provider_used: str
    sources: list[Source]


class EmbeddingState(BaseModel):
    provider: str
    model: str
    dimension: Optional[int] = None
    fingerprint: str


class DocumentRead(BaseModel):
    book_id: str
    title: str
    author: Optional[str] = None
    total_chunks: int
    created_at: datetime
    embedding_provider: str
    embedding_model: str
    embedding_dimension: Optional[int] = None
    embedding_fingerprint: str
    is_embedding_current: bool


class DocumentImportResponse(BaseModel):
    book_id: str
    title: str
    author: Optional[str] = None
    total_chunks: int
    embedding: EmbeddingState


class ProviderFailure(BaseModel):
    provider: str
    error: str
