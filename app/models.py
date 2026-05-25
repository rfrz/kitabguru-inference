from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    book_id: str = Field(primary_key=True, index=True)
    title: str
    author: Optional[str] = None
    total_chunks: int
    created_at: datetime = Field(default_factory=utc_now, nullable=False)

    embedding_provider: str
    embedding_model: str
    embedding_dimension: Optional[int] = None
    embedding_fingerprint: str = Field(index=True)
