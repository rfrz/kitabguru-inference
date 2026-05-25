import pytest
from fastapi import HTTPException
from sqlmodel import Session, SQLModel, create_engine

from app.api.chat import _raise_if_stale_embeddings
from app.models import Document
from app.providers.embeddings import EmbeddingProfile


def test_stale_book_filter_returns_409_with_current_embedding_details():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    current = EmbeddingProfile(
        provider="huggingface",
        model="intfloat/multilingual-e5-base",
        dimension=768,
        behavior="e5-query-passage-prefix",
    )
    stored = EmbeddingProfile(
        provider="huggingface",
        model="intfloat/multilingual-e5-large",
        dimension=1024,
        behavior="e5-query-passage-prefix",
    )

    with Session(engine) as session:
        session.add(
            Document(
                book_id="book-1",
                title="Kitab",
                total_chunks=1,
                embedding_provider=stored.provider,
                embedding_model=stored.model,
                embedding_dimension=stored.dimension,
                embedding_fingerprint=stored.fingerprint,
            )
        )
        session.commit()

        with pytest.raises(HTTPException) as exc:
            _raise_if_stale_embeddings(session, "book-1", current)

    assert exc.value.status_code == 409
    assert exc.value.detail["book_id"] == "book-1"
    assert exc.value.detail["stored_embedding"]["model"] == "intfloat/multilingual-e5-large"
    assert exc.value.detail["current_embedding"]["model"] == "intfloat/multilingual-e5-base"
