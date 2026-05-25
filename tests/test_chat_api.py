from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.config import Settings
from app.database import get_session
from app.dependencies import get_app_settings, get_chroma_store, get_embedding_provider, get_llm_router
from app.main import app
from app.models import Document
from app.providers.embeddings import EmbeddingProfile, build_embedding_profile


class FakeEmbeddingProvider:
    def __init__(self, profile: EmbeddingProfile):
        self.profile = profile

    def embed_query(self, text: str):
        raise AssertionError("stale checks should happen before embedding")

    def embed_documents(self, texts: list[str]):
        raise AssertionError("not used in this test")


class FakeChromaStore:
    def similarity_search(self, **kwargs):
        raise AssertionError("stale checks should happen before vector search")


class FakeLLMRouter:
    def generate(self, prompt: str):
        raise AssertionError("stale checks should happen before LLM generation")


def test_chat_endpoint_returns_409_for_stale_book_filter():
    settings = Settings(
        embedding_provider="huggingface",
        hf_embedding_model="intfloat/multilingual-e5-base",
        hf_api_key="token",
    )
    current = build_embedding_profile(settings)
    stored = EmbeddingProfile(
        provider="huggingface",
        model="intfloat/multilingual-e5-large",
        dimension=1024,
        behavior="e5-query-passage-prefix",
    )
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
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

    def override_session() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider(current)
    app.dependency_overrides[get_chroma_store] = lambda: FakeChromaStore()
    app.dependency_overrides[get_llm_router] = lambda: FakeLLMRouter()
    try:
        response = TestClient(app).post(
            "/api/chat",
            json={"query": "Apa isi kitab?", "book_filter": "book-1"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    body = response.json()["detail"]
    assert body["book_id"] == "book-1"
    assert body["stored_embedding"]["model"] == "intfloat/multilingual-e5-large"
    assert body["current_embedding"]["model"] == "intfloat/multilingual-e5-base"
