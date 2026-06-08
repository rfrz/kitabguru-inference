from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.config import Settings
from app.database import get_session
from app.dependencies import get_app_settings, get_qdrant_store, get_embedding_provider, get_llm_router
from app.main import app
from app.models import Document
from app.providers.embeddings import EmbeddingProfile, build_embedding_profile
from app.providers.llm import GenerationResult
from app.services.qdrant_store import SearchResult


class FakeEmbeddingProvider:
    def __init__(self, profile: EmbeddingProfile):
        self.profile = profile

    def embed_query(self, text: str):
        raise AssertionError("stale checks should happen before embedding")

    def embed_documents(self, texts: list[str]):
        raise AssertionError("not used in this test")


class FakeQdrantStore:
    def similarity_search(self, **kwargs):
        raise AssertionError("stale checks should happen before vector search")


class FakeLLMRouter:
    def generate(self, prompt: str):
        raise AssertionError("stale checks should happen before LLM generation")


class WorkingFakeEmbeddingProvider:
    def __init__(self, profile: EmbeddingProfile):
        self.profile = profile

    def embed_query(self, text: str):
        return [1.0]

    def embed_documents(self, texts: list[str]):
        return [[1.0] for _ in texts]


class CompletenessFakeQdrantStore:
    def __init__(self):
        arabic_headings = [
            "كف الأذى وبذل الندى",
            "البدء بالسلام",
            "طلاقة الوجه",
            "المواساة في الشدة",
            "احترام الخصوصيات",
            "قبول الأعذار",
            "النصح برفق ولين",
            "الستر وترك التعيير",
            "الزيارة",
            "المجاملة اللطيفة",
        ]
        self.chunks = [
            SearchResult(
                id=f"book-1_chapter_1_chunk_{index}",
                document=f"{index}- {heading}\nشرح مختصر.",
                metadata={
                    "book_id": "book-1",
                    "title": "10 طرق لكسب الجيران",
                    "chapter": 1,
                    "chunk_index": index,
                    "heading": heading,
                    "heading_number": index,
                    "prev_id": f"book-1_chapter_1_chunk_{index - 1}" if index > 1 else "",
                    "next_id": f"book-1_chapter_1_chunk_{index + 1}" if index < 10 else "",
                    "embedding_fingerprint": "fp",
                },
            )
            for index, heading in enumerate(arabic_headings, start=1)
        ]

    def similarity_search(self, **kwargs):
        return [self.chunks[7]]

    def get_by_ids(self, ids: list[str]):
        return [chunk for chunk in self.chunks if chunk.id in ids]

    def get_chunks(self, **kwargs):
        return self.chunks


class TenPointFakeLLMRouter:
    def generate(self, prompt: str):
        assert "كف الأذى وبذل الندى" in prompt
        assert "المجاملة اللطيفة" in prompt
        answer = "\n".join(
            [
                "1. Menahan gangguan dan bersikap dermawan. [S1]",
                "2. Memulai salam. [S2]",
                "3. Menampakkan wajah ceria. [S3]",
                "4. Membantu saat kesulitan. [S4]",
                "5. Menghormati privasi. [S5]",
                "6. Menerima alasan atau permintaan maaf. [S6]",
                "7. Menasihati dengan lembut. [S7]",
                "8. Menutup aib dan tidak mencela. [S8]",
                "9. Saling mengunjungi. [S9]",
                "10. Beramah-tamah dengan lembut. [S10]",
            ]
        )
        return GenerationResult(answer=answer, provider_used="fake")


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
    app.dependency_overrides[get_qdrant_store] = lambda: FakeQdrantStore()
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


def test_chat_endpoint_uses_completeness_scan_for_numbered_list_question():
    settings = Settings(
        embedding_provider="huggingface",
        hf_embedding_model="intfloat/multilingual-e5-large",
        hf_api_key="token",
        retrieval_candidate_k=5,
        retrieval_final_k=12,
        retrieval_neighbor_window=1,
        rag_enable_completeness_scan=True,
    )
    current = build_embedding_profile(settings).with_dimension(1024)
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
                title="10 طرق لكسب الجيران",
                total_chunks=10,
                embedding_provider=current.provider,
                embedding_model=current.model,
                embedding_dimension=current.dimension,
                embedding_fingerprint=current.fingerprint,
            )
        )
        session.commit()

    def override_session() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: WorkingFakeEmbeddingProvider(current)
    app.dependency_overrides[get_qdrant_store] = lambda: CompletenessFakeQdrantStore()
    app.dependency_overrides[get_llm_router] = lambda: TenPointFakeLLMRouter()
    try:
        response = TestClient(app).post(
            "/api/chat",
            json={
                "query": "Sebutkan 10 cara memikat hati tetangga yang dibahas dalam buku ini secara lengkap!",
                "book_filter": "book-1",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer_status"] == "complete"
    assert len(body["citations"]) == 10
    assert "dokumen tidak mencantumkan" not in body["answer"].lower()
    assert body["retrieval_summary"]["completeness_found_count"] == 10
