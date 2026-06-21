# Mengimpor Generator dari collections.abc untuk kebutuhan type-hinting generator sesi database
from collections.abc import Generator

# Mengimpor TestClient dari FastAPI untuk mensimulasikan panggilan API HTTP tanpa port server nyata
from fastapi.testclient import TestClient
# Mengimpor StaticPool dari SQLAlchemy untuk pengujian database SQLite in-memory agar satu koneksi tetap terbuka
from sqlalchemy.pool import StaticPool
# Mengimpor Session, SQLModel, dan create_engine dari SQLModel untuk struktur database uji
from sqlmodel import Session, SQLModel, create_engine

# Mengimpor kelas Settings untuk konfigurasi aplikasi
from app.config import Settings
# Mengimpor get_session untuk dependency overriding sesi database
from app.database import get_session
# Mengimpor dependensi app settings, qdrant, embeddings, dan llm router
from app.dependencies import get_app_settings, get_qdrant_store, get_embedding_provider, get_llm_router
# Mengimpor kelas app FastAPI utama
from app.main import app
# Mengimpor model tabel Document
from app.models import Document
# Mengimpor profil embedding dan pembuat profil embedding
from app.providers.embeddings import EmbeddingProfile, build_embedding_profile
# Mengimpor kelas GenerationResult untuk output tiruan LLM
from app.providers.llm import GenerationResult
# Mengimpor kelas SearchResult untuk output tiruan Qdrant
from app.services.qdrant_store import SearchResult


# Provider embedding tiruan (fake) untuk menguji deteksi embedding kedaluwarsa (stale)
class FakeEmbeddingProvider:
    # Inisialisasi profil model
    def __init__(self, profile: EmbeddingProfile):
        self.profile = profile

    # Memaksa error jika dipanggil (karena pengecekan stale harusnya membatalkan proses sebelum pemrosesan query)
    def embed_query(self, text: str):
        raise AssertionError("stale checks should happen before embedding")

    # Memaksa error jika dokumen diproses
    def embed_documents(self, texts: list[str]):
        raise AssertionError("not used in this test")


# Client Qdrant tiruan (fake) yang memicu error jika similarity search terpanggil saat database tidak konsisten
class FakeQdrantStore:
    def similarity_search(self, **kwargs):
        raise AssertionError("stale checks should happen before vector search")


# Router LLM tiruan (fake) yang memicu error jika dipanggil saat model embedding tidak cocok
class FakeLLMRouter:
    def generate(self, prompt: str):
        raise AssertionError("stale checks should happen before LLM generation")


# Provider embedding tiruan yang berhasil mengembalikan vektor berukuran statis
class WorkingFakeEmbeddingProvider:
    def __init__(self, profile: EmbeddingProfile):
        self.profile = profile

    # Mengembalikan vektor dummy satu dimensi
    def embed_query(self, text: str):
        return [1.0]

    # Mengembalikan list vektor dummy untuk setiap dokumen
    def embed_documents(self, texts: list[str]):
        return [[1.0] for _ in texts]


# Client Qdrant tiruan untuk mensimulasikan pengujian pemindaian kelayakan (completeness scan) RAG
class CompletenessFakeQdrantStore:
    def __init__(self):
        # Daftar sepuluh judul heading buku berbahasa Arab
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
        # Merakit sepuluh potongan data SearchResult tiruan
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
            # Iterasi 10 heading
            for index, heading in enumerate(arabic_headings, start=1)
        ]

    # Mengembalikan salah satu chunk default untuk simulasi pencarian
    def similarity_search(self, **kwargs):
        return [self.chunks[7]]

    # Mengambil chunk berdasarkan ID
    def get_by_ids(self, ids: list[str]):
        return [chunk for chunk in self.chunks if chunk.id in ids]

    # Mengembalikan seluruh chunk (scrolling data)
    def get_chunks(self, **kwargs):
        return self.chunks


# Router LLM tiruan khusus untuk menguji format jawaban yang meminta 10 poin
class TenPointFakeLLMRouter:
    def generate(self, prompt: str):
        # Memastikan potongan heading pertama ada di dalam teks prompt
        assert "كف الأذى وبذل الندى" in prompt
        # Memastikan potongan heading kesepuluh ada di dalam teks prompt
        assert "المجاملة اللطيفة" in prompt
        # Menyusun jawaban berangka 1-10 lengkap dengan sitasi [S1]-[S10]
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
        # Mengembalikan objek GenerationResult
        return GenerationResult(answer=answer, provider_used="fake")


# Unit test 1: Memverifikasi endpoint chat mengembalikan error HTTP 409 jika sidik jari model dokumen lama (stale)
def test_chat_endpoint_returns_409_for_stale_book_filter():
    # Menyiapkan konfigurasi model embedding saat ini (multilingual-e5-base)
    settings = Settings(
        embedding_provider="huggingface",
        hf_embedding_model="intfloat/multilingual-e5-base",
        hf_api_key="token",
    )
    # Membangun profil model aktif saat ini
    current = build_embedding_profile(settings)
    # Menyiapkan profil model lama yang tersimpan (multilingual-e5-large)
    stored = EmbeddingProfile(
        provider="huggingface",
        model="intfloat/multilingual-e5-large",
        dimension=1024,
        behavior="e5-query-passage-prefix",
    )
    # Membuat engine SQLite in-memory untuk database uji
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Membuat tabel database uji
    SQLModel.metadata.create_all(engine)
    # Menambahkan satu dokumen lama yang sidik jarinya berbeda ke database uji
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

    # Generator sesi database override
    def override_session() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    # Memasang dependency overrides pada FastAPI app
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider(current)
    app.dependency_overrides[get_qdrant_store] = lambda: FakeQdrantStore()
    app.dependency_overrides[get_llm_router] = lambda: FakeLLMRouter()
    try:
        # Mengirimkan request POST chat ke endpoint simulasi
        response = TestClient(app).post(
            "/api/chat",
            json={"query": "Apa isi kitab?", "book_filter": "book-1"},
        )
    # Menghapus seluruh dependency overrides setelah selesai test
    finally:
        app.dependency_overrides.clear()

    # Memastikan server merespon dengan status kode HTTP 409 Conflict
    assert response.status_code == 409
    body = response.json()["detail"]
    # Memverifikasi detail ID buku konflik
    assert body["book_id"] == "book-1"
    # Memverifikasi detail model tersimpan di database adalah e5-large
    assert body["stored_embedding"]["model"] == "intfloat/multilingual-e5-large"
    # Memverifikasi detail model aktif saat ini adalah e5-base
    assert body["current_embedding"]["model"] == "intfloat/multilingual-e5-base"


# Unit test 2: Memverifikasi fungsionalitas pemindaian kelengkapan (completeness scan) saat kueri menanyakan daftar poin tertentu
def test_chat_endpoint_uses_completeness_scan_for_numbered_list_question():
    # Menyiapkan konfigurasi pengujian dengan mengaktifkan completeness scan
    settings = Settings(
        embedding_provider="huggingface",
        hf_embedding_model="intfloat/multilingual-e5-large",
        hf_api_key="token",
        retrieval_candidate_k=5,
        retrieval_final_k=12,
        retrieval_neighbor_window=1,
        rag_enable_completeness_scan=True,
    )
    # Membangun profil model embedding yang cocok dengan dokumen
    current = build_embedding_profile(settings).with_dimension(1024)
    # Membuat engine SQLite in-memory
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Membuat tabel
    SQLModel.metadata.create_all(engine)
    # Menambahkan rekam buku uji yang cocok
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

    # Generator sesi
    def override_session() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    # Menetapkan dependency overrides pengujian
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_embedding_provider] = lambda: WorkingFakeEmbeddingProvider(current)
    app.dependency_overrides[get_qdrant_store] = lambda: CompletenessFakeQdrantStore()
    app.dependency_overrides[get_llm_router] = lambda: TenPointFakeLLMRouter()
    try:
        # Mengirimkan request chat dengan pertanyaan meminta daftar 10 poin
        response = TestClient(app).post(
            "/api/chat",
            json={
                "query": "Sebutkan 10 cara memikat hati tetangga yang dibahas dalam buku ini secara lengkap!",
                "book_filter": "book-1",
            },
        )
    # Reset overrides
    finally:
        app.dependency_overrides.clear()

    # Memastikan respons sukses (200 OK)
    assert response.status_code == 200
    body = response.json()
    # Memastikan status jawaban dinilai lengkap ('complete')
    assert body["answer_status"] == "complete"
    # Memastikan ada 10 rujukan sitasi terkumpul
    assert len(body["citations"]) == 10
    # Memastikan tidak ada teks penyangkalan parsial di dalam teks jawaban
    assert "dokumen tidak mencantumkan" not in body["answer"].lower()
    # Memastikan data statistik RAG mencatat 10 chunk pelengkap berhasil ditemukan
    assert body["retrieval_summary"]["completeness_found_count"] == 10
