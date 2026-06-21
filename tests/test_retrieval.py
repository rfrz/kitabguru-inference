# Mengimpor kelas Settings untuk konfigurasi aplikasi
from app.config import Settings
# Mengimpor profil model embedding
from app.providers.embeddings import EmbeddingProfile
# Mengimpor kelas SearchResult dari modul qdrant_store
from app.services.qdrant_store import SearchResult
# Mengimpor fungsi retrieve_context dari modul retrieval
from app.services.retrieval import retrieve_context


# Provider embedding tiruan (fake) untuk merekam kueri dan mengembalikan vektor tiruan
class FakeEmbeddingProvider:
    # Menyiapkan konfigurasi profil model e5-large
    profile = EmbeddingProfile(
        provider="huggingface",
        model="intfloat/multilingual-e5-large",
        dimension=1024,
        behavior="e5-query-passage-prefix",
    )

    # Inisialisasi list perekam kueri
    def __init__(self):
        self.queries = []

    # Merekam kueri teks dan mengembalikan nilai float dummy yang dibungkus sebagai list
    def embed_query(self, text: str):
        # Menyimpan teks kueri yang dikirim untuk diuji variannya
        self.queries.append(text)
        # Mengembalikan vektor dummy satu dimensi
        return [float(len(self.queries))]

    # Mengembalikan list vektor dummy untuk setiap dokumen
    def embed_documents(self, texts: list[str]):
        return [[1.0] for _ in texts]


# Client database Qdrant tiruan (fake) untuk mensimulasikan pencarian vektor
class FakeQdrantStore:
    # Menginisialisasi kamus berisi 10 potongan data SearchResult berurutan
    def __init__(self):
        self.chunks = {
            index: SearchResult(
                id=f"book-1_chapter_1_chunk_{index}",
                document=f"{index}- heading {index}\nbody {index}",
                metadata={
                    "book_id": "book-1",
                    "title": "Kitab",
                    "chapter": 1,
                    "chunk_index": index,
                    "heading": f"heading {index}",
                    "heading_number": index,
                    # Pointer ID tetangga sebelum
                    "prev_id": f"book-1_chapter_1_chunk_{index - 1}" if index > 1 else "",
                    # Pointer ID tetangga sesudah
                    "next_id": f"book-1_chapter_1_chunk_{index + 1}" if index < 10 else "",
                    "embedding_fingerprint": "fp",
                },
            )
            # Iterasi nomor 1-10
            for index in range(1, 11)
        }

    # Mengembalikan hasil chunk indeks ke-5 sebagai hasil pencarian awal
    def similarity_search(self, **kwargs):
        return [self.chunks[5]]

    # Mengambil chunk berdasarkan ID
    def get_by_ids(self, ids: list[str]):
        return [chunk for chunk in self.chunks.values() if chunk.id in ids]

    # Mengembalikan seluruh daftar chunk untuk proses completeness scan
    def get_chunks(self, **kwargs):
        return list(self.chunks.values())


# Unit test: Memverifikasi fungsi retrieve_context berhasil memperluas chunk tetangga dan melengkapi heading informasi yang hilang
def test_retrieve_context_expands_neighbors_and_completeness_headings():
    # Menyiapkan konfigurasi parameter pencarian RAG
    settings = Settings(
        retrieval_candidate_k=5,
        retrieval_final_k=2,
        retrieval_neighbor_window=1,
        rag_enable_completeness_scan=True,
        embedding_provider="huggingface",
        hf_api_key="token",
    )

    # Mengeksekusi penarikan (retrieval) konteks RAG
    retrieval = retrieve_context(
        # Mengirim pertanyaan bertipe daftar 10 cara
        query="Sebutkan 10 cara memikat hati tetangga secara lengkap",
        book_filter="book-1",
        embedding_fingerprint="fp",
        settings=settings,
        embedding_provider=FakeEmbeddingProvider(),
        qdrant_store=FakeQdrantStore(),
    )

    # Memverifikasi jumlah poin yang diminta terdeteksi bernilai 10
    assert retrieval.requested_count == 10
    # Memverifikasi jumlah chunk pelengkap yang berhasil dikumpulkan completeness scan bernilai 10
    assert retrieval.completeness_found_count == 10
    # Memverifikasi seluruh nomor heading di dalam dokumen terurut lengkap dari 1 hingga 10
    assert [result.metadata["heading_number"] for result in retrieval.results] == list(range(1, 11))
    # Memverifikasi padanan teks kueri bahasa Arab terdeteksi di dalam list variasi kueri
    assert "طرق لكسب الجيران" in retrieval.query_variants
