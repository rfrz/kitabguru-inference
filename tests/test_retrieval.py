from app.config import Settings
from app.providers.embeddings import EmbeddingProfile
from app.services.chroma_store import SearchResult
from app.services.retrieval import retrieve_context


class FakeEmbeddingProvider:
    profile = EmbeddingProfile(
        provider="huggingface",
        model="intfloat/multilingual-e5-large",
        dimension=1024,
        behavior="e5-query-passage-prefix",
    )

    def __init__(self):
        self.queries = []

    def embed_query(self, text: str):
        self.queries.append(text)
        return [float(len(self.queries))]

    def embed_documents(self, texts: list[str]):
        return [[1.0] for _ in texts]


class FakeChromaStore:
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
                    "prev_id": f"book-1_chapter_1_chunk_{index - 1}" if index > 1 else "",
                    "next_id": f"book-1_chapter_1_chunk_{index + 1}" if index < 10 else "",
                    "embedding_fingerprint": "fp",
                },
            )
            for index in range(1, 11)
        }

    def similarity_search(self, **kwargs):
        return [self.chunks[5]]

    def get_by_ids(self, ids: list[str]):
        return [chunk for chunk in self.chunks.values() if chunk.id in ids]

    def get_chunks(self, **kwargs):
        return list(self.chunks.values())


def test_retrieve_context_expands_neighbors_and_completeness_headings():
    settings = Settings(
        retrieval_candidate_k=5,
        retrieval_final_k=2,
        retrieval_neighbor_window=1,
        rag_enable_completeness_scan=True,
        embedding_provider="huggingface",
        hf_api_key="token",
    )

    retrieval = retrieve_context(
        query="Sebutkan 10 cara memikat hati tetangga secara lengkap",
        book_filter="book-1",
        embedding_fingerprint="fp",
        settings=settings,
        embedding_provider=FakeEmbeddingProvider(),
        chroma_store=FakeChromaStore(),
    )

    assert retrieval.requested_count == 10
    assert retrieval.completeness_found_count == 10
    assert [result.metadata["heading_number"] for result in retrieval.results] == list(range(1, 11))
    assert "طرق لكسب الجيران" in retrieval.query_variants
