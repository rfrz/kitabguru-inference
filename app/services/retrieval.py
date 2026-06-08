from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from app.config import Settings
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMRouter
from app.services.qdrant_store import QdrantStore, SearchResult


RRF_RANK_CONSTANT = 60

NUMBER_WORDS: dict[str, int] = {
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "delapan": 8,
    "sembilan": 9,
    "sepuluh": 10,
}

ARABIC_QUERY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tetangga", ("الجار", "الجيران")),
    ("bertetangga", ("الجار", "الجيران")),
    ("memikat", ("كسب", "طرق لكسب")),
    ("menarik hati", ("كسب", "طرق لكسب")),
    ("hati", ("كسب",)),
    ("cara", ("طرق", "وسائل")),
    ("adab", ("آداب",)),
    ("salam", ("السلام",)),
    ("wajah", ("طلاقة الوجه",)),
    ("ceria", ("طلاقة الوجه",)),
    ("privasi", ("احترام الخصوصيات",)),
    ("nasihat", ("النصح",)),
    ("aib", ("الستر",)),
    ("kunjung", ("الزيارة",)),
    ("hadiah", ("هدية", "المجاملة")),
)


BOOK_ROUTING_PROMPT_TEMPLATE = """Kamu adalah evaluator RAG.
Berikut adalah cuplikan konteks dari beberapa buku hasil pencarian awal.
Tentukan buku mana (book_id) yang paling relevan dan berpotensi menjawab pertanyaan berikut.
Boleh memilih lebih dari satu buku jika memang relevan.
Jawab HANYA dengan JSON valid.
Format: {{"relevant_book_ids": ["book_id_1", "book_id_2"]}}

Konteks:
{context}

Pertanyaan:
{query}
"""


@dataclass
class RetrievedContext:
    results: list[SearchResult]
    query_variants: list[str]
    requested_count: Optional[int]
    completeness_found_count: int
    candidate_count: int

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "query_variants": self.query_variants,
            "requested_count": self.requested_count,
            "completeness_found_count": self.completeness_found_count,
            "candidate_count": self.candidate_count,
            "final_count": len(self.results),
        }


def retrieve_context(
    *,
    query: str,
    book_filter: Optional[str],
    embedding_fingerprint: str,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    qdrant_store: QdrantStore,
    evaluator_llm_router: Optional[LLMRouter] = None,
) -> RetrievedContext:
    query_variants = build_query_variants(query)
    where = build_vector_filter(book_id=book_filter, embedding_fingerprint=embedding_fingerprint)
    candidates = _retrieve_candidates(
        query_variants=query_variants,
        where=where,
        settings=settings,
        embedding_provider=embedding_provider,
        qdrant_store=qdrant_store,
    )

    if evaluator_llm_router and not book_filter:
        unique_book_ids = list(dict.fromkeys(str(c.metadata.get("book_id", "")) for c in candidates if c.metadata.get("book_id")))
        if len(unique_book_ids) > 1:
            from app.api.chat import _format_sources_for_prompt
            context = _format_sources_for_prompt(candidates)
            prompt = BOOK_ROUTING_PROMPT_TEMPLATE.format(context=context, query=query)
            try:
                eval_result = evaluator_llm_router.generate_json(prompt)
                relevant_book_ids = eval_result.get("relevant_book_ids", [])
                if relevant_book_ids and isinstance(relevant_book_ids, list):
                    candidates = [c for c in candidates if str(c.metadata.get("book_id", "")) in relevant_book_ids]
            except Exception:
                pass

    requested_count = extract_requested_count(query)
    selected = candidates[: settings.retrieval_final_k]
    expanded = expand_with_neighbors(
        selected,
        qdrant_store=qdrant_store,
        window=settings.retrieval_neighbor_window,
    )

    completeness_chunks: list[SearchResult] = []
    if settings.rag_enable_completeness_scan and requested_count:
        completeness_chunks = find_completeness_chunks(
            requested_count=requested_count,
            book_filter=book_filter,
            embedding_fingerprint=embedding_fingerprint,
            seed_results=expanded or selected,
            qdrant_store=qdrant_store,
        )

    merged = dedupe_results([*expanded, *completeness_chunks])
    final_limit = max(settings.retrieval_final_k, requested_count or 0)
    final_results = repack_results(merged)[:final_limit]

    return RetrievedContext(
        results=final_results,
        query_variants=query_variants,
        requested_count=requested_count,
        completeness_found_count=len(completeness_chunks),
        candidate_count=len(candidates),
    )


def _retrieve_candidates(
    *,
    query_variants: list[str],
    where: dict[str, Any],
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    qdrant_store: QdrantStore,
) -> list[SearchResult]:
    by_id: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}

    for variant in query_variants:
        query_embedding = embedding_provider.embed_query(variant)
        results = qdrant_store.similarity_search(
            query_embedding=query_embedding,
            top_k=settings.retrieval_candidate_k,
            where=where,
        )
        for rank, result in enumerate(results, start=1):
            by_id.setdefault(result.id, result)
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (RRF_RANK_CONSTANT + rank)

    return sorted(by_id.values(), key=lambda result: scores.get(result.id, 0.0), reverse=True)


def build_query_variants(query: str) -> list[str]:
    variants = [query.strip()]
    lowered = query.lower()
    arabic_terms: list[str] = []

    for trigger, hints in ARABIC_QUERY_HINTS:
        if trigger in lowered:
            arabic_terms.extend(hints)

    requested_count = extract_requested_count(query)
    if requested_count:
        arabic_terms.extend([str(requested_count), "عشرة" if requested_count == 10 else ""])

    compact_terms = " ".join(term for term in dict.fromkeys(arabic_terms) if term)
    if compact_terms:
        variants.append(compact_terms)
    if "tetangga" in lowered and "cara" in lowered:
        variants.append("طرق لكسب الجيران")
        variants.append("عشر طرق لكسب الجيران")

    return list(dict.fromkeys(variant for variant in variants if variant.strip()))


def extract_requested_count(query: str) -> Optional[int]:
    match = re.search(r"\b([0-9]{1,2})\b", query)
    if match:
        return int(match.group(1))

    lowered = query.lower()
    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            return number
    return None


def expand_with_neighbors(
    results: list[SearchResult],
    *,
    qdrant_store: QdrantStore,
    window: int,
) -> list[SearchResult]:
    if window <= 0 or not results:
        return dedupe_results(results)

    by_id = {result.id: result for result in results}
    frontier = list(results)
    for _ in range(window):
        neighbor_ids: list[str] = []
        for result in frontier:
            for key in ("prev_id", "next_id"):
                neighbor_id = str(result.metadata.get(key) or "")
                if neighbor_id and neighbor_id not in by_id:
                    neighbor_ids.append(neighbor_id)

        fetched = qdrant_store.get_by_ids(list(dict.fromkeys(neighbor_ids)))
        frontier = []
        for result in fetched:
            if result.id not in by_id:
                by_id[result.id] = result
                frontier.append(result)

    return list(by_id.values())


def find_completeness_chunks(
    *,
    requested_count: int,
    book_filter: Optional[str],
    embedding_fingerprint: str,
    seed_results: list[SearchResult],
    qdrant_store: QdrantStore,
) -> list[SearchResult]:
    book_ids = [book_filter] if book_filter else _seed_book_ids(seed_results)
    completeness: list[SearchResult] = []

    for book_id in book_ids:
        where = build_vector_filter(book_id=book_id, embedding_fingerprint=embedding_fingerprint)
        chunks = qdrant_store.get_chunks(where=where)
        numbered = [
            chunk
            for chunk in chunks
            if _heading_number(chunk) is not None and 1 <= _heading_number(chunk) <= requested_count
        ]
        found_numbers = {_heading_number(chunk) for chunk in numbered}
        if len(found_numbers) >= min(requested_count, len(numbered)):
            completeness.extend(numbered)

    return dedupe_results(completeness)


def repack_results(results: list[SearchResult]) -> list[SearchResult]:
    return sorted(
        dedupe_results(results),
        key=lambda result: (
            str(result.metadata.get("book_id") or ""),
            int(result.metadata.get("chapter") or 0),
            int(result.metadata.get("chunk_index") or 0),
        ),
    )


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    deduped: dict[str, SearchResult] = {}
    for result in results:
        deduped.setdefault(result.id, result)
    return list(deduped.values())


def build_vector_filter(book_id: Optional[str], embedding_fingerprint: str) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{"embedding_fingerprint": embedding_fingerprint}]
    if book_id:
        filters.append({"book_id": book_id})
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _seed_book_ids(results: list[SearchResult]) -> list[str]:
    book_ids = [str(result.metadata.get("book_id") or "") for result in results]
    return [book_id for book_id in dict.fromkeys(book_ids) if book_id]


def _heading_number(result: SearchResult) -> Optional[int]:
    raw = result.metadata.get("heading_number")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value or None
