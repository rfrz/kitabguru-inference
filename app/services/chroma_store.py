from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.config import Settings


@dataclass
class SearchResult:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: Optional[float] = None


class ChromaStore:
    def __init__(self, settings: Settings):
        import chromadb

        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(settings.chroma_collection)

    def add_chunks(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        where: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        return [
            SearchResult(
                id=item_id,
                document=document,
                metadata=metadata or {},
                distance=distance,
            )
            for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
        ]

    def delete_book(self, book_id: str) -> None:
        self.collection.delete(where={"book_id": book_id})
