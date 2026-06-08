from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import Settings


def _str_to_uuid(string_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, string_id))


@dataclass
class SearchResult:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: Optional[float] = None


class QdrantStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection_name = settings.qdrant_collection
        
        kwargs: dict[str, Any] = {}
        if settings.qdrant_location.startswith(("http://", "https://")):
            kwargs["url"] = settings.qdrant_location
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
        else:
            kwargs["path"] = settings.qdrant_location
            
        self.client = QdrantClient(**kwargs)

    def _ensure_collection_exists(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
            )

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
            
        self._ensure_collection_exists(vector_size=len(embeddings[0]))

        points = []
        for point_id, doc, vec, meta in zip(ids, documents, embeddings, metadatas):
            payload = meta.copy()
            payload["_document"] = doc
            payload["_original_id"] = point_id
            points.append(
                models.PointStruct(
                    id=_str_to_uuid(point_id),
                    vector=vec,
                    payload=payload
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def _build_qdrant_filter(self, where: Optional[dict[str, Any]]) -> Optional[models.Filter]:
        if not where:
            return None
        must_conditions = []
        if "$and" in where:
            for cond in where["$and"]:
                for k, v in cond.items():
                    must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        else:
            for k, v in where.items():
                must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
                
        if must_conditions:
            return models.Filter(must=must_conditions)
        return None

    def similarity_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        where: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        if not self.client.collection_exists(self.collection_name):
            return []
            
        qdrant_filter = self._build_qdrant_filter(where)
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        
        out = []
        for point in response.points:
            payload = (point.payload or {}).copy()
            doc = payload.pop("_document", "")
            original_id = payload.pop("_original_id", str(point.id))
            out.append(
                SearchResult(
                    id=original_id,
                    document=str(doc),
                    metadata=payload,
                    distance=point.score,
                )
            )
        return out

    def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        if not ids or not self.client.collection_exists(self.collection_name):
            return []
            
        qdrant_ids = [_str_to_uuid(i) for i in ids]
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=qdrant_ids,
            with_payload=True,
        )
        
        out = []
        for point in results:
            payload = (point.payload or {}).copy()
            doc = payload.pop("_document", "")
            original_id = payload.pop("_original_id", str(point.id))
            out.append(
                SearchResult(
                    id=original_id,
                    document=str(doc),
                    metadata=payload,
                )
            )
        return out

    def get_chunks(self, *, where: Optional[dict[str, Any]] = None, limit: Optional[int] = None) -> list[SearchResult]:
        if not self.client.collection_exists(self.collection_name):
            return []
            
        qdrant_filter = self._build_qdrant_filter(where)
        
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=limit or 10000,
            with_payload=True,
        )
        
        out = []
        for point in results:
            payload = (point.payload or {}).copy()
            doc = payload.pop("_document", "")
            original_id = payload.pop("_original_id", str(point.id))
            out.append(
                SearchResult(
                    id=original_id,
                    document=str(doc),
                    metadata=payload,
                )
            )
        return out

    def delete_book(self, book_id: str) -> None:
        if not self.client.collection_exists(self.collection_name):
            return
            
        qdrant_filter = models.Filter(
            must=[models.FieldCondition(key="book_id", match=models.MatchValue(value=book_id))]
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=qdrant_filter)
        )
