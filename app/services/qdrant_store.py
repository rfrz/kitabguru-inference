# Mengaktifkan evaluasi tipe data bertunda (postponed evaluation of annotations) agar kompatibel ke belakang
from __future__ import annotations

# Mengimpor modul uuid untuk konversi string ID menjadi UUID v5
import uuid
# Mengimpor dataclass untuk mempermudah struktur model data hasil pencarian
from dataclasses import dataclass
# Mengimpor Any dan Optional untuk anotasi tipe data python
from typing import Any, Optional

# Mengimpor klien Qdrant resmi untuk berkomunikasi dengan database vektor
from qdrant_client import QdrantClient
# Mengimpor objek models dari qdrant_client untuk mendefinisikan struktur data kueri dan filter
from qdrant_client.http import models

# Mengimpor kelas Settings untuk konfigurasi parameter database Qdrant
from app.config import Settings


# Fungsi pembantu untuk mengubah string ID biasa menjadi UUID v5 yang unik secara deterministik
def _str_to_uuid(string_id: str) -> str:
    # Menghasilkan UUID v5 menggunakan ruang nama OID dan string_id target
    return str(uuid.uuid5(uuid.NAMESPACE_OID, string_id))


# Kelas data untuk menampung hasil pencarian vektor (ID, isi teks, metadata, dan skor kemiripan/distance)
@dataclass
class SearchResult:
    # ID potongan asli dokumen
    id: str
    # Isi teks dokumen
    document: str
    # Metadata pendukung
    metadata: dict[str, Any]
    # Skor kemiripan kosinus (cosine distance)
    distance: Optional[float] = None


# Kelas Service untuk mengelola interaksi CRUD dan pencarian vektor ke database Qdrant
class QdrantStore:
    # Inisialisasi klien Qdrant menggunakan konfigurasi koneksi yang diset di settings
    def __init__(self, settings: Settings):
        # Menyimpan instance konfigurasi
        self.settings = settings
        # Mengambil nama koleksi database vektor
        self.collection_name = settings.qdrant_collection
        
        # Inisialisasi argumen koneksi klien Qdrant
        kwargs: dict[str, Any] = {}
        # Jika lokasi Qdrant diawali dengan 'http://' atau 'https://' (akses server remote/cloud)
        if settings.qdrant_location.startswith(("http://", "https://")):
            # Tentukan URL server remote
            kwargs["url"] = settings.qdrant_location
            # Sisipkan API key jika dikonfigurasikan
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
        # Jika berupa path lokal (akses database Qdrant offline/in-memory)
        else:
            # Tentukan path direktori penyimpanan database lokal
            kwargs["path"] = settings.qdrant_location
            
        # Membuat instans QdrantClient dengan argumen koneksi di atas
        self.client = QdrantClient(**kwargs)

    # Memastikan koleksi database vektor sudah terbuat, buat baru jika belum ada
    def _ensure_collection_exists(self, vector_size: int) -> None:
        # Memeriksa keaktifan koleksi di server Qdrant
        if not self.client.collection_exists(self.collection_name):
            # Membuat koleksi baru dengan konfigurasi dimensi vektor dan metode perhitungan jarak (Kosinus)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    # Jumlah dimensi vektor
                    size=vector_size,
                    # Menggunakan jarak Kosinus untuk kemiripan semantik
                    distance=models.Distance.COSINE
                ),
            )

    # Menambahkan sekumpulan chunk dokumen dan vektornya ke database Qdrant
    def add_chunks(
        self,
        *,
        # Daftar ID potongan teks
        ids: list[str],
        # Daftar teks asli dokumen
        documents: list[str],
        # Daftar vektor float hasil embedding
        embeddings: list[list[float]],
        # Daftar metadata tambahan
        metadatas: list[dict[str, Any]],
    ) -> None:
        # Jika list ID kosong, segera hentikan proses
        if not ids:
            return
            
        # Memastikan koleksi Qdrant sudah terbuat dengan ukuran dimensi dari baris vektor pertama
        self._ensure_collection_exists(vector_size=len(embeddings[0]))

        # Inisialisasi daftar point data Qdrant
        points = []
        # Menggabungkan ID, teks, vektor, dan metadata secara pararel menggunakan zip
        for point_id, doc, vec, meta in zip(ids, documents, embeddings, metadatas):
            # Salin metadata agar tidak merusak objek aslinya
            payload = meta.copy()
            # Menyimpan teks asli dokumen di dalam field khusus '_document'
            payload["_document"] = doc
            # Menyimpan ID asli dokumen di dalam field khusus '_original_id'
            payload["_original_id"] = point_id
            # Menyusun objek PointStruct Qdrant
            points.append(
                models.PointStruct(
                    # ID titik vektor dalam format UUID v5 yang sah
                    id=_str_to_uuid(point_id),
                    # Data vektor
                    vector=vec,
                    # Payload metadata
                    payload=payload
                )
            )
            
        # Menyimpan/memperbarui (upsert) titik vektor tersebut ke dalam koleksi Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    # Memperbarui payload metadata (seperti judul dan penulis) pada titik-titik Qdrant terkait book_id tertentu
    def update_book_metadata(self, book_id: str, new_title: Optional[str] = None, new_author: Optional[str] = None) -> None:
        # Jika koleksi tidak terdaftar, hentikan proses
        if not self.client.collection_exists(self.collection_name):
            return
            
        # Membuat payload update
        payload_update: dict[str, Any] = {}
        if new_title is not None:
            payload_update["title"] = new_title
        if new_author is not None:
            payload_update["author"] = new_author
            
        if not payload_update:
            return

        # Menyusun filter pencocokan kolom book_id
        qdrant_filter = models.Filter(
            must=[models.FieldCondition(key="book_id", match=models.MatchValue(value=book_id))]
        )
        
        # Memperbarui payload di Qdrant
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload_update,
            points=models.FilterSelector(filter=qdrant_filter)
        )

    # Mengonversi format filter dictionary kustom menjadi objek models.Filter resmi Qdrant
    def _build_qdrant_filter(self, where: Optional[dict[str, Any]]) -> Optional[models.Filter]:
        # Jika filter kosong, kembalikan None
        if not where:
            return None
        # List kondisi wajib cocok
        must_conditions = []
        # Jika filter menggunakan operator logika '$and'
        if "$and" in where:
            # Iterasi setiap kondisi di dalam array '$and'
            for cond in where["$and"]:
                # Ambil key dan value kondisi pencocokan kolom
                for k, v in cond.items():
                    # Tambahkan kondisi pencocokan FieldCondition ke list
                    must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        # Jika berupa filter pencocokan field biasa tanpa operator logika
        else:
            for k, v in where.items():
                must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
                
        # Jika terdapat kondisi yang berhasil disusun
        if must_conditions:
            # Mengembalikan objek Filter Qdrant
            return models.Filter(must=must_conditions)
        return None

    # Melakukan pencarian vektor kemiripan (similarity search) di Qdrant
    def similarity_search(
        self,
        *,
        # Vektor embedding dari teks kueri pertanyaan
        query_embedding: list[float],
        # Jumlah maksimal dokumen relevan yang dikembalikan (top K)
        top_k: int,
        # Opsi filter pencocokan (seperti filter ID buku)
        where: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        # Jika koleksi tidak ada di Qdrant, langsung kembalikan list kosong
        if not self.client.collection_exists(self.collection_name):
            return []
            
        # Merakit objek filter kueri Qdrant
        qdrant_filter = self._build_qdrant_filter(where)
        
        # Mengeksekusi pencarian vektor ke Qdrant menggunakan metode query_points
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        
        # Menampung list hasil akhir
        out = []
        # Iterasi setiap titik hasil pencarian yang dikembalikan
        for point in response.points:
            # Menyalin payload
            payload = (point.payload or {}).copy()
            # Mengekstrak teks dokumen asli dari payload
            doc = payload.pop("_document", "")
            # Mengekstrak ID asli dokumen dari payload
            original_id = payload.pop("_original_id", str(point.id))
            # Memasukkan ke list hasil akhir SearchResult
            out.append(
                SearchResult(
                    id=original_id,
                    document=str(doc),
                    metadata=payload,
                    # Menyertakan skor kemiripan kosinus
                    distance=point.score,
                )
            )
        # Mengembalikan daftar hasil pencarian terurut dari yang paling relevan
        return out

    # Mengambil detail beberapa titik chunk secara langsung berdasarkan daftar ID-nya
    def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        # Jika list ID kosong atau koleksi tidak ada di Qdrant, kembalikan list kosong
        if not ids or not self.client.collection_exists(self.collection_name):
            return []
            
        # Mengonversi seluruh ID asli menjadi UUID v5 Qdrant
        qdrant_ids = [_str_to_uuid(i) for i in ids]
        # Mengambil data titik vektor dari server Qdrant secara asinkron/sinkron
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=qdrant_ids,
            with_payload=True,
        )
        
        # Menampung hasil akhir
        out = []
        # Iterasi setiap titik yang berhasil ditarik
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
        # Mengembalikan list hasil chunk
        return out

    # Mengambil kumpulan chunk dari database Qdrant dengan filter dan batasan limit (scrolling data)
    def get_chunks(self, *, where: Optional[dict[str, Any]] = None, limit: Optional[int] = None) -> list[SearchResult]:
        # Jika koleksi tidak ada di Qdrant, kembalikan list kosong
        if not self.client.collection_exists(self.collection_name):
            return []
            
        # Merakit objek filter
        qdrant_filter = self._build_qdrant_filter(where)
        
        # Mengeksekusi penarikan data secara gulir (scroll) dari Qdrant
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=limit or 10000,
            with_payload=True,
        )
        
        # Menampung list hasil akhir
        out = []
        # Iterasi hasil scroll
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
        # Mengembalikan daftar hasil chunk
        return out

    # Menghapus seluruh titik vektor terkait book_id tertentu dari koleksi Qdrant
    def delete_book(self, book_id: str) -> None:
        # Jika koleksi tidak terdaftar, hentikan proses
        if not self.client.collection_exists(self.collection_name):
            return
            
        # Menyusun filter pencocokan kolom book_id
        qdrant_filter = models.Filter(
            must=[models.FieldCondition(key="book_id", match=models.MatchValue(value=book_id))]
        )
        # Menghapus seluruh data vektor yang cocok dengan kriteria filter tersebut
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=qdrant_filter)
        )
