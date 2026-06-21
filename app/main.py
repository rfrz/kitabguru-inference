# Mengimpor modul logging untuk pencatatan logs sistem inferensi
import logging
# Mengimpor utilitas context manager asinkron untuk lifecycle FastAPI
from contextlib import asynccontextmanager

# Mengimpor class utama FastAPI untuk inisiasi web framework
from fastapi import FastAPI
# Mengimpor middleware CORS untuk menangani akses lintas asal
from fastapi.middleware.cors import CORSMiddleware

# Mengimpor router API chat dari submodule api
from app.api.chat import router as chat_router
# Mengimpor router API manajemen dokumen dari submodule api
from app.api.documents import router as documents_router
# Mengimpor helper get_settings
from app.config import get_settings
# Mengimpor fungsi inisiasi tabel database
from app.database import init_db

# Mengonfigurasi logger dasar ke tingkat INFO
logging.basicConfig(level=logging.INFO)

# Mengambil konfigurasi settings global
settings = get_settings()

# Mengimpor helper penutup koneksi Qdrant
from app.dependencies import close_qdrant_store


# Context manager asinkron untuk mengelola siklus hidup startup dan shutdown FastAPI
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ── Startup Hook ──────────────────────────────────────────────────────
    # Membuat tabel database SQLite lokal jika belum ada di disk
    init_db()
    # Mengalirkan kendali kembali ke FastAPI
    yield
    # ── Shutdown Hook ─────────────────────────────────────────────────────
    # Menutup koneksi client database vektor Qdrant secara bersih demi membebaskan resource
    close_qdrant_store()


# Membuat instance aplikasi web FastAPI dengan konfigurasi lifespan dan judul
app = FastAPI(title=settings.app_name, lifespan=lifespan)

# Memasang middleware CORS agar service backend utama atau frontend dapat memanggil endpoint RAG
app.add_middleware(
    CORSMiddleware,
    # Mengambil daftar domain asal yang diizinkan dari settings
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoint kesehatan dasar (Health Check)
@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    # Mengembalikan status json ok
    return {"status": "ok"}


# Mendaftarkan router endpoint API obrolan chat RAG
app.include_router(chat_router, prefix=settings.api_prefix)
# Mendaftarkan router endpoint API impor dan manajemen dokumen epub
app.include_router(documents_router, prefix=settings.api_prefix)
