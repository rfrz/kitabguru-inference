# Mengimpor Generator untuk type-hinting fungsi yield session database
from collections.abc import Generator

# Mengimpor Session, SQLModel, dan create_engine dari pustaka SQLModel (ORM modern berbasis SQLAlchemy)
from sqlmodel import Session, SQLModel, create_engine

# Mengimpor Settings dan get_settings dari config
from app.config import Settings, get_settings


# Fungsi internal pembangun engine koneksi database SQLModel
def create_db_engine(settings: Settings):
    # Menyusun argumen koneksi tambahan database
    connect_args = {}
    # Jika database menggunakan SQLite file lokal
    if settings.database_url.startswith("sqlite"):
        # Menonaktifkan pembatasan thread SQLite agar kompatibel dengan pemrosesan FastAPI asinkron
        connect_args["check_same_thread"] = False
    # Membuat engine koneksi dengan parameter URL database
    return create_engine(settings.database_url, connect_args=connect_args)


# Membuat instance engine koneksi database global dengan memanggil settings ter-cache
engine = create_db_engine(get_settings())


# Menginisialisasi database dengan membuat seluruh tabel yang didefinisikan oleh model SQLModel jika belum dibuat
def init_db() -> None:
    """FastAPI startup hook: membuat seluruh tabel database jika belum ada."""
    SQLModel.metadata.create_all(engine)


# Fungsi dependensi FastAPI untuk menyediakan sesi database SQLModel per request body
def get_session() -> Generator[Session, None, None]:
    """Menyediakan sesi database SQLModel."""
    # Membuka sesi database baru secara aman
    with Session(engine) as session:
        # Memberikan session ke endpoint pemanggil
        yield session
