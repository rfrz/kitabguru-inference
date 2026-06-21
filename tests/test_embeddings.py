# Mengimpor kelas Settings untuk penyiapan konfigurasi testing model embedding
from app.config import Settings
# Mengimpor fungsi pembuat profil embedding dan fungsi pensisip prefix e5
from app.providers.embeddings import build_embedding_profile, prefixed_for_e5


# Unit test 1: Menguji kestabilan nilai sidik jari (fingerprint) model pada konfigurasi yang sama
def test_embedding_fingerprint_is_stable_for_same_config():
    # Menyiapkan konfigurasi model embedding e5-large
    settings = Settings(
        embedding_provider="huggingface",
        hf_embedding_model="intfloat/multilingual-e5-large",
        hf_api_key="token",
    )

    # Membuat profil pertama
    first = build_embedding_profile(settings)
    # Membuat profil kedua menggunakan pengaturan yang sama
    second = build_embedding_profile(settings)

    # Memastikan sidik jari profil pertama sama persis dengan sidik jari profil kedua (konsisten)
    assert first.fingerprint == second.fingerprint
    # Memastikan dimensi default terdeteksi bernilai 1024 kolom vektor
    assert first.dimension == 1024
    # Memastikan perilaku terdeteksi menggunakan skema awalan e5
    assert first.behavior == "e5-query-passage-prefix"


# Unit test 2: Menguji perubahan nilai sidik jari jika nama model di konfigurasi diubah
def test_embedding_fingerprint_changes_when_model_changes():
    # Membuat profil model lama (e5-large)
    old = build_embedding_profile(
        Settings(
            embedding_provider="huggingface",
            hf_embedding_model="intfloat/multilingual-e5-large",
            hf_api_key="token",
        )
    )
    # Membuat profil model baru (e5-base)
    new = build_embedding_profile(
        Settings(
            embedding_provider="huggingface",
            hf_embedding_model="intfloat/multilingual-e5-base",
            hf_api_key="token",
        )
    )

    # Memverifikasi sidik jari model lama berbeda dengan sidik jari model baru
    assert old.fingerprint != new.fingerprint


# Unit test 3: Menguji apakah fungsi penyematan awalan e5 bekerja dengan benar untuk query dan passage
def test_e5_prefixes_query_and_document_text():
    # Menyimpan nama model e5
    model = "intfloat/multilingual-e5-large"

    # Memastikan teks pertanyaan disisipkan awalan 'query: ' di depannya
    assert prefixed_for_e5(model, "Apa itu tauhid?", is_query=True) == "query: Apa itu tauhid?"
    # Memastikan teks dokumen disisipkan awalan 'passage: ' di depannya
    assert prefixed_for_e5(model, "النص العربي", is_query=False) == "passage: النص العربي"
    # Memastikan teks yang sudah memiliki awalan 'query: ' sebelumnya tidak disisipkan awalan ganda
    assert prefixed_for_e5(model, "query: Apa itu tauhid?", is_query=True) == "query: Apa itu tauhid?"
