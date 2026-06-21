# Mengimpor pytest untuk menangkap exception pengujian
import pytest

# Mengimpor exception AllLLMProvidersFailed dan kelas LLMRouter dari modul llm
from app.providers.llm import AllLLMProvidersFailed, LLMRouter


# Provider tiruan (fake) untuk mensimulasikan respons sukses atau error saat testing
class FakeProvider:
    # Inisialisasi nama provider, opsi teks jawaban, dan opsi objek error
    def __init__(self, name, answer=None, error=None):
        self.name = name
        self.answer = answer
        self.error = error

    # Menghasilkan teks jawaban atau melemparkan error tiruan jika diset
    def generate(self, prompt: str) -> str:
        # Jika parameter error diisi
        if self.error:
            # Segera lemparkan error tersebut
            raise self.error
        # Kembalikan teks jawaban tiruan
        return self.answer


# Unit test 1: Menguji apakah router LLM berhasil mengembalikan respons dari provider sukses pertama saat provider utama gagal
def test_llm_router_returns_first_successful_provider():
    # Merakit router dengan provider pertama error (rate limit) dan provider kedua sukses ("jawaban")
    router = LLMRouter(
        providers=[
            FakeProvider("gemini", error=RuntimeError("rate limited")),
            FakeProvider("groq", answer="jawaban"),
        ]
    )

    # Mengirimkan prompt ke router
    result = router.generate("prompt")

    # Memastikan teks jawaban yang diterima sukses bernilai "jawaban"
    assert result.answer == "jawaban"
    # Memastikan model/provider yang tercatat digunakan adalah "groq"
    assert result.provider_used == "groq"


# Unit test 2: Menguji apakah router LLM melemparkan exception AllLLMProvidersFailed jika seluruh provider dalam daftar gagal merespon
def test_llm_router_raises_when_all_providers_fail():
    # Merakit router dengan provider pertama mati ("down") dan provider kedua mengembalikan teks kosong ""
    router = LLMRouter(
        providers=[
            FakeProvider("gemini", error=RuntimeError("down")),
            FakeProvider("groq", answer=""),
        ]
    )

    # Memastikan pemanggilan generate memicu exception AllLLMProvidersFailed
    with pytest.raises(AllLLMProvidersFailed) as exc:
        router.generate("prompt")

    # Memverifikasi bahwa daftar provider yang dicatat gagal adalah "gemini" dan "groq"
    assert [failure.provider for failure in exc.value.failures] == ["gemini", "groq"]
