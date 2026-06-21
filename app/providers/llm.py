# Mengaktifkan evaluasi tipe data bertunda (postponed evaluation of annotations) agar kompatibel ke belakang
from __future__ import annotations

# Mengimpor modul logging untuk pencatatan log sistem
import logging
# Mengimpor dataclass untuk representasi struktur data respons LLM
from dataclasses import dataclass
# Mengimpor komponen Protocol dan Any untuk type-hinting antarmuka
from typing import Any, Protocol
# Mengimpor modul json untuk parsing respons teks JSON dari LLM
import json
# Mengimpor modul regex (re) untuk mendeteksi tag code-block JSON pada output LLM
import re

# Mengimpor kelas Settings untuk membaca konfigurasi provider LLM
from app.config import Settings

# Menginisialisasi logger khusus modul llm
logger = logging.getLogger(__name__)


# Struktur data untuk menampung hasil pembuatan teks (generate) yang sukses
@dataclass
class GenerationResult:
    # Teks jawaban hasil generate
    answer: str
    # Nama model/provider yang sukses digunakan
    provider_used: str


# Struktur data untuk mencatat detail kegagalan suatu provider LLM
@dataclass
class ProviderFailure:
    # Nama provider yang gagal dipanggil
    provider: str
    # Detail pesan kesalahan (error)
    error: str


# Exception khusus yang dilemparkan jika seluruh provider LLM dalam antrean fallback gagal merespon
class AllLLMProvidersFailed(RuntimeError):
    # Menyimpan daftar detail kegagalan provider
    def __init__(self, failures: list[ProviderFailure]):
        self.failures = failures
        # Memanggil konstruktor super dengan pesan default
        super().__init__("All LLM providers failed")


# Antarmuka (Protocol) standar yang wajib dimiliki oleh setiap class provider LLM
class LLMProvider(Protocol):
    # Nama provider
    name: str
    # Nama model
    model: str

    # Metode sinkron untuk menghasilkan teks dari prompt
    def generate(self, prompt: str) -> str:
        ...


# Kelas implementasi provider LLM menggunakan Google GenAI SDK (Gemini)
class GeminiLLMProvider:
    # Menetapkan nama provider ke 'gemini'
    name = "gemini"

    # Inisialisasi parameter model, temperatur, dan mengimpor modul google-genai
    def __init__(self, api_key: str, model: str, temperature: float):
        # Nama model target (misal: 'gemini-2.5-flash')
        self.model = model
        # Nilai temperatur kreativitas LLM
        self.temperature = temperature
        # Mengimpor modul SDK secara lokal
        from google import genai
        from google.genai import types

        # Membuat client GenAI SDK dengan API Key
        self.client = genai.Client(api_key=api_key)
        # Menyimpan types untuk konfigurasi
        self.types = types

    # Mengirim request teks ke model Gemini
    def generate(self, prompt: str) -> str:
        # Memanggil generator konten secara sinkron
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            # Menyusun konfigurasi temperatur LLM
            config=self.types.GenerateContentConfig(temperature=self.temperature),
        )
        # Mengembalikan teks hasil generate (atau string kosong jika kosong/gagal)
        return getattr(response, "text", None) or ""


# Kelas implementasi provider LLM menggunakan API Groq
class GroqLLMProvider:
    # Menetapkan nama provider ke 'groq'
    name = "groq"

    # Inisialisasi parameter model, temperatur, dan client SDK Groq
    def __init__(self, api_key: str, model: str, temperature: float):
        # Nama model target (misal: 'llama3-70b-8192')
        self.model = model
        # Nilai temperatur
        self.temperature = temperature
        # Mengimpor kelas Groq dari pustaka resmi groq
        from groq import Groq

        # Membuat instans client Groq
        self.client = Groq(api_key=api_key)

    # Mengirim request chat completion ke API Groq
    def generate(self, prompt: str) -> str:
        # Memanggil API chat completion secara sinkron
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        # Mengembalikan konten pesan hasil tanggapan
        return completion.choices[0].message.content or ""


# Kelas implementasi provider LLM untuk server alternatif dengan standar kompatibilitas API OpenAI (OpenRouter, local LLM, dll)
class OpenAICompatibleLLMProvider:
    # Inisialisasi nama provider kustom, API Key, base URL, model, dan temperatur
    def __init__(self, *, name: str, api_key: str, model: str, base_url: str, temperature: float):
        # Nama provider
        self.name = name
        # Nama model target
        self.model = model
        # Nilai temperatur
        self.temperature = temperature
        # Mengimpor kelas OpenAI dari pustaka resmi openai
        from openai import OpenAI

        # Membuat instans client OpenAI mengarah ke base_url kustom
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    # Mengirim request chat completion ke server kompatibel OpenAI
    def generate(self, prompt: str) -> str:
        # Memanggil API chat completion secara sinkron
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        # Mengembalikan teks tanggapan pertama
        return completion.choices[0].message.content or ""


# Kelas Router untuk mengatur urutan prioritas pemanggilan model LLM dan mekanisme fallback otomatis
class LLMRouter:
    # Inisialisasi daftar provider aktif dan daftar provider yang tidak tersedia sejak awal
    def __init__(self, providers: list[LLMProvider], unavailable: list[ProviderFailure] | None = None):
        # Menyimpan daftar objek provider LLM
        self.providers = providers
        # Menyimpan daftar kegagalan provider yang tidak aktif
        self.unavailable = unavailable or []

    # Metode kelas (factory method) untuk merakit LLMRouter berdasarkan setelan di settings
    @classmethod
    def from_settings(cls, settings: Settings, is_evaluator: bool = False) -> "LLMRouter":
        # Inisialisasi list penampung provider sukses
        providers: list[LLMProvider] = []
        # Inisialisasi list penampung provider gagal
        unavailable: list[ProviderFailure] = []
        
        # Menentukan urutan provider: gunakan urutan khusus evaluator jika parameter is_evaluator diset True
        provider_order = settings.evaluator_llm_provider_order if is_evaluator else settings.llm_provider_order

        # Melakukan iterasi nama provider sesuai dengan urutan prioritas fallback di settings
        for provider_name in provider_order:
            try:
                # Mencoba merakit objek provider LLM menggunakan fungsi pembantu build_llm_provider
                provider = build_llm_provider(provider_name, settings, is_evaluator)
            # Menangkap kegagalan jika kunci API/URL/konfigurasi provider tersebut tidak lengkap
            except Exception as exc:
                # Catat kesalahan ke list unavailable
                unavailable.append(ProviderFailure(provider=provider_name, error=str(exc)))
                # Cetak log info pelewatan provider
                logger.info("Skipping unavailable LLM provider %s: %s", provider_name, exc)
                # Lanjutkan loop ke provider berikutnya
                continue
            # Masukkan provider yang sukses dirakit ke list aktif
            providers.append(provider)

        # Mengembalikan instans LLMRouter yang baru
        return cls(providers=providers, unavailable=unavailable)

    # Mengirimkan prompt ke seluruh daftar LLM provider aktif satu per satu sampai ada yang berhasil merespon
    def generate(self, prompt: str) -> GenerationResult:
        # Menyalin daftar kegagalan awal (termasuk yang tidak aktif sejak start)
        failures = list(self.unavailable)
        # Iterasi setiap provider aktif
        for provider in self.providers:
            try:
                # Memanggil metode generate pada provider
                answer = provider.generate(prompt)
                # Jika respons yang diterima kosong
                if not answer.strip():
                    # Lempar runtime error untuk memicu mekanisme fallback ke provider berikutnya
                    raise RuntimeError("Provider returned an empty response")
                # Jika sukses mendapatkan teks jawaban, segera kembalikan hasilnya
                return GenerationResult(answer=answer, provider_used=provider.model)
            # Menangkap error jika API provider tersebut sedang mati / kuota habis / timeout
            except Exception as exc:
                # Catat detail kesalahan provider
                failure = ProviderFailure(provider=provider.name, error=str(exc))
                # Masukkan ke list failures
                failures.append(failure)
                # Cetak log warning
                logger.warning("LLM provider %s failed: %s", provider.name, exc)
        # Jika seluruh provider dalam list aktif dan tidak aktif gagal mengembalikan jawaban
        raise AllLLMProvidersFailed(failures=failures)

    # Mengirimkan prompt dan memparsing jawaban langsung menjadi dictionary JSON Python
    def generate_json(self, prompt: str) -> dict[str, Any]:
        # Memanggil generate teks biasa
        result = self.generate(prompt)
        # Memparsing teks jawaban mentah menjadi objek JSON/dict
        return _parse_json(result.answer)


# Fungsi pembantu untuk memparsing teks jawaban LLM menjadi objek JSON valid
def _parse_json(text: str) -> dict[str, Any]:
    # Membersihkan whitespace di ujung string
    text = text.strip()
    # Mencari pola pembatas code block JSON (```json ... ```) pada respons teks LLM menggunakan regex
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    # Jika tag code block ditemukan
    if match:
        # Ambil teks di dalam block tersebut dan buang spasi ujungnya
        text = match.group(1).strip()
    try:
        # Mencoba memparsing string bersih menjadi JSON
        return json.loads(text)
    # Menangkap error jika teks jawaban tidak berformat JSON valid
    except json.JSONDecodeError:
        # Catat warning ke log
        logger.warning("Failed to parse JSON from LLM response: %s", text)
        # Mengembalikan dict kosong sebagai fallback aman
        return {}


# Fungsi pembantu untuk merakit objek class LLMProvider berdasarkan nama provider dan parameter config
def build_llm_provider(provider_name: str, settings: Settings, is_evaluator: bool = False) -> LLMProvider:
    # Membersihkan whitespace dan menormalkan ke huruf kecil
    provider_name = provider_name.strip().lower()
    
    # ── Perakitan Provider Gemini ─────────────────────────────────────────────
    if provider_name == "gemini":
        # Menentukan API Key dan Model: pilih konfigurasi evaluator jika parameter is_evaluator aktif
        api_key = (settings.evaluator_gemini_api_key or settings.gemini_api_key) if is_evaluator else settings.gemini_api_key
        model = (settings.evaluator_gemini_llm_model or settings.gemini_llm_model) if is_evaluator else settings.gemini_llm_model
        # Pastikan API key terisi
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing")
        # Mengembalikan objek GeminiLLMProvider
        return GeminiLLMProvider(api_key, model, settings.llm_temperature)
        
    # ── Perakitan Provider Groq ───────────────────────────────────────────────
    if provider_name == "groq":
        # Menentukan API Key dan Model untuk Groq
        api_key = (settings.evaluator_groq_api_key or settings.groq_api_key) if is_evaluator else settings.groq_api_key
        model = (settings.evaluator_groq_llm_model or settings.groq_llm_model) if is_evaluator else settings.groq_llm_model
        # Pastikan API key terisi
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")
        # Mengembalikan objek GroqLLMProvider
        return GroqLLMProvider(api_key, model, settings.llm_temperature)
        
    # ── Perakitan Provider OpenRouter ─────────────────────────────────────────
    if provider_name == "openrouter":
        # Menentukan API Key dan Model untuk OpenRouter
        api_key = (settings.evaluator_openrouter_api_key or settings.openrouter_api_key) if is_evaluator else settings.openrouter_api_key
        model = (settings.evaluator_openrouter_llm_model or settings.openrouter_llm_model) if is_evaluator else settings.openrouter_llm_model
        # Pastikan API key terisi
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")
        # Mengembalikan instans OpenAICompatibleLLMProvider dengan mengarah ke endpoint resmi OpenRouter
        return OpenAICompatibleLLMProvider(
            name="openrouter",
            api_key=api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1",
            temperature=settings.llm_temperature,
        )
        
    # ── Perakitan Provider Kompatibel OpenAI Umum ─────────────────────────────
    if provider_name == "openai_compatible":
        # Menentukan API Key, Base URL, dan Model untuk server kompatibel OpenAI kustom
        api_key = (settings.evaluator_openai_compatible_api_key or settings.openai_compatible_api_key) if is_evaluator else settings.openai_compatible_api_key
        base_url = (settings.evaluator_openai_compatible_base_url or settings.openai_compatible_base_url) if is_evaluator else settings.openai_compatible_base_url
        model = (settings.evaluator_openai_compatible_model or settings.openai_compatible_model) if is_evaluator else settings.openai_compatible_model
        
        # Memastikan seluruh parameter wajib terisi lengkap
        if not api_key:
            raise ValueError("OPENAI_COMPATIBLE_API_KEY is missing")
        if not base_url:
            raise ValueError("OPENAI_COMPATIBLE_BASE_URL is missing")
        if not model:
            raise ValueError("OPENAI_COMPATIBLE_MODEL is missing")
            
        # Mengembalikan instans OpenAICompatibleLLMProvider kustom
        return OpenAICompatibleLLMProvider(
            name="openai_compatible",
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=settings.llm_temperature,
        )
        
    # Lempar error jika nama provider tidak dikenali
    raise ValueError(f"Unsupported LLM provider: {provider_name}")
