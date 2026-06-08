from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol
import json
import re

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    answer: str
    provider_used: str


@dataclass
class ProviderFailure:
    provider: str
    error: str


class AllLLMProvidersFailed(RuntimeError):
    def __init__(self, failures: list[ProviderFailure]):
        self.failures = failures
        super().__init__("All LLM providers failed")


class LLMProvider(Protocol):
    name: str

    def generate(self, prompt: str) -> str:
        ...


class GeminiLLMProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=api_key)
        self.types = types

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(temperature=self.temperature),
        )
        return getattr(response, "text", None) or ""


class GroqLLMProvider:
    name = "groq"

    def __init__(self, api_key: str, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        from groq import Groq

        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return completion.choices[0].message.content or ""


class OpenAICompatibleLLMProvider:
    def __init__(self, *, name: str, api_key: str, model: str, base_url: str, temperature: float):
        self.name = name
        self.model = model
        self.temperature = temperature
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return completion.choices[0].message.content or ""


class LLMRouter:
    def __init__(self, providers: list[LLMProvider], unavailable: list[ProviderFailure] | None = None):
        self.providers = providers
        self.unavailable = unavailable or []

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMRouter":
        providers: list[LLMProvider] = []
        unavailable: list[ProviderFailure] = []

        for provider_name in settings.llm_provider_order:
            try:
                provider = build_llm_provider(provider_name, settings)
            except Exception as exc:
                unavailable.append(ProviderFailure(provider=provider_name, error=str(exc)))
                logger.info("Skipping unavailable LLM provider %s: %s", provider_name, exc)
                continue
            providers.append(provider)

        return cls(providers=providers, unavailable=unavailable)

    def generate(self, prompt: str) -> GenerationResult:
        failures = list(self.unavailable)
        for provider in self.providers:
            try:
                answer = provider.generate(prompt)
                if not answer.strip():
                    raise RuntimeError("Provider returned an empty response")
                return GenerationResult(answer=answer, provider_used=provider.name)
            except Exception as exc:
                failure = ProviderFailure(provider=provider.name, error=str(exc))
                failures.append(failure)
                logger.warning("LLM provider %s failed: %s", provider.name, exc)
        raise AllLLMProvidersFailed(failures=failures)

    def generate_json(self, prompt: str) -> dict[str, Any]:
        result = self.generate(prompt)
        return _parse_json(result.answer)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from LLM response: %s", text)
        return {}


def build_llm_provider(provider_name: str, settings: Settings) -> LLMProvider:
    provider_name = provider_name.strip().lower()
    if provider_name == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is missing")
        return GeminiLLMProvider(settings.gemini_api_key, settings.gemini_llm_model, settings.llm_temperature)
    if provider_name == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing")
        return GroqLLMProvider(settings.groq_api_key, settings.groq_llm_model, settings.llm_temperature)
    if provider_name == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")
        return OpenAICompatibleLLMProvider(
            name="openrouter",
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_llm_model,
            base_url="https://openrouter.ai/api/v1",
            temperature=settings.llm_temperature,
        )
    if provider_name == "openai_compatible":
        if not settings.openai_compatible_api_key:
            raise ValueError("OPENAI_COMPATIBLE_API_KEY is missing")
        if not settings.openai_compatible_base_url:
            raise ValueError("OPENAI_COMPATIBLE_BASE_URL is missing")
        if not settings.openai_compatible_model:
            raise ValueError("OPENAI_COMPATIBLE_MODEL is missing")
        return OpenAICompatibleLLMProvider(
            name="openai_compatible",
            api_key=settings.openai_compatible_api_key,
            model=settings.openai_compatible_model,
            base_url=settings.openai_compatible_base_url,
            temperature=settings.llm_temperature,
        )
    raise ValueError(f"Unsupported LLM provider: {provider_name}")
