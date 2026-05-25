import pytest

from app.providers.llm import AllLLMProvidersFailed, LLMRouter


class FakeProvider:
    def __init__(self, name, answer=None, error=None):
        self.name = name
        self.answer = answer
        self.error = error

    def generate(self, prompt: str) -> str:
        if self.error:
            raise self.error
        return self.answer


def test_llm_router_returns_first_successful_provider():
    router = LLMRouter(
        providers=[
            FakeProvider("gemini", error=RuntimeError("rate limited")),
            FakeProvider("groq", answer="jawaban"),
        ]
    )

    result = router.generate("prompt")

    assert result.answer == "jawaban"
    assert result.provider_used == "groq"


def test_llm_router_raises_when_all_providers_fail():
    router = LLMRouter(
        providers=[
            FakeProvider("gemini", error=RuntimeError("down")),
            FakeProvider("groq", answer=""),
        ]
    )

    with pytest.raises(AllLLMProvidersFailed) as exc:
        router.generate("prompt")

    assert [failure.provider for failure in exc.value.failures] == ["gemini", "groq"]
