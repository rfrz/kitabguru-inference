from app.config import Settings
from app.providers.embeddings import build_embedding_profile, prefixed_for_e5


def test_embedding_fingerprint_is_stable_for_same_config():
    settings = Settings(
        embedding_provider="huggingface",
        hf_embedding_model="intfloat/multilingual-e5-large",
        hf_api_key="token",
    )

    first = build_embedding_profile(settings)
    second = build_embedding_profile(settings)

    assert first.fingerprint == second.fingerprint
    assert first.dimension == 1024
    assert first.behavior == "e5-query-passage-prefix"


def test_embedding_fingerprint_changes_when_model_changes():
    old = build_embedding_profile(
        Settings(
            embedding_provider="huggingface",
            hf_embedding_model="intfloat/multilingual-e5-large",
            hf_api_key="token",
        )
    )
    new = build_embedding_profile(
        Settings(
            embedding_provider="huggingface",
            hf_embedding_model="intfloat/multilingual-e5-base",
            hf_api_key="token",
        )
    )

    assert old.fingerprint != new.fingerprint


def test_e5_prefixes_query_and_document_text():
    model = "intfloat/multilingual-e5-large"

    assert prefixed_for_e5(model, "Apa itu tauhid?", is_query=True) == "query: Apa itu tauhid?"
    assert prefixed_for_e5(model, "النص العربي", is_query=False) == "passage: النص العربي"
    assert prefixed_for_e5(model, "query: Apa itu tauhid?", is_query=True) == "query: Apa itu tauhid?"
