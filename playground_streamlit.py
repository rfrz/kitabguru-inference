from __future__ import annotations

import json
from typing import Any

import httpx
import streamlit as st


DEFAULT_API_BASE_URL = ""
REQUEST_TIMEOUT = 30.0
CHAT_TIMEOUT = 900.0
IMPORT_TIMEOUT = 600.0


def normalize_base_url(value: str) -> str:
    cleaned = value.strip().rstrip("/")
    return cleaned or DEFAULT_API_BASE_URL


def api_url(base_url: str, path: str) -> str:
    return f"{normalize_base_url(base_url)}{path}"


def response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or response.reason_phrase

    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        message = detail.get("message")
        if message:
            extra = {key: value for key, value in detail.items() if key != "message"}
            if extra:
                return f"{message}\n\n{json.dumps(extra, indent=2, ensure_ascii=False)}"
            return message
    return json.dumps(detail, indent=2, ensure_ascii=False)


def request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    timeout: float = REQUEST_TIMEOUT,
    **kwargs: Any,
) -> tuple[Any | None, str | None]:
    try:
        response = httpx.request(method, api_url(base_url, path), timeout=timeout, follow_redirects=True, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"Tidak bisa menghubungi backend: {exc}"

    if response.is_success:
        if response.status_code == 204:
            return None, None
        try:
            return response.json(), None
        except ValueError:
            return None, None

    return None, f"{response.status_code} {response.reason_phrase}: {response_detail(response)}"


def fetch_documents(base_url: str, headers: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], str | None]:
    payload, error = request_json("GET", base_url, "/api/documents", headers=headers)
    if error:
        return [], error
    return payload or [], None


def document_label(document: dict[str, Any]) -> str:
    title = document.get("title") or "Untitled"
    author = document.get("author")
    chunks = document.get("total_chunks", 0)
    if author:
        return f"{title} - {author} ({chunks} chunks)"
    return f"{title} ({chunks} chunks)"


def show_source(source: dict[str, Any], index: int) -> None:
    metadata = source.get("metadata") or {}
    title = metadata.get("title") or "Dokumen"
    chapter = metadata.get("chapter")
    chunk_index = metadata.get("chunk_index")
    distance = source.get("distance")

    details = []
    if chapter is not None:
        details.append(f"chapter {chapter}")
    if chunk_index is not None:
        details.append(f"chunk {chunk_index}")
    if distance is not None:
        details.append(f"distance {distance:.4f}")
    suffix = f" - {', '.join(details)}" if details else ""

    with st.expander(f"Sumber {index}: {title}{suffix}"):
        st.write(source.get("document") or "")
        if metadata:
            st.json(metadata)


def render_health(base_url: str, headers: dict[str, str] | None = None) -> None:
    payload, error = request_json("GET", base_url, "/health", timeout=5.0, headers=headers)
    if error:
        st.error(error)
        return
    status = payload.get("status") if isinstance(payload, dict) else None
    if status == "ok":
        st.success("Backend online")
    else:
        st.warning("Backend merespons, tapi status health tidak dikenali.")


def render_document_manager(base_url: str, documents: list[dict[str, Any]], headers: dict[str, str] | None = None) -> None:
    st.subheader("Import EPUB")
    with st.form("import_document_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("File EPUB", type=["epub"], accept_multiple_files=False)
        title = st.text_input("Title override", placeholder="Opsional")
        author = st.text_input("Author override", placeholder="Opsional")
        submitted = st.form_submit_button("Import")

    if submitted:
        if uploaded_file is None:
            st.warning("Pilih file EPUB dulu.")
        else:
            data = {}
            if title.strip():
                data["title"] = title.strip()
            if author.strip():
                data["author"] = author.strip()
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/epub+zip",
                )
            }
            with st.spinner("Mengimport dan membuat embedding..."):
                payload, error = request_json(
                    "POST",
                    base_url,
                    "/api/documents/import",
                    data=data,
                    files=files,
                    timeout=IMPORT_TIMEOUT,
                    headers=headers,
                )
            if error:
                st.error(error)
            else:
                st.success(f"Berhasil import: {payload.get('title', uploaded_file.name)}")
                st.rerun()

    st.subheader("Dokumen")
    if not documents:
        st.info("Belum ada dokumen.")
        return

    for document in documents:
        is_current = document.get("is_embedding_current", False)
        status_text = "current" if is_current else "stale"
        with st.container(border=True):
            top_left, top_right = st.columns([0.78, 0.22])
            with top_left:
                st.markdown(f"**{document_label(document)}**")
                st.caption(
                    f"book_id: {document.get('book_id')} | "
                    f"embedding: {document.get('embedding_provider')} / "
                    f"{document.get('embedding_model')} | {status_text}"
                )
            with top_right:
                delete_key = f"delete_{document.get('book_id')}"
                if st.button("Delete", key=delete_key, use_container_width=True):
                    with st.spinner("Menghapus dokumen..."):
                        _, error = request_json(
                            "DELETE",
                            base_url,
                            f"/api/documents/{document.get('book_id')}",
                            headers=headers,
                        )
                    if error:
                        st.error(error)
                    else:
                        st.success("Dokumen dihapus.")
                        st.rerun()


def render_chat(base_url: str, documents: list[dict[str, Any]], headers: dict[str, str] | None = None) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    filter_options = ["Semua dokumen"] + [document_label(document) for document in documents]
    selected_label = st.selectbox("Filter dokumen", filter_options)
    selected_book_id = None
    if selected_label != "Semua dokumen":
        selected_index = filter_options.index(selected_label) - 1
        selected_book_id = documents[selected_index].get("book_id")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("provider_used"):
                st.caption(f"Provider: {message['provider_used']}")
            for index, source in enumerate(message.get("sources", []), start=1):
                show_source(source, index)

    query = st.chat_input("Tulis pertanyaan untuk inference...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    request_payload: dict[str, Any] = {"query": query}
    if selected_book_id:
        request_payload["book_filter"] = selected_book_id

    with st.chat_message("assistant"):
        with st.spinner("Mengambil konteks dan meminta jawaban LLM..."):
            payload, error = request_json(
                "POST",
                base_url,
                "/api/chat",
                json=request_payload,
                timeout=CHAT_TIMEOUT,
                headers=headers,
            )
        if error:
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "sources": []})
            return

        answer = payload.get("answer", "")
        provider_used = payload.get("provider_used", "unknown")
        sources = payload.get("sources", [])
        st.write(answer)
        st.caption(f"Provider: {provider_used}")
        for index, source in enumerate(sources, start=1):
            show_source(source, index)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "provider_used": provider_used,
            "sources": sources,
        }
    )


def main() -> None:
    st.set_page_config(page_title="KitabGuru Playground", layout="wide")
    st.title("KitabGuru Inference Playground")

    with st.sidebar:
        st.header("Backend")
        base_url = st.text_input("API Base URL", value=DEFAULT_API_BASE_URL)
        auth_token = st.text_input("Auth Token (Bearer)", value="", type="password", help="Opsional, untuk private endpoint (misal HF Spaces)")
        normalized_base_url = normalize_base_url(base_url)
        
        headers = {}
        if auth_token.strip():
            headers["Authorization"] = f"Bearer {auth_token.strip()}"
            
        render_health(normalized_base_url, headers)
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    documents, documents_error = fetch_documents(normalized_base_url, headers)
    if documents_error:
        st.warning(f"Gagal mengambil daftar dokumen: {documents_error}")

    chat_tab, documents_tab = st.tabs(["Chat", "Dokumen"])
    with chat_tab:
        render_chat(normalized_base_url, documents, headers)
    with documents_tab:
        render_document_manager(normalized_base_url, documents, headers)


if __name__ == "__main__":
    main()
