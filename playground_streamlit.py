# Mengaktifkan evaluasi tipe data bertunda (postponed evaluation of annotations) agar kompatibel ke belakang
from __future__ import annotations

# Mengimpor modul json untuk pengolahan string JSON respons backend
import json
# Mengimpor Any untuk type-hinting python
from typing import Any

# Mengimpor modul httpx untuk melakukan request HTTP ke server backend FastAPI
import httpx
# Mengimpor modul streamlit untuk membangun antarmuka web playground RAG
import streamlit as st


# URL dasar default untuk API backend (kosong berarti menggunakan URL relatif / saat ini)
DEFAULT_API_BASE_URL = ""
# Timeout default untuk request HTTP biasa (30 detik)
REQUEST_TIMEOUT = 30.0
# Timeout panjang untuk request chat RAG karena membutuhkan generasi LLM (15 menit)
CHAT_TIMEOUT = 900.0
# Timeout panjang untuk proses import file EPUB dan pembuatan embedding (10 menit)
IMPORT_TIMEOUT = 600.0


# Merapikan format string URL dasar API agar konsisten tanpa tanda slash di ujung
def normalize_base_url(value: str) -> str:
    # Membuang spasi kosong dan karakter slash penutup di ujung kanan
    cleaned = value.strip().rstrip("/")
    # Mengembalikan URL yang bersih (atau default URL jika kosong)
    return cleaned or DEFAULT_API_BASE_URL


# Merakit URL lengkap endpoint API berdasarkan base URL dan segment path target
def api_url(base_url: str, path: str) -> str:
    return f"{normalize_base_url(base_url)}{path}"


# Mengekstrak detail pesan kesalahan (error detail) dari objek respons HTTP
def response_detail(response: httpx.Response) -> str:
    try:
        # Mencoba membaca body respons dalam format JSON
        payload = response.json()
    # Jika respons bukan JSON, kembalikan teks polos atau frasa status HTTP
    except ValueError:
        return response.text or response.reason_phrase

    # Mengambil detail error: cari field 'detail' jika payload berupa dictionary
    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    # Jika detail berbentuk string polos, langsung kembalikan
    if isinstance(detail, str):
        return detail
    # Jika detail berupa sub-dictionary
    if isinstance(detail, dict):
        # Ambil field 'message' jika ada
        message = detail.get("message")
        if message:
            # Mengumpulkan field tambahan selain 'message'
            extra = {key: value for key, value in detail.items() if key != "message"}
            # Jika ada metadata tambahan, gabungkan ke teks pesan dalam bentuk string JSON terformat
            if extra:
                return f"{message}\n\n{json.dumps(extra, indent=2, ensure_ascii=False)}"
            # Kembalikan hanya teks pesan utama
            return message
    # Default: kembalikan representasi string JSON lengkap dari objek detail
    return json.dumps(detail, indent=2, ensure_ascii=False)


# Fungsi pembantu untuk mengirim request HTTP dan memparsing hasil JSON
def request_json(
    # Metode HTTP (GET/POST/DELETE, dll)
    method: str,
    # Base URL API
    base_url: str,
    # Path endpoint target
    path: str,
    # *, menandai parameter setelahnya wajib dikirim sebagai keyword-arguments
    *,
    # Batas waktu timeout request
    timeout: float = REQUEST_TIMEOUT,
    # Argumen tambahan HTTPX lainnya (headers, json, data, dll)
    **kwargs: Any,
) -> tuple[Any | None, str | None]:
    try:
        # Mengirim request HTTP menggunakan client HTTPX
        response = httpx.request(method, api_url(base_url, path), timeout=timeout, follow_redirects=True, **kwargs)
    # Menangkap kegagalan jaringan atau server offline
    except httpx.HTTPError as exc:
        # Mengembalikan None untuk payload dan string error detail
        return None, f"Tidak bisa menghubungi backend: {exc}"

    # Jika respons status sukses (200-299)
    if response.is_success:
        # Status 204 No Content menandakan sukses tanpa kembalian data
        if response.status_code == 204:
            return None, None
        try:
            # Mengembalikan payload JSON dan None error
            return response.json(), None
        # Menangkap error jika respons sukses tapi isinya bukan JSON valid
        except ValueError:
            return None, None

    # Mengembalikan None payload dan detail error jika status HTTP gagal (400, 500, dll)
    return None, f"{response.status_code} {response.reason_phrase}: {response_detail(response)}"


# Mengambil daftar seluruh dokumen buku yang terdaftar di database backend
def fetch_documents(base_url: str, headers: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], str | None]:
    # Mengirim request GET ke endpoint '/api/documents'
    payload, error = request_json("GET", base_url, "/api/documents", headers=headers)
    # Jika terjadi error
    if error:
        # Kembalikan list kosong dan detail error
        return [], error
    # Mengembalikan payload list dokumen
    return payload or [], None


# Menyusun teks label penamaan dokumen untuk ditampilkan di komponen pilihan selectbox Streamlit
def document_label(document: dict[str, Any]) -> str:
    # Membaca judul buku (default 'Untitled' jika kosong)
    title = document.get("title") or "Untitled"
    # Membaca nama penulis
    author = document.get("author")
    # Membaca total jumlah potongan chunk
    chunks = document.get("total_chunks", 0)
    # Jika nama penulis tertera
    if author:
        # Gabungkan judul, penulis, dan jumlah chunk
        return f"{title} - {author} ({chunks} chunks)"
    # Gabungkan judul dan jumlah chunk
    return f"{title} ({chunks} chunks)"


# Menampilkan data sumber kutipan dokumen di dalam widget expander collapse Streamlit
def show_source(source: dict[str, Any], index: int) -> None:
    # Membaca metadata kutipan
    metadata = source.get("metadata") or {}
    # Membaca judul dokumen
    title = metadata.get("title") or "Dokumen"
    # Membaca nomor bab
    chapter = metadata.get("chapter")
    # Membaca indeks potongan chunk
    chunk_index = metadata.get("chunk_index")
    # Membaca skor jarak kemiripan
    distance = source.get("distance")

    # Inisialisasi list detail rincian metadata
    details = []
    # Jika nomor bab tersedia
    if chapter is not None:
        details.append(f"chapter {chapter}")
    # Jika indeks chunk tersedia
    if chunk_index is not None:
        details.append(f"chunk {chunk_index}")
    # Jika skor kemiripan tersedia
    if distance is not None:
        # Format nilai float dengan 4 angka di belakang koma
        details.append(f"distance {distance:.4f}")
    # Menyusun string detail rincian sebagai akhiran label expander
    suffix = f" - {', '.join(details)}" if details else ""

    # Membuat komponen expander kolaps Streamlit
    with st.expander(f"Sumber {index}: {title}{suffix}"):
        # Menulis isi teks potongan kutipan dokumen
        st.write(source.get("document") or "")
        # Jika metadata tidak kosong, tampilkan data mentah JSON metadatanya
        if metadata:
            st.json(metadata)


# Memeriksa status kesehatan server backend dan menampilkan hasilnya di sidebar
def render_health(base_url: str, headers: dict[str, str] | None = None) -> None:
    # Mengirim request GET ke endpoint '/health' dengan timeout pendek 5 detik
    payload, error = request_json("GET", base_url, "/health", timeout=5.0, headers=headers)
    # Jika server mati / gagal dihubungi
    if error:
        # Tampilkan kotak pesan error merah
        st.error(error)
        return
    # Membaca status keaktifan
    status = payload.get("status") if isinstance(payload, dict) else None
    # Jika status bernilai 'ok'
    if status == "ok":
        # Tampilkan kotak pesan sukses hijau online
        st.success("Backend online")
    # Jika respons sukses tapi isi statusnya berbeda
    else:
        # Tampilkan kotak peringatan kuning warning
        st.warning("Backend merespons, tapi status health tidak dikenali.")


# Merender tab antarmuka pengelolaan impor dan penghapusan buku dokumen EPUB
def render_document_manager(base_url: str, documents: list[dict[str, Any]], headers: dict[str, str] | None = None) -> None:
    st.subheader("Import EPUB")
    # Membuat form input di Streamlit untuk pendaftaran berkas EPUB
    with st.form("import_document_form", clear_on_submit=True):
        # Widget pengunggah berkas khusus ekstensi '.epub'
        uploaded_file = st.file_uploader("File EPUB", type=["epub"], accept_multiple_files=False)
        # Input teks opsional untuk mengganti judul buku asli
        title = st.text_input("Title override", placeholder="Opsional")
        # Input teks opsional untuk mengganti nama penulis asli
        author = st.text_input("Author override", placeholder="Opsional")
        # Tombol submit form
        submitted = st.form_submit_button("Import")

    # Jika admin mengklik tombol submit Import
    if submitted:
        # Memastikan berkas EPUB telah dipilih
        if uploaded_file is None:
            # Tampilkan pesan peringatan kuning
            st.warning("Pilih file EPUB dulu.")
        else:
            # Menyiapkan payload form teks
            data = {}
            if title.strip():
                data["title"] = title.strip()
            if author.strip():
                data["author"] = author.strip()
            # Mempersiapkan payload file biner multipart untuk diunggah
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/epub+zip",
                )
            }
            # Menampilkan spinner animasi loading proses pemrosesan RAG
            with st.spinner("Mengimport dan membuat embedding..."):
                # Mengirim request POST impor dokumen ke backend
                payload, error = request_json(
                    "POST",
                    base_url,
                    "/api/documents/import",
                    data=data,
                    files=files,
                    timeout=IMPORT_TIMEOUT,
                    headers=headers,
                )
            # Jika proses impor gagal
            if error:
                # Tampilkan kotak error merah
                st.error(error)
            # Jika sukses
            else:
                # Tampilkan pesan sukses hijau
                st.success(f"Berhasil import: {payload.get('title', uploaded_file.name)}")
                # Memuat ulang halaman Streamlit agar daftar dokumen diperbarui
                st.rerun()

    st.subheader("Dokumen")
    # Jika tidak ada dokumen buku yang terdaftar
    if not documents:
        # Tampilkan kotak informasi biru
        st.info("Belum ada dokumen.")
        return

    # Iterasi setiap dokumen buku yang terdaftar
    for document in documents:
        # Memeriksa apakah model embedding dokumen cocok dengan setelan sistem saat ini
        is_current = document.get("is_embedding_current", False)
        # Menentukan string penanda status model
        status_text = "current" if is_current else "stale"
        # Membuat kotak kontainer untuk tata letak baris dokumen
        with st.container(border=True):
            # Membagi kolom baris menjadi 2 bagian (kiri lebar untuk info, kanan kecil untuk tombol Delete)
            top_left, top_right = st.columns([0.78, 0.22])
            with top_left:
                # Menampilkan label penamaan dokumen
                st.markdown(f"**{document_label(document)}**")
                # Menampilkan detail ID buku dan nama model embedding
                st.caption(
                    f"book_id: {document.get('book_id')} | "
                    f"embedding: {document.get('embedding_provider')} / "
                    f"{document.get('embedding_model')} | {status_text}"
                )
            with top_right:
                # Menyiapkan key unik untuk tombol delete di baris ini
                delete_key = f"delete_{document.get('book_id')}"
                # Menampilkan tombol Delete berwarna merah jika diklik
                if st.button("Delete", key=delete_key, use_container_width=True):
                    # Menampilkan spinner animasi loading hapus
                    with st.spinner("Menghapus dokumen..."):
                        # Mengirim request DELETE ke API backend
                        _, error = request_json(
                            "DELETE",
                            base_url,
                            f"/api/documents/{document.get('book_id')}",
                            headers=headers,
                        )
                    # Jika gagal menghapus
                    if error:
                        st.error(error)
                    # Jika sukses
                    else:
                        st.success("Dokumen dihapus.")
                        # Memuat ulang antarmuka
                        st.rerun()


# Merender antarmuka obrolan tanya jawab RAG (Chat UI)
def render_chat(base_url: str, documents: list[dict[str, Any]], headers: dict[str, str] | None = None) -> None:
    # Memastikan list histori pesan diinisialisasi dalam Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Menyusun opsi filter dokumen (default 'Semua dokumen' disusul daftar nama buku)
    filter_options = ["Semua dokumen"] + [document_label(document) for document in documents]
    # Menampilkan dropdown selectbox pilihan filter buku
    selected_label = st.selectbox("Filter dokumen", filter_options)
    # Menyimpan ID buku yang difilter
    selected_book_id = None
    # Jika memilih buku tertentu
    if selected_label != "Semua dokumen":
        # Ambil indeks buku
        selected_index = filter_options.index(selected_label) - 1
        # Ambil book_id dari daftar dokumen
        selected_book_id = documents[selected_index].get("book_id")

    # Menampilkan seluruh riwayat percakapan yang tersimpan di session state ke layar chat
    for message in st.session_state.messages:
        # Membuat balon chat sesuai dengan perannya (user / assistant)
        with st.chat_message(message["role"]):
            # Menulis konten teks percakapan
            st.write(message["content"])
            # Jika ada informasi model LLM yang digunakan, tampilkan keterangan di bawah chat
            if message.get("provider_used"):
                st.caption(f"Provider: {message['provider_used']}")
            # Menampilkan daftar link rujukan sumber dokumen jika pesan memiliki sitasi
            for index, source in enumerate(message.get("sources", []), start=1):
                show_source(source, index)

    # Widget input chat baris bawah Streamlit untuk mengetik pertanyaan
    query = st.chat_input("Tulis pertanyaan untuk inference...")
    # Jika pengguna belum menginput atau menekan enter
    if not query:
        return

    # Menyimpan pesan pertanyaan user ke list session state
    st.session_state.messages.append({"role": "user", "content": query})
    # Menampilkan balon pesan user baru secara instan ke layar chat
    with st.chat_message("user"):
        st.write(query)

    # Mempersiapkan payload JSON request
    request_payload: dict[str, Any] = {"query": query}
    # Menyisipkan filter buku jika diaktifkan
    if selected_book_id:
        request_payload["book_filter"] = selected_book_id

    # Membuat balon pesan assistant dengan animasi loading
    with st.chat_message("assistant"):
        # Menampilkan spinner animasi loading RAG
        with st.spinner("Mengambil konteks dan meminta jawaban LLM..."):
            # Mengirim request POST tanya jawab ke API backend '/api/chat'
            payload, error = request_json(
                "POST",
                base_url,
                "/api/chat",
                json=request_payload,
                timeout=CHAT_TIMEOUT,
                headers=headers,
            )
        # Jika request gagal (galat jaringan/error server)
        if error:
            # Tampilkan pesan error di balon chat
            st.error(error)
            # Simpan pesan error di histori chat agar antarmuka tidak rusak
            st.session_state.messages.append({"role": "assistant", "content": error, "sources": []})
            return

        # Mengambil data jawaban, nama provider LLM, dan daftar sumber dari JSON respons backend
        answer = payload.get("answer", "")
        provider_used = payload.get("provider_used", "unknown")
        sources = payload.get("sources", [])
        # Menulis teks jawaban RAG di balon chat
        st.write(answer)
        # Menampilkan nama model LLM penjawab
        st.caption(f"Provider: {provider_used}")
        # Menampilkan daftar expander rujukan sumber kutipan
        for index, source in enumerate(sources, start=1):
            show_source(source, index)

    # Menyimpan pesan balasan asisten ke list session state agar menetap di layar chat
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "provider_used": provider_used,
            "sources": sources,
        }
    )


# Fungsi entri utama aplikasi Streamlit Playground
def main() -> None:
    # Mengeset judul tab browser dan tipe layout lebar (wide)
    st.set_page_config(page_title="KitabGuru Playground", layout="wide")
    # Menulis judul header halaman web
    st.title("KitabGuru Inference Playground")

    # Membuat antarmuka sidebar di sebelah kiri
    with st.sidebar:
        st.header("Backend")
        # Input teks URL dasar server API
        base_url = st.text_input("API Base URL", value=DEFAULT_API_BASE_URL)
        # Input kata sandi untuk otentikasi token bearer (opsional)
        auth_token = st.text_input("Auth Token (Bearer)", value="", type="password", help="Opsional, untuk private endpoint (misal HF Spaces)")
        # Menormalkan base URL inputan
        normalized_base_url = normalize_base_url(base_url)
        
        # Menyiapkan header Authorization Bearer jika token diisi
        headers = {}
        if auth_token.strip():
            headers["Authorization"] = f"Bearer {auth_token.strip()}"
            
        # Merender status online/kesehatan backend di sidebar
        render_health(normalized_base_url, headers)
        # Menampilkan tombol Clear Chat untuk mereset riwayat percakapan di layar
        if st.button("Clear chat", use_container_width=True):
            # Mengosongkan list messages
            st.session_state.messages = []
            # Memuat ulang halaman Streamlit
            st.rerun()

    # Memuat daftar buku dokumen dari backend
    documents, documents_error = fetch_documents(normalized_base_url, headers)
    # Jika gagal mengambil daftar dokumen
    if documents_error:
        # Tampilkan kotak peringatan kuning warning
        st.warning(f"Gagal mengambil daftar dokumen: {documents_error}")

    # Membagi layout utama menjadi 2 tab navigasi: 'Chat' dan 'Dokumen'
    chat_tab, documents_tab = st.tabs(["Chat", "Dokumen"])
    # Tab Chat
    with chat_tab:
        # Memanggil render antarmuka tanya jawab
        render_chat(normalized_base_url, documents, headers)
    # Tab Dokumen
    with documents_tab:
        # Memanggil render pengelola dokumen buku
        render_document_manager(normalized_base_url, documents, headers)


# Jalankan fungsi main jika file dieksekusi secara langsung oleh python interpreter
if __name__ == "__main__":
    main()
