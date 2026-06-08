# Menggunakan image Python resmi
FROM python:3.14-slim

# Mencegah Python membuat file .pyc dan memastikan log terminal langsung muncul
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Buat direktori kerja
WORKDIR /app

# Install system dependencies (jika ada)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements (pyproject.toml)
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir .

# Copy seluruh kode sumber ke dalam container
COPY . .

# Hugging Face Spaces Persistent Storage biasanya di mount di /data
# Jadi kita buat direktorinya (opsional, karena HF akan nge-mount otomatis jika diaktifkan)
RUN mkdir -p /data

# Berikan hak akses (permissions) ke direktori /data dan /app agar user 'user' (yang dipakai HF) bisa menulis
RUN useradd -m -u 1000 user
RUN chown -R user:user /app /data

# Pindah ke user non-root
USER user

# Hugging Face Spaces mengharuskan aplikasi berjalan di port 7860
EXPOSE 7860

# Command untuk menjalankan FastAPI (Inference Engine) di port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
