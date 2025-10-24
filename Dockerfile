# Dockerfile para NASA Studies API con GGUF
FROM python:3.10-slim

# Variables de build
ARG DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero (cache de Docker)
COPY requirements.txt .

# Instalar dependencias Python básicas
RUN pip install --no-cache-dir -r requirements.txt

# Instalar llama-cpp-python con optimizaciones CPU
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install llama-cpp-python --no-cache-dir

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/models /app/data/processed /app/data/raw

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV GGUF_MODEL_PATH=/app/app/models/odr_model_q5_k_m.gguf

# Puerto para FastAPI
EXPOSE 80

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Comando por defecto
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]