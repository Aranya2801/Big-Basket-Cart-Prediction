# ╔══════════════════════════════════════════════════════════════════╗
# ║         BigBasket Cart Prediction — Multi-Stage Dockerfile       ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Stage 1: Base ────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="Aranya2801 <github.com/Aranya2801>"
LABEL description="BigBasket Cart Prediction — AI-Powered Grocery Intelligence"
LABEL version="2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Stage 2: Dependencies ─────────────────────────────────────────────
FROM base AS dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Stage 3: Application ──────────────────────────────────────────────
FROM dependencies AS app

COPY . .

# Generate dataset if not present
RUN python data/synthetic/generate_dataset.py

# Pre-train models
RUN python train_pipeline.py --data_dir data/raw --output_dir models || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000 8501

# Default: run both API + dashboard via supervisor or entrypoint
CMD ["sh", "-c", "python src/api/app.py & streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]
