# Space Docker du dashboard de monitoring — Streamlit, dépendances via uv + pyproject
# (même outillage que l'API). Lit Neon via le secret DATABASE_URL.
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Dépendances (couche cachée tant que pyproject/lock ne changent pas).
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Application + référence pré-scorée pour le calcul de drift.
COPY app.py reference_scored.parquet ./

EXPOSE 7860

CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.port", "7860", "--server.address", "0.0.0.0"]
