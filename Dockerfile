# Image de l'API de scoring crédit — FastAPI + LightGBM (Projet 8, MLOps 2/2)
FROM python:3.10-slim

# libgomp1 : runtime OpenMP requis par LightGBM à l'exécution.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Dépendances (uniquement API + modèle, pas le monitoring) — couche cachée.
COPY pyproject.toml README.md ./
COPY app/ ./app/
RUN pip install --no-cache-dir .

# Artefacts du modèle (copiés après l'install pour préserver le cache de couches).
COPY models/ ./models/

EXPOSE 7860

# Vérifie l'état de l'API via l'endpoint /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
