"""API de scoring crédit — Prêt à Dépenser (MLOps Partie 2/2).

Expose le modèle LightGBM (804 features, seuil métier 0.49) du Projet 6 :
- ``GET /health``     : vérification de l'état de l'API
- ``POST /predict``   : probabilité de défaut + décision (accordé/refusé)
- ``GET /model/info`` : métadonnées du modèle déployé
"""

import time
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app import monitoring, predictor
from app.demo import build_demo
from app.schemas import ModelInfo, PredictionInput, PredictionOutput


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Au démarrage : charge le modèle une seule fois + initialise le stockage des logs."""
    predictor.load_model()
    monitoring.init_db()
    yield


app = FastAPI(
    title="API de scoring crédit — Prêt à Dépenser",
    description=(
        "Déploiement du modèle de scoring crédit (Projet 6) — MLOps Partie 2/2.\n\n"
        "**Modèle** : LGBMClassifier (LightGBM, 804 features), seuil métier **0.49**.\n\n"
        "La classe positive correspond au **défaut de paiement** : une probabilité "
        "supérieure ou égale au seuil entraîne un **refus** du crédit."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    """Redirige vers la documentation Swagger."""
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Health check", tags=["Santé"])
def health():
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Prédire le risque de défaut d'un client",
    tags=["Prédiction"],
)
def predict(payload: PredictionInput) -> PredictionOutput:
    """Retourne la probabilité de défaut et la décision métier (seuil 0.49).

    Chaque appel est journalisé (latence, statut, sortie, features) pour le
    monitoring de production — y compris les erreurs d'inférence.
    """
    start = time.perf_counter()
    try:
        result = predictor.predict(payload.features)
    except Exception as exc:  # erreur d'inférence -> 500
        monitoring.log_prediction(
            features=predictor.add_derived_features(payload.features),
            latency_ms=(time.perf_counter() - start) * 1000,
            http_status=500,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {exc}") from exc
    monitoring.log_prediction(
        features=predictor.add_derived_features(payload.features),
        latency_ms=(time.perf_counter() - start) * 1000,
        http_status=200,
        result=result,
    )
    return PredictionOutput(**result)


@app.get(
    "/model/info",
    response_model=ModelInfo,
    summary="Métadonnées du modèle",
    tags=["Modèle"],
)
def model_info() -> ModelInfo:
    """Retourne les métadonnées du modèle déployé."""
    return ModelInfo(**predictor.get_model_info())


# Interface de démonstration Gradio, montée sous /demo (même process que l'API).
app = gr.mount_gradio_app(app, build_demo(), path="/demo")
