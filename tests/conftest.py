"""Fixtures partagées — client de test API et payload d'exemple réel."""

import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Désactive la journalisation des prédictions par défaut en test : les tests qui
# la ciblent (test_monitoring.py) la réactivent sur une base SQLite temporaire.
os.environ.setdefault("MONITORING_ENABLED", "false")

from app.main import app  # noqa: E402  (après le réglage d'environnement ci-dessus)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def client():
    """Client de test FastAPI. Le `with` déclenche le lifespan (chargement modèle)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_features() -> dict:
    """Features du client 0 de l'échantillon de référence (valeurs réelles, NaN exclus)."""
    ref = pd.read_parquet(ROOT / "monitoring" / "reference_sample.parquet")
    row = ref.iloc[0].to_dict()
    return {k: float(v) for k, v in row.items() if pd.notna(v)}
