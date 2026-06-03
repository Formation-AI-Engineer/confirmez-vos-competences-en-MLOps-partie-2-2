"""Tests des métadonnées du modèle et de son chargement unique."""

import joblib

from app import predictor


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "LGBMClassifier"
    assert body["n_features"] == 804
    assert body["decision_threshold"] == 0.49
    assert body["dataset_version"] == "4247222185ae"


def test_model_loaded_once(client, monkeypatch):
    """Le modèle est un singleton : un nouvel appel ne recharge pas depuis le disque."""
    # Le fixture `client` a déjà déclenché le chargement au démarrage (lifespan).
    calls = {"n": 0}
    real_load = joblib.load

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting_load)

    predictor.load_model()
    predictor.load_model()

    assert calls["n"] == 0  # aucun rechargement disque (modèle déjà en mémoire)
