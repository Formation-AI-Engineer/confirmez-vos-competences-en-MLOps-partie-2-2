"""Service de prédiction — chargement du modèle (singleton) et inférence.

Le modèle et ses métadonnées sont chargés **une seule fois** (au démarrage de
l'API via ``load_model``), puis réutilisés à chaque requête.
"""

import json

import joblib
import pandas as pd

from app import config

_model = None
_feature_names: list[str] | None = None
_meta: dict | None = None


def load_model():
    """Charge le modèle, la liste des features et les métadonnées (singleton)."""
    global _model, _feature_names, _meta
    if _model is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {config.MODEL_PATH}. "
                "Vérifier les artefacts du Projet 6 dans models/."
            )
        _model = joblib.load(config.MODEL_PATH)
        _feature_names = json.loads(config.FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
        _meta = json.loads(config.MODEL_META_PATH.read_text(encoding="utf-8"))
    return _model


def get_feature_names() -> list[str]:
    """Retourne la liste ordonnée des 804 features attendues."""
    load_model()
    return _feature_names


def _add_derived_features(features: dict) -> dict:
    """Calcule les features dérivées du Projet 6 à partir des valeurs brutes.

    Aujourd'hui : ``CREDIT_TERM`` = annuité / crédit — la feature la plus
    importante du modèle. Cela rend la saisie de montants bruts (API ou démo)
    cohérente avec l'entraînement. Une valeur fournie explicitement n'est pas
    écrasée.
    """
    f = dict(features)
    if "CREDIT_TERM" not in f and f.get("AMT_CREDIT") and f.get("AMT_ANNUITY") is not None:
        f["CREDIT_TERM"] = f["AMT_ANNUITY"] / f["AMT_CREDIT"]
    return f


def predict(features: dict) -> dict:
    """Prédit le risque de défaut pour un client.

    Aligne les features fournies sur les 804 colonnes attendues (ordre du modèle),
    les manquantes devenant NaN, puis applique le seuil de décision métier.
    """
    model = load_model()
    names = _feature_names
    features = _add_derived_features(features)

    # Alignement sur les 804 features attendues : manquantes -> NaN, inconnues ignorées.
    row = pd.DataFrame([features]).reindex(columns=names).astype("float64")
    n_received = sum(1 for k in features if k in set(names))

    proba = float(model.predict_proba(row)[0, 1])
    decision = "refusé" if proba >= config.DECISION_THRESHOLD else "accordé"

    return {
        "probability": round(proba, 4),
        "decision": decision,
        "threshold": config.DECISION_THRESHOLD,
        "n_features_received": n_received,
        "n_features_expected": len(names),
    }


def get_model_info() -> dict:
    """Retourne les métadonnées du modèle déployé."""
    load_model()
    return {
        "model_type": _meta["model_type"],
        "n_features": _meta["n_features"],
        "decision_threshold": config.DECISION_THRESHOLD,
        "dataset_version": _meta["dataset_version"],
        "train_size": _meta["train_size"],
    }
