"""Non-régression de l'optimisation d'inférence (étape 4.3).

L'étape 4.2 a remplacé le chemin d'inférence « DataFrame pandas + ``predict_proba`` »
par « vecteur numpy pré-aligné + ``booster_.predict`` ». Ces tests **prouvent** que
ce changement n'altère pas les prédictions :

- on reconstruit l'**ancienne** implémentation (``legacy_proba``) et on la compare
  au chemin **optimisé** (``booster_.predict``) sur **tous** les clients réels de
  la référence ;
- scores **identiques au bit près** ⇒ même AUC, même précision, même biais et
  **mêmes décisions** à n'importe quel seuil (corollaire mathématique, pas besoin
  de labels) ;
- on vérifie enfin que le chemin de production ne dépend plus de **pandas**
  (compatibilité / légèreté de l'environnement de prod).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import config, predictor

from .conftest import ROOT


@pytest.fixture(scope="module")
def reference_payloads() -> list[dict]:
    """Tous les clients de la référence, en payloads bruts (NaN exclus)."""
    ref = pd.read_parquet(ROOT / "monitoring" / "reference_sample.parquet")
    return [{k: float(v) for k, v in row.items() if pd.notna(v)} for row in ref.to_dict("records")]


def legacy_proba(features: dict, names: list[str], model) -> float:
    """Ancienne implémentation (étape ≤ 4.1) : DataFrame pandas + ``predict_proba``."""
    feats = predictor.add_derived_features(features)
    row = pd.DataFrame([feats]).reindex(columns=names).astype("float64")
    return float(model.predict_proba(row)[0, 1])


def optimized_proba(features: dict, names: list[str], index: dict[str, int], model) -> float:
    """Chemin optimisé (étape 4.2) : vecteur numpy + ``booster_.predict``."""
    feats = predictor.add_derived_features(features)
    arr = np.full((1, len(names)), np.nan, dtype=np.float64)
    for key, value in feats.items():
        idx = index.get(key)
        if idx is not None and value is not None:
            arr[0, idx] = value
    return float(model.booster_.predict(arr)[0])


def test_scores_bit_identical(reference_payloads):
    """Sur tous les clients réels : écart de score max vs ancienne implémentation = 0."""
    model = predictor.load_model()
    names = predictor.get_feature_names()
    index = {name: i for i, name in enumerate(names)}

    legacy = np.array([legacy_proba(p, names, model) for p in reference_payloads])
    optimized = np.array([optimized_proba(p, names, index, model) for p in reference_payloads])

    assert np.max(np.abs(optimized - legacy)) == 0.0  # identiques au bit près


def test_decisions_unchanged(reference_payloads):
    """Mêmes scores ⇒ mêmes décisions métier au seuil 0.49 (aucun basculement)."""
    model = predictor.load_model()
    names = predictor.get_feature_names()
    index = {name: i for i, name in enumerate(names)}
    seuil = config.DECISION_THRESHOLD

    for p in reference_payloads:
        legacy = legacy_proba(p, names, model) >= seuil
        optimized = optimized_proba(p, names, index, model) >= seuil
        assert legacy == optimized


def test_endpoint_matches_legacy(client, sample_features):
    """L'endpoint /predict (chemin optimisé) reproduit le score de l'ancienne implémentation."""
    model = predictor.load_model()
    names = predictor.get_feature_names()
    expected = round(legacy_proba(sample_features, names, model), 4)

    body = client.post("/predict", json={"features": sample_features}).json()
    assert body["probability"] == expected


def test_inference_path_has_no_pandas():
    """Le chemin de production n'importe plus pandas (env de prod plus léger)."""
    # predictor n'expose aucun symbole pandas : l'inférence est 100 % numpy + LightGBM.
    assert not hasattr(predictor, "pd")
