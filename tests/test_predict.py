"""Tests de l'endpoint POST /predict — cas nominal et cas critiques."""


def test_predict_nominal(client, sample_features):
    """Prédiction nominale : entrée valide → score cohérent."""
    r = client.post("/predict", json={"features": sample_features})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in ("accordé", "refusé")
    assert body["threshold"] == 0.49
    assert body["n_features_expected"] == 804


def test_predict_known_client(client, sample_features):
    """Le client 0 de la référence doit donner proba 0.367 / accordé (cohérence étape 0)."""
    r = client.post("/predict", json={"features": sample_features})
    body = r.json()
    assert body["probability"] == 0.367
    assert body["decision"] == "accordé"


def test_predict_subset_of_features(client):
    """Un sous-ensemble de features est accepté (les manquantes deviennent NaN)."""
    r = client.post(
        "/predict", json={"features": {"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 180000}}
    )
    assert r.status_code == 200
    assert r.json()["n_features_received"] == 2


def test_predict_derives_credit_term(client):
    """CREDIT_TERM (feature n°1) est recalculée depuis annuité + crédit."""
    r = client.post("/predict", json={"features": {"AMT_CREDIT": 500000, "AMT_ANNUITY": 25000}})
    assert r.status_code == 200
    # AMT_CREDIT + AMT_ANNUITY + CREDIT_TERM dérivée = 3 features reconnues.
    assert r.json()["n_features_received"] == 3


def test_predict_unknown_keys_ignored(client):
    """Les clés inconnues sont ignorées, pas une erreur."""
    r = client.post(
        "/predict",
        json={"features": {"FEATURE_INEXISTANTE": 1.0, "AMT_CREDIT": 500000}},
    )
    assert r.status_code == 200
    assert r.json()["n_features_received"] == 1  # seule AMT_CREDIT est reconnue


# --- Cas critiques (validation Pydantic → 422) ---


def test_predict_empty_features(client):
    """Dictionnaire de features vide → 422 (au moins 1 feature requise)."""
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 422


def test_predict_missing_features_key(client):
    """Champ obligatoire `features` manquant → 422."""
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_wrong_type(client):
    """Valeur non numérique (texte là où un nombre est attendu) → 422."""
    r = client.post("/predict", json={"features": {"AMT_CREDIT": "beaucoup"}})
    assert r.status_code == 422
