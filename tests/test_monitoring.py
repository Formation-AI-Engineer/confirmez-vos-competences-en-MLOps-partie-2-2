"""Tests du stockage des données de production (journalisation PostgreSQL — tâches 3.1/3.2).

Nécessite un PostgreSQL accessible via ``DATABASE_URL`` (conteneur local
``docker compose up -d db`` ou service Postgres de la CI). Chaque test repart d'une
table vide (``TRUNCATE``) pour être isolé.
"""

import psycopg
import pytest

from app import config, monitoring, predictor


@pytest.fixture
def temp_db(monkeypatch):
    """Active le monitoring sur la base de test et la vide (isolée par test)."""
    monkeypatch.setattr(config, "MONITORING_ENABLED", True)
    monitoring.reset_pool()  # repart sur l'URL courante
    monitoring.init_db()
    with psycopg.connect(config.DATABASE_URL) as conn:
        conn.execute("TRUNCATE predictions RESTART IDENTITY")
        conn.commit()
    yield
    monitoring.reset_pool()


def _rows() -> list[dict]:
    with psycopg.connect(config.DATABASE_URL) as conn:
        cur = conn.execute("SELECT * FROM predictions ORDER BY id")
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def test_init_db_creates_empty_table(temp_db):
    """La table est créée et vide tant qu'aucun appel n'a eu lieu."""
    assert _rows() == []


def test_successful_prediction_is_logged(temp_db, client, sample_features):
    """Un /predict réussi écrit une ligne complète (proba, décision, latence, features)."""
    resp = client.post("/predict", json={"features": sample_features})
    assert resp.status_code == 200

    rows = _rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["http_status"] == 200
    assert row["error"] is None
    assert row["decision"] in {"accordé", "refusé"}
    assert 0.0 <= row["probability"] <= 1.0
    assert row["latency_ms"] >= 0
    assert row["n_features_received"] > 0

    # JSONB -> psycopg renvoie déjà un dict. Seules les features surveillées sont stockées,
    # et CREDIT_TERM (enrichi) en fait partie.
    feats = row["features"]
    assert set(feats).issubset(set(monitoring.MONITORED_FEATURES))
    assert "CREDIT_TERM" in feats


def test_failed_prediction_is_logged(temp_db, client, monkeypatch):
    """Une erreur d'inférence est journalisée avec http_status=500 et le message."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(predictor, "predict", _boom)
    resp = client.post("/predict", json={"features": {"EXT_SOURCE_2": 0.5}})
    assert resp.status_code == 500

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["http_status"] == 500
    assert "boom" in rows[0]["error"]
    assert rows[0]["probability"] is None


def test_disabled_monitoring_does_not_log(temp_db, client, sample_features, monkeypatch):
    """Quand le monitoring est désactivé, aucun appel n'est écrit."""
    monkeypatch.setattr(config, "MONITORING_ENABLED", False)

    client.post("/predict", json={"features": sample_features})
    assert _rows() == []
