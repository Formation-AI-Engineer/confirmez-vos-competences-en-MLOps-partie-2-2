"""Tests du stockage des données de production (journalisation SQLite — tâches 3.1/3.2)."""

import json
import sqlite3

import pytest

from app import config, monitoring, predictor


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Active le monitoring sur une base SQLite temporaire (isolée par test)."""
    db = tmp_path / "logs.db"
    monkeypatch.setattr(config, "LOG_DB_PATH", db)
    monkeypatch.setattr(config, "MONITORING_ENABLED", True)
    monitoring.init_db()
    return db


def _rows(db) -> list[dict]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM predictions")]


def test_init_db_creates_empty_table(temp_db):
    """La table est créée et vide tant qu'aucun appel n'a eu lieu."""
    assert _rows(temp_db) == []


def test_successful_prediction_is_logged(temp_db, client, sample_features):
    """Un /predict réussi écrit une ligne complète (proba, décision, latence, features)."""
    resp = client.post("/predict", json={"features": sample_features})
    assert resp.status_code == 200

    rows = _rows(temp_db)
    assert len(rows) == 1
    row = rows[0]
    assert row["http_status"] == 200
    assert row["error"] is None
    assert row["decision"] in {"accordé", "refusé"}
    assert 0.0 <= row["probability"] <= 1.0
    assert row["latency_ms"] >= 0
    assert row["n_features_received"] > 0

    feats = json.loads(row["features"])
    # Seules les features surveillées sont stockées, et CREDIT_TERM (enrichi) en fait partie.
    assert set(feats).issubset(set(monitoring.MONITORED_FEATURES))
    assert "CREDIT_TERM" in feats


def test_failed_prediction_is_logged(temp_db, client, monkeypatch):
    """Une erreur d'inférence est journalisée avec http_status=500 et le message."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(predictor, "predict", _boom)
    resp = client.post("/predict", json={"features": {"EXT_SOURCE_2": 0.5}})
    assert resp.status_code == 500

    rows = _rows(temp_db)
    assert len(rows) == 1
    assert rows[0]["http_status"] == 500
    assert "boom" in rows[0]["error"]
    assert rows[0]["probability"] is None


def test_disabled_monitoring_does_not_log(tmp_path, monkeypatch, client, sample_features):
    """Quand le monitoring est désactivé, aucun appel n'est écrit."""
    db = tmp_path / "logs.db"
    monkeypatch.setattr(config, "LOG_DB_PATH", db)
    monkeypatch.setattr(config, "MONITORING_ENABLED", False)
    monitoring.init_db()  # no-op car désactivé

    client.post("/predict", json={"features": sample_features})
    assert not db.exists()
