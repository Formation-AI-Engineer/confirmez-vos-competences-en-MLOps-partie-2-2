"""Stockage des données de production — journalisation des prédictions (SQLite).

Tâches 3.1 + 3.2 : chaque appel à ``/predict`` est journalisé dans une base
SQLite (``monitoring/production_logs.db`` par défaut) — horodatage, latence,
statut HTTP, sortie (proba + décision) et un sous-ensemble ciblé des features
reçues. Ces données alimentent l'analyse de drift et le dashboard de monitoring.

Choix d'implémentation :
- **SQLite** : aucun serveur à gérer, requêtable en SQL (screenshots livrables),
  et largement dimensionné pour le volume d'un PoC (millions de lignes possibles,
  un seul writer). La base est ignorée par git et n'entre pas dans l'image Docker.
- **Sous-ensemble de features** (``MONITORED_FEATURES``, top-30 par importance du
  modèle) : suffisant pour un drift parlant tout en gardant ~0,5 Ko/ligne.
- **RGPD** : features déjà encodées/anonymisées (feature engineering du Projet 6),
  aucune donnée personnelle directe n'est stockée.

La journalisation est *best-effort* : une erreur d'écriture ne doit jamais faire
échouer une prédiction.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from app import config

# Top-30 des features par importance du modèle (toutes présentes dans la
# référence drift). Sous-ensemble figé journalisé à chaque appel — déterminant
# pour la décision, donc pertinent pour la détection de drift.
MONITORED_FEATURES = [
    "CREDIT_TERM",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "APPROVED_CNT_PAYMENT_MEAN",
    "DAYS_EMPLOYED",
    "INSTAL_DPD_MEAN",
    "INSTAL_AMT_PAYMENT_SUM",
    "OWN_CAR_AGE",
    "AMT_GOODS_PRICE",
    "CODE_GENDER",
    "AMT_CREDIT",
    "PREV_CNT_PAYMENT_MEAN",
    "ANNUITY_INCOME_PERCENT",
    "ACTIVE_DAYS_CREDIT_MAX",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
    "DAYS_EMPLOYED_PERCENT",
    "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN",
    "NAME_EDUCATION_TYPE",
    "PREV_APP_CREDIT_PERC_MIN",
    "DAYS_ID_PUBLISH",
    "ACTIVE_DAYS_CREDIT_ENDDATE_MAX",
    "INSTAL_PAYMENT_DIFF_MEAN",
    "POS_MONTHS_BALANCE_SIZE",
    "PREV_NAME_YIELD_GROUP_low_action_MEAN",
    "BURO_DAYS_CREDIT_MAX",
    "CREDIT_INCOME_PERCENT",
    "NAME_FAMILY_STATUS_Married",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL,   -- horodatage UTC ISO-8601
    latency_ms          REAL    NOT NULL,   -- latence d'inférence (ms)
    http_status         INTEGER NOT NULL,   -- 200 si OK, 500 si erreur d'inférence
    probability         REAL,               -- proba de défaut (NULL si erreur)
    decision            TEXT,               -- accordé / refusé (NULL si erreur)
    n_features_received INTEGER,            -- features reconnues parmi les 804
    error               TEXT,               -- message d'erreur (NULL si OK)
    features            TEXT    NOT NULL     -- JSON {feature: valeur} (sous-ensemble surveillé)
);
"""


def _connect() -> sqlite3.Connection:
    """Ouvre une connexion SQLite (crée le dossier parent si besoin)."""
    config.LOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(config.LOG_DB_PATH)


def init_db() -> None:
    """Crée la table de logs si nécessaire (idempotent). No-op si monitoring désactivé."""
    if not config.MONITORING_ENABLED:
        return
    with _connect() as conn:
        conn.execute(_SCHEMA)


def _monitored_subset(features: dict) -> dict:
    """Extrait le sous-ensemble surveillé des features fournies."""
    return {k: features[k] for k in MONITORED_FEATURES if k in features}


def log_prediction(
    features: dict,
    latency_ms: float,
    http_status: int,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    """Journalise un appel à ``/predict``. *Best-effort* : ne lève jamais.

    ``features`` doit être déjà enrichi (``CREDIT_TERM`` calculé) pour que le
    drift porte sur ce que le modèle a réellement vu.
    """
    if not config.MONITORING_ENABLED:
        return
    try:
        subset = _monitored_subset(features)
        with _connect() as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(ts, latency_ms, http_status, probability, decision, "
                "n_features_received, error, features) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    round(latency_ms, 3),
                    http_status,
                    result.get("probability") if result else None,
                    result.get("decision") if result else None,
                    result.get("n_features_received") if result else None,
                    error,
                    json.dumps(subset, ensure_ascii=False),
                ),
            )
    except Exception:
        # Le monitoring ne doit jamais casser l'API : on avale toute erreur d'écriture.
        pass
