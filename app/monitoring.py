"""Stockage des données de production — journalisation des prédictions (PostgreSQL).

Tâches 3.1 + 3.2 : chaque appel à ``/predict`` est journalisé dans une base
**PostgreSQL** — horodatage, latence, statut HTTP, sortie (proba + décision) et un
sous-ensemble ciblé des features reçues. Ces données alimentent l'analyse de drift
et le dashboard de monitoring.

Choix d'implémentation :
- **PostgreSQL** (et non plus SQLite) : stockage **persistant et partagé** —
  en local via un conteneur Docker, en production via **Neon** (Postgres serverless).
  L'API déployée (Space HF) **écrit** dans Neon ; le dashboard Streamlit (Space dédié)
  **lit** la même base → vraie persistance, plus de système de fichiers éphémère.
- **Pool de connexions** (``psycopg_pool``) : les connexions sont réutilisées, ce qui
  évite de payer une poignée de main TLS vers Neon à chaque prédiction.
- **Sous-ensemble de features** (``MONITORED_FEATURES``, top-30 par importance du
  modèle) stocké en **JSONB** : suffisant pour un drift parlant tout en gardant ~0,5 Ko/ligne.
- **RGPD** : features déjà encodées/anonymisées (feature engineering du Projet 6),
  aucune donnée personnelle directe n'est stockée.

La journalisation est *best-effort* : une erreur d'écriture ne doit jamais faire
échouer une prédiction.
"""

from __future__ import annotations

from datetime import datetime, timezone

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

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
    id                  BIGSERIAL PRIMARY KEY,
    ts                  TIMESTAMPTZ      NOT NULL,  -- horodatage UTC
    latency_ms          DOUBLE PRECISION NOT NULL,  -- latence d'inférence (ms)
    http_status         INTEGER          NOT NULL,  -- 200 si OK, 500 si erreur d'inférence
    probability         DOUBLE PRECISION,           -- proba de défaut (NULL si erreur)
    decision            TEXT,                       -- accordé / refusé (NULL si erreur)
    n_features_received INTEGER,                    -- features reconnues parmi les 804
    error               TEXT,                       -- message d'erreur (NULL si OK)
    features            JSONB            NOT NULL   -- {feature: valeur} (sous-ensemble surveillé)
);
"""

# Pool de connexions partagé (créé à l'init, réutilisé à chaque écriture).
_pool: ConnectionPool | None = None
# Passe à False si la base est injoignable au démarrage : on coupe alors la
# journalisation pour la session, pour ne pas payer un timeout à chaque prédiction.
_healthy: bool = True


def _get_pool() -> ConnectionPool:
    """Pool de connexions PostgreSQL (créé paresseusement depuis ``DATABASE_URL``)."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            config.DATABASE_URL,
            min_size=1,
            max_size=4,
            # autocommit : un INSERT = une transaction validée d'office.
            # connect_timeout court : si la base est down, on échoue vite (best-effort).
            kwargs={"autocommit": True, "connect_timeout": 3},
            timeout=3,  # attente max pour obtenir une connexion du pool
            open=True,
        )
    return _pool


def reset_pool() -> None:
    """Ferme le pool courant (utilisé par les tests pour repartir propre)."""
    global _pool, _healthy
    if _pool is not None:
        _pool.close()
        _pool = None
    _healthy = True


def init_db() -> None:
    """Crée la table de logs si nécessaire (idempotent). No-op si monitoring désactivé.

    *Best-effort* : si la base est injoignable au démarrage (Neon down, URL absente),
    on n'empêche pas l'API de démarrer — la journalisation reprendra dès que possible.
    """
    global _healthy
    if not config.MONITORING_ENABLED:
        return
    try:
        with _get_pool().connection() as conn:
            conn.execute(_SCHEMA)
        _healthy = True
    except Exception:
        # Base injoignable : on n'empêche pas l'API de démarrer, et on coupe la
        # journalisation pour la session (évite un timeout à chaque prédiction).
        _healthy = False


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
    if not config.MONITORING_ENABLED or not _healthy:
        return
    try:
        subset = _monitored_subset(features)
        with _get_pool().connection() as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(ts, latency_ms, http_status, probability, decision, "
                "n_features_received, error, features) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    datetime.now(timezone.utc),
                    round(latency_ms, 3),
                    http_status,
                    result.get("probability") if result else None,
                    result.get("decision") if result else None,
                    result.get("n_features_received") if result else None,
                    error,
                    Jsonb(subset),
                ),
            )
    except Exception:
        # Le monitoring ne doit jamais casser l'API : on avale toute erreur d'écriture.
        pass
