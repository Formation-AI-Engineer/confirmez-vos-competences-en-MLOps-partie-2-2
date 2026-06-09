"""Configuration de l'API — chargée depuis les variables d'environnement / .env.

Les valeurs par défaut reprennent celles de ``.env.example`` (seuil métier 0.49,
chemins des artefacts du Projet 6). Les chemins relatifs sont résolus depuis la
racine du projet.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

# Secrets locaux d'abord (``.env.local``, gitignoré : URL Neon, etc.), puis ``.env``.
# ``load_dotenv`` n'écrase pas une variable déjà définie : ``.env.local`` est donc
# prioritaire sur ``.env``, et les vraies variables d'environnement (CI) priment sur tout.
load_dotenv(ROOT_DIR / ".env.local")
load_dotenv(ROOT_DIR / ".env")


def _resolve(value: str) -> Path:
    """Résout un chemin relatif depuis la racine du projet."""
    p = Path(value)
    return p if p.is_absolute() else ROOT_DIR / p


MODEL_PATH = _resolve(os.getenv("MODEL_PATH", "models/lgbm_final.joblib"))
MODEL_META_PATH = _resolve(os.getenv("MODEL_META_PATH", "models/model_meta.json"))
FEATURE_NAMES_PATH = _resolve(os.getenv("FEATURE_NAMES_PATH", "models/feature_names.json"))

# Seuil de décision métier : refus du crédit si proba de défaut >= seuil.
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.49"))

# --- Monitoring / stockage des données de production (étape 3) ---
# Base PostgreSQL des appels journalisés (inputs, sortie, latence, statut).
# - en local : conteneur Docker (cf. docker-compose.yml) ;
# - en production : Neon (Postgres serverless), URL fournie via le secret DATABASE_URL.
# Le secret n'est jamais committé ; seul ce défaut local l'est.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://scoring:scoring@localhost:5432/scoring")
# Active la journalisation des prédictions (désactivable en test / si non voulu).
MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "true").lower() in {"1", "true", "yes"}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
