---
title: Scoring Crédit Prêt à Dépenser
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Déployez et monitorez votre modèle de scoring — Prêt à Dépenser (MLOps 2/2)

Mise en production du modèle de scoring crédit développé au Projet 6 (« Initiez-vous au MLOps ») :
**API** de prédiction, **conteneurisation Docker**, **pipeline CI/CD** et **monitoring du data drift**.

> Projet OpenClassrooms — *Confirmez vos compétences en MLOps (Partie 2/2)*.
> Suivi détaillé des tâches dans [`docs/fiche_taches_projet8.md`](docs/fiche_taches_projet8.md).

## Démo en ligne

- **Space Hugging Face** : <https://huggingface.co/spaces/lcamara/scoring-credit-pret-a-depenser>
- **Swagger UI** : <https://lcamara-scoring-credit-pret-a-depenser.hf.space/docs>
- **Interface de démo (Gradio)** : <https://lcamara-scoring-credit-pret-a-depenser.hf.space/demo>

## Sommaire

- [Modèle servi](#modèle-servi)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Lancer l'API](#lancer-lapi)
- [Exemples d'appels](#exemples-dappels)
- [Tests](#tests)
- [Monitoring du data drift](#monitoring-du-data-drift)
- [Déploiement](#déploiement)
- [Structure du projet](#structure-du-projet)

## Modèle servi

`LGBMClassifier` (LightGBM, **804 features**), seuil métier **0.49** (un faux négatif coûte ~10 × un faux positif). La classe positive correspond au **défaut de paiement** : une probabilité ≥ seuil entraîne un **refus** du crédit. Les artefacts (`models/`) proviennent du Projet 6 ; les fichiers dérivés sont régénérés via `scripts/generate_model_artifacts.py`.

## Prérequis

- Python **3.10–3.12** (résolu en 3.10.12, cohérence avec le Projet 6)
- [`uv`](https://docs.astral.sh/uv/) pour la gestion des dépendances (ou `pip`)
- Docker (optionnel, pour la conteneurisation)

## Installation

```bash
# Cloner le dépôt
git clone git@github.com:Formation-AI-Engineer/confirmez-vos-competences-en-MLOps-partie-2-2.git
cd confirmez-vos-competences-en-MLOps-partie-2-2

# API seule (dépendances principales)
uv sync

# + outils de dev (tests) et de monitoring (étape 3)
uv sync --extra dev --extra monitoring
```

> Alternative `pip` : `pip install -e ".[dev,monitoring]"`
>
> L'image Docker de l'API n'installe que les dépendances principales (sans le monitoring) pour rester légère.

## Configuration

Copier le template et adapter si besoin :

```bash
cp .env.example .env
```

| Variable | Description | Défaut |
|---|---|---|
| `MODEL_PATH` | Chemin du modèle sérialisé | `models/lgbm_final.joblib` |
| `MODEL_META_PATH` | Métadonnées du modèle | `models/model_meta.json` |
| `FEATURE_NAMES_PATH` | Liste ordonnée des 804 features | `models/feature_names.json` |
| `DECISION_THRESHOLD` | Seuil de refus (proba ≥ seuil → refusé) | `0.49` |
| `LOG_LEVEL` | Niveau de log | `INFO` |

`.env` est gitignoré ; seul `.env.example` est versionné comme template.

## Lancer l'API

### En local (développement)

```bash
uv run uvicorn app.main:app --reload
```

Le modèle est chargé **une seule fois** au démarrage (singleton en mémoire). Documentation interactive (Swagger UI) : <http://localhost:8000/docs>. Interface de démo Gradio : <http://localhost:8000/demo>.

### Avec Docker

```bash
docker build -t scoring-api .
docker run --rm -p 7860:7860 scoring-api
```

Swagger UI : <http://localhost:7860/docs> (port 7860, convention Hugging Face Spaces).

## Exemples d'appels

Exemples sur l'instance locale (port 8000 ; remplacer par 7860 sous Docker).

**Health check**

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

**Prédiction** (`POST /predict`) — un sous-ensemble des 804 features suffit, les manquantes sont traitées comme NaN (gérées nativement par LightGBM). Les features les plus déterminantes sont les scores externes `EXT_SOURCE_1/2/3` (0–1, ↑ = moins risqué) et `CREDIT_TERM` (= annuité / crédit, **recalculée automatiquement** si absente) :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": 0.5,
                    "AMT_CREDIT": 500000, "AMT_ANNUITY": 25000, "AMT_INCOME_TOTAL": 180000,
                    "DAYS_BIRTH": -12000, "DAYS_EMPLOYED": -2000}}'
```

Réponse (avec `EXT_SOURCE_* = 0.5`) :

```json
{
  "probability": 0.3288,
  "decision": "accordé",
  "threshold": 0.49,
  "n_features_received": 9,
  "n_features_expected": 804
}
```

> Abaisser les scores externes à `0.15` fait passer la probabilité à **0.78 → refusé** : la décision réagit bien aux features déterminantes.

**Métadonnées du modèle** (`GET /model/info`)

```bash
curl http://localhost:8000/model/info
# {"model_type":"LGBMClassifier","n_features":804,"decision_threshold":0.49,...}
```

## Tests

**23 tests** couvrent le health check, une prédiction nominale, les cas critiques (champ manquant, mauvais type → `422`), la feature dérivée `CREDIT_TERM`, le chargement unique du modèle, l'interface de démo (`/demo`) et la journalisation du monitoring (appel réussi/erreur loggé, monitoring désactivé → rien écrit) :

```bash
uv run pytest
```

## Monitoring du data drift

Chaque appel à `/predict` est journalisé, puis les données de production sont analysées pour détecter la **dérive** (data drift) et les **problèmes opérationnels** (taux d'erreur, latence). Étude complète et recommandations : [`docs/etape3_rapport_drift.md`](docs/etape3_rapport_drift.md).

Prérequis : `uv sync --extra monitoring`.

### 1. Journalisation des prédictions (stockage)

Chaque appel est écrit dans une base **SQLite** (`monitoring/production_logs.db`) : horodatage UTC, latence, statut HTTP, proba + décision, et un sous-ensemble figé des features (top-30 par importance). La journalisation est *best-effort* (n'échoue jamais une prédiction) et désactivable.

| Variable | Description | Défaut |
|---|---|---|
| `LOG_DB_PATH` | Base SQLite des appels journalisés | `monitoring/production_logs.db` |
| `MONITORING_ENABLED` | Active la journalisation | `true` |

RGPD : les features sont déjà encodées/anonymisées (feature engineering du Projet 6) — aucune donnée personnelle directe n'est stockée.

### 2. Générer du trafic de production (PoC)

```bash
uv run uvicorn app.main:app --port 7860                              # terminal 1
uv run python scripts/simulate_traffic.py --n 2000 --drift-ratio 0.3 # terminal 2
```

Rejoue des clients réels (référence) + une fraction **perturbée** (population plus risquée) contre l'API → remplit la base avec un drift contrôlé.

### 3. Analyser le drift (Evidently)

```bash
uv run python scripts/analyze_drift.py
```

Compare la prod à la **référence** (`monitoring/reference_sample.parquet`, re-scorée par le modèle) sur les features surveillées **et le score prédit**. Produit `monitoring/reports/drift_report.html` (+ `.json`) et un résumé console des colonnes dérivantes.

### 4. Dashboard de monitoring (Streamlit)

```bash
uv run streamlit run monitoring/dashboard.py     # → http://localhost:8501
```

KPI (volume, taux d'erreur, taux de refus, latence moy/p95), distribution des scores prédits, latence d'inférence, débit/erreurs dans le temps, et indicateurs de drift.

### Et en production ?

Le Space Hugging Face déployé **n'expose que l'API** (`/docs`, `/demo`) ; le dashboard Streamlit et le rapport Evidently sont des **outils locaux**. De plus, le système de fichiers d'un Space HF gratuit est **éphémère** (la base SQLite ne survit pas à un rebuild). L'approche retenue est donc un **PoC** : logs générés par l'API, récupérés puis analysés en local. Pour une visualisation en production, il faudrait un **stockage persistant** (HF persistent storage ou une base type PostgreSQL) et un **Space Streamlit dédié** lisant cette base.

## Déploiement

L'API est déployée automatiquement sur **Hugging Face Spaces** (type Docker) via GitHub Actions.

Pipeline : [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml)

```
lint (ruff) → test (pytest + --check artefacts) → build (docker) → deploy (HF Spaces)
```

Déclencheurs :
- `push` sur `dev` ou PR vers `main` → **lint + test + build** (feedback rapide, pas de deploy)
- `push` sur `main` ou tag `v*` → **lint + test + build + deploy** vers le Space HF

Le job `deploy` envoie les fichiers nécessaires (Dockerfile, `app/`, `models/`, README, `pyproject.toml`) vers le Space via l'API `huggingface_hub` (`scripts/deploy_hf.py`) — ce qui gère nativement le modèle binaire (`.joblib`) en LFS/Xet, là où un `git push` classique est refusé par HF. Le Space (mode **Docker**, port 7860 via l'en-tête YAML de ce README) reconstruit alors l'image à partir du `Dockerfile`. Secret requis côté GitHub : `HF_TOKEN` (scope *write*).

## Structure du projet

```
app/         API FastAPI (config, schemas, predictor, monitoring, main) + démo Gradio (demo)
tests/       tests unitaires (pytest, 23 tests)
scripts/     artefacts dérivés + simulation de trafic + analyse de drift
models/      artefacts du modèle (issus du Projet 6)
monitoring/  référence drift, dashboard Streamlit, base de logs + rapports (générés)
docs/        fiches d'étape, suivi, rapport de drift
```
```
