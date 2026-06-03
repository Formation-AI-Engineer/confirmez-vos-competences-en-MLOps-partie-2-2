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

## Sommaire

- [Modèle servi](#modèle-servi)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Lancer l'API](#lancer-lapi)
- [Exemples d'appels](#exemples-dappels)
- [Tests](#tests)
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

Le modèle est chargé **une seule fois** au démarrage (singleton en mémoire). Documentation interactive (Swagger UI) : <http://localhost:8000/docs>

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

**Prédiction** (`POST /predict`) — un sous-ensemble des 804 features suffit, les manquantes sont traitées comme NaN (gérées nativement par LightGBM) :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 180000, "DAYS_BIRTH": -12000}}'
```

Réponse :

```json
{
  "probability": 0.3328,
  "decision": "accordé",
  "threshold": 0.49,
  "n_features_received": 3,
  "n_features_expected": 804
}
```

**Métadonnées du modèle** (`GET /model/info`)

```bash
curl http://localhost:8000/model/info
# {"model_type":"LGBMClassifier","n_features":804,"decision_threshold":0.49,...}
```

## Tests

**11 tests** couvrent le health check, une prédiction nominale, les cas critiques (champ manquant, mauvais type → `422`) et le chargement unique du modèle :

```bash
uv run pytest
```

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
app/         API FastAPI (config, schemas, predictor, main)
tests/       tests unitaires (pytest)
scripts/     génération/contrôle des artefacts dérivés du bundle
models/      artefacts du modèle (issus du Projet 6)
monitoring/  référence drift + dashboard (étape 3)
notebooks/   analyses (drift, profiling)
docs/        fiches d'étape et de suivi
```

> La section **Monitoring du data drift** sera ajoutée à l'étape 3.
```
