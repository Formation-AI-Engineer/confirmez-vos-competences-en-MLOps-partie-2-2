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

## Démarrage rapide

> Le plus simple pour tout lancer en local **sans rien configurer** — il suffit de Docker (avec Docker Compose).

```bash
git clone git@github.com:Formation-AI-Engineer/confirmez-vos-competences-en-MLOps-partie-2-2.git
cd confirmez-vos-competences-en-MLOps-partie-2-2

# Stack complète : Postgres + API + dashboard, câblés ensemble
docker compose up --build
```

Puis, dans un autre terminal, **remplir la base** avec du trafic de production simulé (drift inclus) :

```bash
docker compose --profile traffic up traffic
```

Et c'est tout. Les services :

| Service | URL | Rôle |
|---|---|---|
| **API** (Swagger) | <http://localhost:7860/docs> | prédire (`POST /predict`) — chaque appel est journalisé |
| **API** (démo Gradio) | <http://localhost:7860/demo> | même prédiction, vue « métier » |
| **Dashboard** Streamlit | <http://localhost:8501> | KPI + data drift (lit la même base) |

Arrêt et nettoyage : `docker compose down` (ajouter `-v` pour effacer aussi les données Postgres).

> Sans Docker (dev Python) : `uv sync --extra dev --extra monitoring`, puis `docker compose up -d db` (Postgres seul) et `uv run uvicorn app.main:app --reload`. Détails dans les sections ci-dessous.

## Sommaire

- [Démarrage rapide](#démarrage-rapide)
- [Modèle servi](#modèle-servi)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Lancer l'API](#lancer-lapi)
- [Exemples d'appels](#exemples-dappels)
- [Tests](#tests)
- [Monitoring du data drift](#monitoring-du-data-drift)
- [Optimisation des performances](#optimisation-des-performances)
- [Déploiement](#déploiement)
- [Structure du projet](#structure-du-projet)

## Modèle servi

`LGBMClassifier` (LightGBM, **804 features**), seuil métier **0.49** (un faux négatif coûte ~10 × un faux positif). La classe positive correspond au **défaut de paiement** : une probabilité ≥ seuil entraîne un **refus** du crédit. Les artefacts (`models/`) proviennent du Projet 6 ; les fichiers dérivés sont régénérés via `scripts/generate_model_artifacts.py`.

## Prérequis

Selon la façon de lancer le projet :

- **Tout en Docker** (voie recommandée, cf. [Démarrage rapide](#démarrage-rapide)) : **Docker** + le plugin **Docker Compose** (`docker compose version`). Rien d'autre à installer.
- **Dev Python** (hors conteneur) : **Python 3.10–3.12** (résolu en 3.10.12, cohérence avec le Projet 6) et [`uv`](https://docs.astral.sh/uv/) (ou `pip`). Docker reste utile pour la base Postgres locale (`docker compose up -d db`).

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

**Optionnelle en local** : les valeurs par défaut suffisent (`docker compose up` injecte déjà la bonne `DATABASE_URL`, et le défaut pointe vers le Postgres local). On ne configure que pour **changer un défaut** ou **viser Neon en production**. Le cas échéant, copier le template :

```bash
cp .env.example .env.local     # secrets (DATABASE_URL Neon…) — gitignoré, NE PAS committer
```

| Variable | Description | Défaut |
|---|---|---|
| `MODEL_PATH` | Chemin du modèle sérialisé | `models/lgbm_final.joblib` |
| `MODEL_META_PATH` | Métadonnées du modèle | `models/model_meta.json` |
| `FEATURE_NAMES_PATH` | Liste ordonnée des 804 features | `models/feature_names.json` |
| `DECISION_THRESHOLD` | Seuil de refus (proba ≥ seuil → refusé) | `0.49` |
| `DATABASE_URL` | Base PostgreSQL des logs (Docker local / Neon en prod) | `postgresql://scoring:scoring@localhost:5432/scoring` |
| `MONITORING_ENABLED` | Active la journalisation des prédictions | `true` |
| `LOG_LEVEL` | Niveau de log | `INFO` |

Deux fichiers sont chargés (dans l'ordre, sans écraser une variable déjà définie) : **`.env.local` d'abord** (gitignoré — y mettre les secrets comme l'URL Neon), puis `.env`. Les vraies variables d'environnement (CI, conteneur) priment sur les deux. Seul `.env.example` est versionné comme template.

## Lancer l'API

### En local (développement)

```bash
uv run uvicorn app.main:app --reload
```

Le modèle est chargé **une seule fois** au démarrage (singleton en mémoire). Documentation interactive (Swagger UI) : <http://localhost:8000/docs>. Interface de démo Gradio : <http://localhost:8000/demo>.

### Avec Docker

**Stack complète en une commande** — reproduit l'architecture de prod (API qui écrit dans une base, dashboard qui la lit) en local :

```bash
docker compose up --build        # Postgres + API (:7860) + dashboard Streamlit (:8501)
```

L'API (`api`) est reliée au service `db` via `DATABASE_URL` (défini dans `docker-compose.yml`) : chaque prédiction est journalisée. Le `dashboard` (image exacte du Space HF) lit la même base. Swagger UI : <http://localhost:7860/docs> (port 7860, convention Hugging Face Spaces) ; dashboard : <http://localhost:8501>. Arrêt : `docker compose down` (ajouter `-v` pour effacer aussi les données Postgres).

**API seule** (sans la base) — l'image construite par le `Dockerfile`, telle que déployée sur HF :

```bash
docker build -t scoring-api .
docker run --rm -p 7860:7860 scoring-api
```

Sans `DATABASE_URL` joignable, la journalisation se désactive d'elle-même (*best-effort*) sans bloquer l'API.

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

**27 tests** couvrent le health check, une prédiction nominale, les cas critiques (champ manquant, mauvais type → `422`), la feature dérivée `CREDIT_TERM`, le chargement unique du modèle, l'interface de démo (`/demo`), la journalisation du monitoring (appel réussi/erreur loggé, monitoring désactivé → rien écrit) et la **non-régression** de l'optimisation d'inférence (scores identiques à l'implémentation initiale) :

```bash
uv run pytest
```

## Monitoring du data drift

Chaque appel à `/predict` est journalisé dans **PostgreSQL**, puis les données de production sont analysées pour détecter la **dérive** (data drift) et les **problèmes opérationnels** (taux d'erreur, latence). Étude complète et recommandations : [`docs/etape3_rapport_drift.md`](docs/etape3_rapport_drift.md).

Prérequis : `uv sync --extra monitoring` + une base PostgreSQL démarrée :

```bash
docker compose up -d db        # Postgres local (miroir de Neon en prod)
cp .env.example .env.local     # puis renseigner DATABASE_URL (NE PAS committer)
```

### 1. Journalisation des prédictions (stockage)

Chaque appel est écrit dans **PostgreSQL** (`DATABASE_URL`) : horodatage UTC, latence, statut HTTP, proba + décision, et un sous-ensemble figé des features (top-30 par importance, en **JSONB**). La journalisation est *best-effort* (n'échoue jamais une prédiction), réutilise un **pool de connexions** et est désactivable.

| Variable | Description | Défaut |
|---|---|---|
| `DATABASE_URL` | URL PostgreSQL (Docker local / Neon en prod) | `postgresql://scoring:scoring@localhost:5432/scoring` |
| `MONITORING_ENABLED` | Active la journalisation | `true` |

Stockage : **Docker local** en dev, **Neon** (Postgres serverless) en production — l'URL Neon est fournie via le secret `DATABASE_URL`, jamais committée. RGPD : les features sont déjà encodées/anonymisées (feature engineering du Projet 6) — aucune donnée personnelle directe n'est stockée.

### 2. Générer du trafic de production (PoC)

Le script est un **client HTTP** de l'API : il rejoue des clients réels (référence) + une fraction **perturbée** (population plus risquée) sur `POST /predict` → l'API remplit la base avec un drift contrôlé.

**Sur l'hôte** (l'API tourne via `uv` ou Docker, sur le port 7860) :

```bash
uv run uvicorn app.main:app --port 7860                              # terminal 1
uv run python scripts/simulate_traffic.py --n 2000 --drift-ratio 0.3 # terminal 2
```

**Tout en Docker** — si la stack tourne déjà (`docker compose up`), ajouter juste le générateur ; sinon `--profile traffic up` démarre la stack **et** le trafic :

```bash
docker compose --profile traffic up traffic        # 2000 appels, 30 % perturbés
N=5000 DRIFT=0.5 docker compose --profile traffic up traffic   # paramètres surchargés
```

Le service `traffic` (profil dédié, ne démarre pas par défaut) attend que l'API soit *healthy*, envoie le trafic une fois puis s'arrête.

> Le script peut aussi viser l'**API déployée** : `--url https://lcamara-scoring-credit-pret-a-depenser.hf.space` (remplit alors Neon). Pour du trafic ponctuel sans script, Swagger (`/docs`) et la démo Gradio (`/demo`) génèrent aussi des appels journalisés, une requête à la fois.

#### Réinitialiser la base

> **Vider les lignes, garder la table** (repartir d'un jeu propre, sans rien reconstruire) :
>
> ```bash
> docker compose exec db psql -U scoring -d scoring -c "TRUNCATE predictions RESTART IDENTITY;"
> ```
>
> **Tout détruire** (volume Postgres inclus — base recréée au prochain `docker compose up`) :
>
> ```bash
> docker compose down -v
> ```
>
> | Commande | Conteneurs | Données |
> |---|---|---|
> | `TRUNCATE predictions …` | inchangés | lignes effacées, **table conservée** |
> | `docker compose down` | supprimés | **conservées** (volume `pgdata`) |
> | `docker compose down -v` | supprimés | **effacées** (volume supprimé) |
>
> ⚠️ `-v` est **destructif et irréversible** (données locales perdues). Sans effet sur **Neon** (base distante séparée).

### 3. Analyser le drift (Evidently)

```bash
uv run python scripts/analyze_drift.py
```

Compare la prod à la **référence** (`monitoring/reference_sample.parquet`, re-scorée par le modèle) sur les features surveillées **et le score prédit**. Produit `monitoring/reports/drift_report.html` (+ `.json`) et un résumé console des colonnes dérivantes.

### 4. Dashboard de monitoring (Streamlit)

KPI (volume, taux d'erreur, taux de refus, latence moy/p95), distribution des scores prédits, latence d'inférence, débit/erreurs dans le temps, et indicateurs de drift. Deux variantes existent :

**Dashboard du dépôt** (`monitoring/dashboard.py`, dépend de `app.config`) :

```bash
uv run streamlit run monitoring/dashboard.py     # → http://localhost:8501
```

**Image du Space en local** (`streamlit_space/`, autonome — celle déployée sur HF) — incluse dans la stack `docker compose up` (service `dashboard`, branché sur le Postgres local au lieu de Neon) → tu la testes **à l'identique** avant de pousser sur HF :

```bash
docker compose up dashboard                      # → http://localhost:8501
```

> Variante hors Docker : `cd streamlit_space && uv sync && DATABASE_URL=… uv run streamlit run app.py`. L'app lit la base via `DATABASE_URL` (Postgres local ou URL Neon) ; sa référence de drift est pré-scorée (`reference_scored.parquet`), aucun modèle requis.

### En production

Le monitoring est **persistant** : l'API déployée (Space HF) écrit chaque prédiction dans **Neon** (PostgreSQL serverless), et un **Space Streamlit dédié** ([`scoring-credit-stream`](https://huggingface.co/spaces/lcamara/scoring-credit-stream)) lit cette même base pour afficher le dashboard en ligne. Le code du Space est dans [`streamlit_space/`](streamlit_space/). Plus de système de fichiers éphémère : les données survivent aux redéploiements.

Il reste deux secrets à définir côté Hugging Face : `DATABASE_URL` (URL Neon) dans **les deux Spaces** (l'API en écriture, le dashboard en lecture).

## Optimisation des performances

Le temps d'inférence a été profilé puis optimisé (étape 4). Le profiling a montré que le goulot
n'était **pas le calcul des arbres** mais le **glue pandas↔numpy** (~75 % du temps). Le chemin
d'inférence construit désormais un **vecteur numpy pré-aligné** et appelle directement
`booster_.predict`, court-circuitant la construction du DataFrame **et** le wrapper sklearn
`predict_proba` : **~38× plus rapide** (1,97 ms → 0,05 ms par appel), avec des scores **identiques
au bit près** (zéro régression, prouvée par `tests/test_non_regression.py`) et **aucune dépendance
ajoutée**. ONNX a été testé puis écarté (gain absolu négligeable face au coût d'une dépendance runtime).

Étude complète, benchmark chiffré et justification de la configuration finale :
[`docs/etape4_rapport_optimisation.md`](docs/etape4_rapport_optimisation.md).

```bash
uv run python scripts/profile_inference.py        # profiling décomposé (4.1)
uv run python scripts/benchmark_optimization.py   # benchmark des stratégies (4.2)
```

Après un (re-)déploiement, vérifier l'URL publique (non-régression en ligne) :

```bash
uv run python scripts/smoke_test.py               # /health, /model/info, /predict
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
app/                 API FastAPI (config, schemas, predictor, monitoring, main) + démo Gradio (demo)
tests/               tests unitaires (pytest, 27 tests dont la non-régression de l'optimisation)
scripts/             artefacts dérivés, trafic, drift, profiling/benchmark, export ONNX, smoke test, deploy
models/              artefacts du modèle (issus du Projet 6)
monitoring/          référence drift, dashboard Streamlit, rapports (générés)
streamlit_space/     dashboard de monitoring en ligne (Space HF dédié, lit Neon) — autonome
docs/                fiches d'étape, suivi, rapports de drift et d'optimisation
Dockerfile           image de l'API (déployée sur HF Spaces)
docker-compose.yml   stack locale API + Postgres (miroir de la prod HF + Neon)
```
