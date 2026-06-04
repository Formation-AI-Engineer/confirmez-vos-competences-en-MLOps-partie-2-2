# Étape 2 — Déployez le modèle via une API et automatisez avec CI/CD

## Objectif
Exposer le modèle de scoring via une **API** robuste, la **conteneuriser** avec Docker, et **automatiser**
tests + build + déploiement via un pipeline CI/CD (GitHub Actions).

## Résultats attendus (énoncé)
- Code source fonctionnel pour l'API.
- `Dockerfile` qui crée une image de l'API.
- Pipeline CI/CD fonctionnel et automatisé, visible sur la plateforme, qui déploie l'API.
- Tests automatisés intégrés au pipeline.

## Tâches

### 2.1 API de prédiction (FastAPI recommandé)
- [x] Endpoint racine / health check (`GET /` → redirige vers Swagger ; `GET /health`)
- [x] Endpoint `POST /predict` : reçoit les données d'un client, retourne **probabilité + décision** (seuil 0.49)
- [x] **Charger le modèle une seule fois au démarrage** (`lifespan` → `predictor.load_model`, singleton) — pas à chaque requête
- [x] Schéma d'entrée **Pydantic** (`PredictionInput` : dict de features, ≥ 1 item, valeurs numériques)
- [x] Gestion des features manquantes / alignement sur les **804 colonnes** (reindex → NaN gérés par LightGBM, clés inconnues ignorées)
- [x] **Gestion des erreurs** : 422 (validation Pydantic), 500 (erreur d'inférence) + Swagger auto. Endpoint bonus `GET /model/info`
- [x] (Optionnel) interface **Gradio** de démonstration montée dans FastAPI (`gr.mount_gradio_app` → `GET /demo`) : chargement de clients réels d'exemple, JSON éditable, proba + décision affichées (`app/demo.py`)

### 2.2 Tests unitaires automatisés
- [x] Test du health check (`/health` + redirection `/`)
- [x] Test d'une prédiction nominale (client 0 de la référence → proba 0.367 / accordé)
- [x] **Cas critiques** :
  - [x] champ obligatoire manquant (`features` absent + dict vide → 422)
  - [N/A] valeur hors plage : non applicable — les 804 features sont déjà encodées et certaines sont légitimement négatives (`DAYS_BIRTH`…). Remplacé par : clés inconnues ignorées sans erreur
  - [x] mauvais type (texte là où un nombre est attendu → 422)
- [x] Test que le modèle est chargé une seule fois (singleton, monkeypatch sur `joblib.load`)
- [x] Tests de l'**interface de démo** (`/demo` monté, prédiction depuis JSON, JSON invalide géré, exemples embarqués)
- [x] `pytest` vert en local (**15 tests passés**)

### 2.3 Conteneurisation Docker
- [x] `Dockerfile` (`python:3.10-slim` + `libgomp1` pour LightGBM, install deps API-only, copie code + modèle, `uvicorn` port 7860)
- [x] `.dockerignore` (déjà présent ; exclut venv, docs, notebooks, PDF, données…)
- [x] `docker build` + `docker run` testés en local (`/health`, `/model/info`, `/predict` OK sur le port 7860)
- [x] Empreinte maîtrisée : **image 739 Mo** (monitoring déplacé en extra ; +~100 Mo pour Gradio/démo), `HEALTHCHECK` sur `/health` → `healthy`

### 2.4 Pipeline CI/CD (GitHub Actions)
- [x] Workflow `.github/workflows/ci-cd.yml` (push `main`/`dev`/tags `v*`, PR vers `main`) ; install via `uv`
- [x] Job **lint** : `ruff check` + `ruff format --check` (config `[tool.ruff]`, line-length 100)
- [x] Job **test** : `uv sync --extra dev` + `pytest` (needs lint)
- [x] Étape **vérification des artefacts** : `python scripts/generate_model_artifacts.py --check` (échoue si `feature_names.json` / `model_meta.json` / `reference_sample.parquet` sont désynchronisés du bundle)
- [x] Job **build** : `docker build` de l'image si les tests passent (needs test)
- [x] Job **deploy** : force-push du dépôt vers le Space Hugging Face (needs build, `main`/tags uniquement)
- [x] **Secrets** via GitHub Secrets (`HF_TOKEN`, `HF_SPACE_ID`) — *à configurer à l'étape 2.5 pour activer le deploy*

### 2.5 Déploiement
- [x] Plateforme choisie : **Hugging Face Spaces** (type Docker)
- [x] Space créé : `lcamara/scoring-credit-pret-a-depenser` ; secret `HF_TOKEN` ajouté côté GitHub
- [x] En-tête YAML HF dans le `README.md` (`sdk: docker`, `app_port: 7860`) + `HF_SPACE_ID` dans le workflow
- [x] Déploiement automatisé depuis le pipeline (job `deploy` sur push `main`/tags)
- [x] **URL publique** de l'API testée (curl) + Swagger accessible : <https://lcamara-scoring-credit-pret-a-depenser.hf.space> — `/health` ✔, `/docs` (Swagger) ✔, `/model/info` ✔, `/predict` ✔
  - [ ] interface de démo `/demo` (2.1) **live** : codée + testée en local, à déployer (commit du code Gradio puis push `main`)

## Points de vigilance
- **Charger le modèle une seule fois** : sinon lenteurs / échec sous charge (rappel fort de l'énoncé).
- **Sécuriser** l'API et le pipeline : gestion des secrets, validation d'entrée stricte.
- Vérifier que l'**environnement de déploiement** a les ressources nécessaires (mémoire pour 804 features).
- **Séparer** build / test / deploy dans le pipeline ; commencer simple puis itérer.
- Tests **fiables** couvrant les cas critiques.

## Outils & ressources
- FastAPI / Gradio, Docker, Postman/curl, GitHub Actions, Pytest.
- Plateformes : Hugging Face Spaces, Cloud Run, Heroku.

## Statut : TERMINÉ (API en prod sur HF Spaces ; démo `/demo` codée — live au prochain push `main`)
