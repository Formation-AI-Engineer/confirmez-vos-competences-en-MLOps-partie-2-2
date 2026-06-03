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
- [ ] (Optionnel) interface **Gradio/Streamlit** de démonstration par-dessus l'API

### 2.2 Tests unitaires automatisés
- [x] Test du health check (`/health` + redirection `/`)
- [x] Test d'une prédiction nominale (client 0 de la référence → proba 0.367 / accordé)
- [x] **Cas critiques** :
  - [x] champ obligatoire manquant (`features` absent + dict vide → 422)
  - [N/A] valeur hors plage : non applicable — les 804 features sont déjà encodées et certaines sont légitimement négatives (`DAYS_BIRTH`…). Remplacé par : clés inconnues ignorées sans erreur
  - [x] mauvais type (texte là où un nombre est attendu → 422)
- [x] Test que le modèle est chargé une seule fois (singleton, monkeypatch sur `joblib.load`)
- [x] `pytest` vert en local (**11 tests passés**)

### 2.3 Conteneurisation Docker
- [ ] `Dockerfile` (image légère type `python:3.11-slim`, install deps, copie code + modèle, `uvicorn`)
- [ ] `.dockerignore`
- [ ] `docker build` + `docker run` testés en local (API répond sur le port exposé)
- [ ] Vérifier l'empreinte mémoire / le temps de démarrage

### 2.4 Pipeline CI/CD (GitHub Actions)
- [ ] Workflow `.github/workflows/ci-cd.yml` déclenché sur push `main`
- [ ] Job **test** : install deps + `pytest`
- [ ] Étape **vérification des artefacts** : `python scripts/generate_model_artifacts.py --check` (échoue si `feature_names.json` / `model_meta.json` / `reference_sample.parquet` sont désynchronisés du bundle)
- [ ] Job **build** : build de l'image Docker si les tests passent
- [ ] Job **deploy** : push de l'image / déploiement sur l'environnement cible (simulé ou réel)
- [ ] **Secrets** gérés via GitHub Secrets (jamais en clair)

### 2.5 Déploiement
- [ ] Choisir la plateforme : **Hugging Face Spaces** (simple) / Google Cloud Run / Heroku
- [ ] Déploiement automatisé depuis le pipeline
- [ ] **URL publique** de l'API testée (curl/Postman) + Swagger accessible

## Points de vigilance
- **Charger le modèle une seule fois** : sinon lenteurs / échec sous charge (rappel fort de l'énoncé).
- **Sécuriser** l'API et le pipeline : gestion des secrets, validation d'entrée stricte.
- Vérifier que l'**environnement de déploiement** a les ressources nécessaires (mémoire pour 804 features).
- **Séparer** build / test / deploy dans le pipeline ; commencer simple puis itérer.
- Tests **fiables** couvrant les cas critiques.

## Outils & ressources
- FastAPI / Gradio, Docker, Postman/curl, GitHub Actions, Pytest.
- Plateformes : Hugging Face Spaces, Cloud Run, Heroku.

## Statut : À FAIRE
