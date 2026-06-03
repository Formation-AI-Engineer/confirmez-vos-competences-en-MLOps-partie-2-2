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
- [ ] Endpoint racine / health check (`GET /` ou `/health`)
- [ ] Endpoint `POST /predict` : reçoit les données d'un client, retourne **probabilité + décision** (seuil 0.49)
- [ ] **Charger le modèle une seule fois au démarrage** (événement `startup` / variable globale) — pas à chaque requête
- [ ] Schéma d'entrée **Pydantic** (validation des champs, types, plages)
- [ ] Gestion des features manquantes / alignement sur les **804 colonnes** attendues par le modèle
- [ ] **Gestion des erreurs** (HTTP 422/400/500) + documentation Swagger (auto avec FastAPI)
- [ ] (Optionnel) interface **Gradio/Streamlit** de démonstration par-dessus l'API

### 2.2 Tests unitaires automatisés
- [ ] Test du health check
- [ ] Test d'une prédiction nominale (valeurs valides → score cohérent)
- [ ] **Cas critiques** :
  - [ ] champ obligatoire manquant
  - [ ] valeur hors plage (ex. âge négatif, revenu = 0)
  - [ ] mauvais type (texte là où un nombre est attendu)
- [ ] Test que le modèle est chargé une seule fois (pas de rechargement par requête)
- [ ] `pytest` vert en local

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
