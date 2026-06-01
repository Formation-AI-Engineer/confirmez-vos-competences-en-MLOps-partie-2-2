# Étape 0 — Préparez l'environnement de travail

## Objectif
Mettre en place un environnement Python isolé et reproductible pour le déploiement, et récupérer les artefacts
du modèle de scoring produits au **Projet 6** sur lesquels toute la mission s'appuie.

## Résultats attendus
- Environnement virtuel fonctionnel + dépendances de déploiement/monitoring.
- Artefacts du modèle Projet 6 présents dans `models/`.
- Structure de dossiers en place.

## Tâches

### 0.1 Initialisation de l'environnement
- [x] `uv init` / `pyproject.toml` ; Python 3.10–3.12 (résolu en **3.10.12**, cohérence Projet 6)
- [x] Dépendances API : `fastapi`, `uvicorn`, `pydantic`
- [x] Dépendances modèle : `lightgbm==4.6.0`, `scikit-learn==1.7.2`, `joblib==1.5.3`, `pandas`, `numpy` (versions **épinglées** sur le Projet 6)
- [x] Dépendances tests : `pytest`, `httpx` (extra `dev`)
- [x] Dépendances monitoring : `evidently`, `streamlit`, `plotly`, `matplotlib`
- [x] `uv sync --extra dev` exécuté avec succès

### 0.2 Récupération des artefacts du Projet 6
- [x] Copier `lgbm_final.joblib` et `lgbm_final_bundle.joblib` dans `Projet 8/models/`
- [x] Vérifier le chargement : `LGBMClassifier`, **804 features**, seuil `0.49`
- [x] Échantillon de validation figé : `monitoring/reference_sample.parquet` (100 × 804) — **référence drift**
- [x] `dataset_version` = `4247222185ae` ; métadonnées extraites dans `models/model_meta.json`
- [x] Liste des 804 features figée dans `models/feature_names.json`

### 0.3 Structure de dossiers
- [x] `app/` (+ `__init__.py`) — code source de l'API
- [x] `tests/` (+ `__init__.py`) — tests unitaires
- [x] `monitoring/` — scripts/notebook de drift + dashboard
- [x] `notebooks/` — analyses (drift, profiling)
- [x] `models/` — artefacts du modèle
- [x] `docs/` — fiches d'étape
- [x] `.github/workflows/` — pipeline CI/CD

### 0.4 Fichiers de config
- [x] `.gitignore` (venv, données, logs, secrets)
- [x] `.env.example` (chemins modèle, seuil, log path)
- [x] `.dockerignore`
- [x] `README.md` initial

### 0.5 Vérification de bout en bout
- [x] Modèle chargé hors API + prédiction sur l'échantillon de validation
- [x] Probabilité + décision (seuil 0.49) cohérentes : proba moy. 0.359, 24/100 refus, client 0 → proba 0.367 / accordé

## Points de vigilance
- Les artefacts du modèle peuvent être **volumineux** → décider du versioning (Git LFS, ou stockage externe + script de download). Voir étape 1.
- Garder la **même version de LightGBM** que le Projet 6 pour éviter les incompatibilités de désérialisation.
- Ne pas committer les données brutes Home Credit (gros CSV).

## Statut : TERMINÉ
