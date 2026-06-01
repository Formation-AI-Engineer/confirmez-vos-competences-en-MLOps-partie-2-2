# Contexte général — Projet 8

## Sujet
**Déployez et monitorez votre modèle de scoring** — Confirmez vos compétences en MLOps (Partie 2/2), OpenClassrooms.

## Mission (résumé de l'énoncé)
Vous êtes Data Scientist chez **"Prêt à Dépenser"**. Le modèle de scoring crédit développé et versionné au
**Projet 6** (« Initiez-vous au MLOps », partie 1/2) doit maintenant passer **en production**.

Message Slack de **Chloé Dubois** (Lead Data Scientist) :
> « Il nous faut absolument une **API fonctionnelle et déployable (Docker Ready!)** d'ici la fin de la semaine
> prochaine. On a aussi besoin d'un **dashboard ou rapport de suivi** pour vérifier que tout se passe bien une
> fois en prod (distribution des scores, temps de réponse, ce genre de choses). »

Le département **« Crédit Express »** veut traiter les demandes **en quasi temps réel**.

## Le modèle réutilisé (issu du Projet 6)
- **Type** : `LGBMClassifier` (LightGBM).
- **Artefacts** (`Projet 6/models/`) :
  - `lgbm_final.joblib` — le modèle entraîné.
  - `lgbm_final_bundle.joblib` — métadonnées : `best_params`, `best_score_cv` (AUC CV ≈ 0.772),
    `best_threshold = 0.49`, `val_best_threshold = 0.54`, `dataset_version`, `train_size`,
    `proba_val`, `y_val_arr`, `X_val_sample` (100 lignes pour les tests / référence drift).
- **Features attendues** : **804 colonnes** (issues du feature engineering Home Credit Default Risk).
- **Seuil métier** : FN = 10 × FP → le seuil optimal (≈ 0.49) **n'est pas** 0.5.
- **Sortie API** : probabilité de défaut + décision (accordé / refusé) selon le seuil métier.

## Données de référence pour le drift
`X_val_sample` du bundle + le jeu de validation du Projet 6 servent de **référence** pour comparer la
distribution des features et des scores en production (data drift).

## Livrables attendus (énoncé)
1. **Historique des versions** — dépôt GitHub public, commits explicites.
2. **Scripts** :
   1. **API fonctionnelle** (FastAPI ou Gradio) : entrée = données client, sortie = score de prédiction.
   2. **Tests unitaires** automatisés (cas critiques : champs manquants, valeurs hors plage, mauvais types).
   3. **Dockerfile** pour la conteneurisation.
   4. **Analyse du Data Drift** : tableau de bord / rapport de monitoring (distribution des scores, latence,
      temps d'inférence…) + screenshots du stockage des données de production.
   5. **Pipeline CI/CD** : YAML (GitHub Actions) automatisant tests + build + déploiement sur push `main`.
   6. **README** : comment lancer l'API et interpréter le monitoring.

## Choix techniques pressentis (à confirmer / justifier en soutenance)
- **API** : FastAPI (perf, validation Pydantic, Swagger auto) — Gradio possible pour une démo interactive.
- **Conteneurisation** : Docker.
- **CI/CD** : GitHub Actions.
- **Déploiement** : Hugging Face Spaces (simple) ou Google Cloud Run.
- **Drift / monitoring** : Evidently AI + dashboard Streamlit (ou Dash).
- **Stockage logs prod** : logging structuré JSON → fichier / SQLite / PostgreSQL.
- **Optimisation** : cProfile + ONNX Runtime.

## Points de vigilance transverses
- **Charger le modèle une seule fois** au démarrage de l'API (pas à chaque requête).
- **Ne jamais committer** de données sensibles ; dépôt **public**.
- **Validation d'entrée** stricte + gestion des erreurs documentée (Swagger).
- **Gestion des secrets** (credentials de déploiement) via les secrets GitHub / plateforme.
- **RGPD** sur les données de production loggées.
- La détection de drift nécessite une **référence** (données d'entraînement / fenêtre stable du Projet 6).
- Un **PoC entièrement local** est acceptable si pas de cloud (logs générés par l'API puis analysés en local).
