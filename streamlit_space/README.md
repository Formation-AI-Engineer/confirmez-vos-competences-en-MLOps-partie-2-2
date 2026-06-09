---
title: Scoring Crédit — Monitoring
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Monitoring de production — API de scoring crédit

Dashboard Streamlit qui visualise, **en direct**, les prédictions journalisées par
l'API de scoring crédit (Projet 8) dans une base **PostgreSQL Neon**.

- **KPI** : volume, taux d'erreur, taux de refus, latence moyenne / p95.
- **Distribution des scores** prédits (avec le seuil métier 0,49).
- **Latence** et **débit** dans le temps.
- **Data drift** à la demande (Evidently) : production vs référence pré-scorée du Projet 6.

## Configuration

Définir le secret **`DATABASE_URL`** dans *Settings → Variables and secrets* du Space :
l'URL de connexion Neon (la même que celle utilisée en écriture par l'API). Optionnel :
`DECISION_THRESHOLD` (défaut `0.49`).

## Déploiement

Ces fichiers (`Dockerfile`, `pyproject.toml`, `app.py`, `reference_scored.parquet`,
`README.md`) constituent le contenu du Space **Docker** `lcamara/scoring-credit-stream`.
Les pousser vers le dépôt du Space ; HF construit l'image via le `Dockerfile` (uv +
`pyproject.toml`, **même outillage que l'API** — pas de `requirements.txt`).

> La référence `reference_scored.parquet` (100 clients du Projet 6, déjà scorés) est
> embarquée pour calculer le drift **sans** charger le modèle.
