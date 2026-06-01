# Déployez et monitorez votre modèle de scoring — Prêt à Dépenser (MLOps 2/2)

Mise en production du modèle de scoring crédit développé au Projet 6 (« Initiez-vous au MLOps ») :
**API** de prédiction, **conteneurisation Docker**, **pipeline CI/CD** et **monitoring du data drift**.

> Projet OpenClassrooms — *Confirmez vos compétences en MLOps (Partie 2/2)*.
> Suivi détaillé des tâches dans [`docs/fiche_taches_projet8.md`](docs/fiche_taches_projet8.md).

## Modèle servi
`LGBMClassifier` (LightGBM, 804 features), seuil métier **0.49** (FN = 10 × FP). Artefacts dans `models/`.

## Installation
```bash
uv sync --extra dev
```

## Structure
```
app/         code de l'API (FastAPI)
tests/       tests unitaires
monitoring/  analyse de drift + dashboard
notebooks/   analyses (drift, profiling)
models/      artefacts du modèle (issus du Projet 6)
docs/        fiches d'étape et de suivi
```

_La section « lancer l'API » et « interpréter le monitoring » sera complétée en fin de projet (étape 4)._
