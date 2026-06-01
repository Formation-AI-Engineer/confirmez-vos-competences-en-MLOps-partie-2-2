# Fiche de suivi — Projet 8

**Sujet** : Déployez et monitorez votre modèle de scoring (Home Credit Default Risk)
**Finalité** : Mise en production MLOps du modèle de scoring du Projet 6 — API + Docker + CI/CD + monitoring/drift
**Structure officielle** : 4 étapes du PDF OpenClassrooms + une étape 0 de préparation d'environnement.
**Prérequis** : modèle de scoring entraîné et versionné au **Projet 6** (« Initiez-vous au MLOps », partie 1/2).

## Documents du projet

| Document | Description |
|----------|-------------|
| [`contexte_general.md`](contexte_general.md) | Contexte métier, mission Chloé, modèle réutilisé, livrables |
| [`etape0_preparation_environnement.md`](etape0_preparation_environnement.md) | Environnement Python + récupération des artefacts du Projet 6 + structure projet |
| [`etape1_versioning_depot.md`](etape1_versioning_depot.md) | Dépôt Git, structure claire, commits explicites, push GitHub public |
| [`etape2_api_cicd.md`](etape2_api_cicd.md) | API (FastAPI/Gradio) + tests + Dockerfile + pipeline CI/CD GitHub Actions + déploiement |
| [`etape3_stockage_monitoring_drift.md`](etape3_stockage_monitoring_drift.md) | Stockage des données de prod + analyse data drift (Evidently) + dashboard de monitoring |
| [`etape4_optimisation_performances.md`](etape4_optimisation_performances.md) | Profiling, optimisation du temps d'inférence (ONNX…), re-déploiement via CI/CD |

## Livrables (énoncé)

| Livrable | Étape | Statut |
|----------|-------|--------|
| Environnement virtuel + dépendances (uv) | 0 | [x] |
| Artefacts du modèle Projet 6 récupérés dans `models/` (+ features, méta, réf. drift) | 0 | [x] |
| Structure de projet en place (`app/`, `tests/`, `monitoring/`, `notebooks/`…) | 0 | [x] |
| Dépôt Git public sur GitHub | 1 | [ ] |
| `.gitignore` (venv, données sensibles, gros fichiers) | 1 | [ ] |
| Historique de commits clair et explicite | 1 | [ ] |
| API fonctionnelle (FastAPI/Gradio) chargeant le modèle au démarrage | 2 | [ ] |
| Schéma de validation d'entrée + gestion d'erreurs + Swagger | 2 | [ ] |
| Tests unitaires (champs manquants, hors plage, mauvais types) | 2 | [ ] |
| Dockerfile + image qui build et tourne en local | 2 | [ ] |
| Pipeline CI/CD `.github/workflows/*.yml` (test → build → deploy) | 2 | [ ] |
| API déployée (HF Spaces / Cloud Run) + URL publique | 2 | [ ] |
| Logging structuré (JSON) des appels (inputs/outputs/latence) | 3 | [ ] |
| Solution de stockage des données de prod (décrite + screenshots) | 3 | [ ] |
| Script/notebook d'analyse de drift (Evidently) avec référence Projet 6 | 3 | [ ] |
| Dashboard / rapport de monitoring (distribution scores, latence…) | 3 | [ ] |
| Rapport de profiling + goulots d'étranglement identifiés | 4 | [ ] |
| Version optimisée du modèle (ex. ONNX) re-déployée via CI/CD | 4 | [ ] |
| Justification de la config finale + gain de temps d'inférence démontré | 4 | [ ] |
| Fiche d'auto-évaluation OpenClassrooms complétée | 4 | [ ] |
| `README.md` final (lancer l'API + interpréter le monitoring) | 4 | [ ] |

## Vision d'ensemble

```
Étape 0 ──► Étape 1 ──► Étape 2 ──────────► Étape 3 ──────────► Étape 4
   │           │            │                    │                   │
  env +      dépôt Git    API + tests +        stockage logs +    profiling +
 artefacts   GitHub       Docker + CI/CD +     analyse drift +    optimisation +
 Projet 6    public       déploiement          dashboard          re-déploiement
```

## Points de vigilance transverses (PDF + email Chloé)

- **Chargement du modèle une seule fois** au démarrage de l'API (perf, scalabilité, mémoire).
- **Validation d'entrée stricte** + gestion d'erreurs documentée (Swagger) — cas critiques testés.
- **Seuil métier** repris du Projet 6 (≈ 0.49, FN = 10 × FP), pas le 0.5 par défaut.
- **Sécurité** : secrets de déploiement via GitHub Secrets / plateforme ; jamais de données sensibles commitées.
- **Dépôt public** obligatoire.
- **Drift** : nécessite une référence (jeu de validation / `X_val_sample` du Projet 6).
- **PoC local accepté** : logs générés par l'API puis stockés/analysés localement si pas de cloud.
- **RGPD** sur les données de production loggées.

## Lien avec le Projet 6
Le modèle, son bundle de métadonnées, le jeu de validation et les modules `src/` (features, metrics) viennent
de `../Projet 6/`. L'étape 0 consiste à récupérer/figer ces artefacts dans `Projet 8/models/`.
Détail du modèle dans [`contexte_general.md`](contexte_general.md).
