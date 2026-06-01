# Étape 3 — Implémentez le stockage et l'analyse des données de production

## Objectif
Stocker les données générées par l'API en production (logs, inputs, outputs, temps d'exécution), puis analyser
automatiquement ces données pour détecter la **dérive (data drift)** et les **problèmes opérationnels**
(taux d'erreur, latence). Restituer le tout dans un **dashboard / rapport de monitoring**.

## Résultats attendus (énoncé)
- Une solution de stockage des données de production décrite et/ou implémentée.
- Un script ou notebook réalisant l'analyse automatique (détection de drift, anomalies).
- Une présentation de l'étude sur la dérive des données et les points de vigilance qui en résultent.

## Tâches

### 3.1 Logging structuré dans l'API
- [ ] Logger chaque appel en **JSON** : timestamp, inputs, output (proba + décision), latence, statut
- [ ] Logger les erreurs (taux d'erreur exploitable ensuite)
- [ ] S'assurer que les inputs/outputs loggés permettent une **analyse de drift ultérieure**

### 3.2 Solution de stockage des données de prod
- [ ] Choisir le support : fichier JSON/CSV, **SQLite**, ou **PostgreSQL** (PoC local accepté)
- [ ] Écrire les logs de l'API vers ce stockage
- [ ] **Screenshots** de la solution de stockage (exigence livrable)
- [ ] Documenter le schéma des données stockées (RGPD : pas de données sensibles inutiles)

### 3.3 Analyse du data drift
- [ ] Définir la **référence** : jeu de validation / `X_val_sample` du Projet 6
- [ ] Comparer la distribution des **features** de prod vs référence (**Evidently AI** ou NannyML)
- [ ] Comparer la distribution des **scores prédits** (référence vs prod)
- [ ] Générer un **rapport Evidently** (data drift + qualité des données)
- [ ] Identifier les **points de vigilance** (features dérivantes, seuils d'alerte)

### 3.4 Dashboard / rapport de monitoring
- [ ] Tableau de bord (**Streamlit** / Dash, ou notebook) avec métriques clés :
  - [ ] **distribution des scores** prédits
  - [ ] **latence** de l'API / **temps d'inférence**
  - [ ] **taux d'erreur** / volume d'appels
  - [ ] indicateurs de **drift**
- [ ] Visualisations claires + interprétation

### 3.5 Restitution
- [ ] Notebook ou rapport présentant l'étude de drift et les conclusions
- [ ] Recommandations (ré-entraînement, alertes) en cas de drift avéré

## Points de vigilance
- **Référence indispensable** pour le drift (sans elle, pas de comparaison).
- Contraintes de **stockage / coût** ; conformité **RGPD**.
- Logging **structuré** (JSON) dès l'API pour rendre l'analyse possible.
- PoC **local** suffisant : logs générés par l'API (cloud) puis téléchargés et analysés localement.

## Outils & ressources
- Logging Python, SQLite/PostgreSQL/Elasticsearch.
- **Evidently AI** / NannyML (drift).
- Streamlit/Dash, Grafana/Kibana (visualisation).
- Article « monitoring ML en python », documentation Evidently.

## Statut : À FAIRE
