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
- [x] Logger chaque appel : timestamp UTC, inputs (features surveillées), output (proba + décision), latence (`time.perf_counter`), statut HTTP (`app/monitoring.py` ↔ `/predict`)
- [x] Logger les erreurs d'inférence (`http_status=500` + message → taux d'erreur exploitable)
- [x] Inputs/outputs loggés exploitables pour le drift : sous-ensemble figé `MONITORED_FEATURES` (**top-30 features par importance**, présentes dans la référence) + features enrichies (`CREDIT_TERM`)
- [x] Logging **best-effort** (n'échoue jamais une prédiction) + désactivable (`MONITORING_ENABLED`)

### 3.2 Solution de stockage des données de prod
- [x] Support choisi : **SQLite** (`monitoring/production_logs.db`) — requêtable en SQL, sans serveur, hors git/image Docker
- [x] Écriture des logs depuis l'API (`init_db` au démarrage, `log_prediction` à chaque appel)
- [x] **Génération de trafic de production** : `scripts/simulate_traffic.py` rejoue des clients réels (référence) + une fraction **perturbée** (scores externes ↓, montants ↑, âge/ancienneté ↓) contre l'API → remplit la base. Validé : 100 appels, 0 erreur, drift visible sur `EXT_SOURCE_*`
- [x] **Screenshots** de la solution de stockage (exigence livrable) — base ouverte dans **DB Browser for SQLite** (`docs/images/etape3/`)
- [x] Schéma documenté (table `predictions` : `ts, latency_ms, http_status, probability, decision, n_features_received, error, features`) ; RGPD : features déjà encodées/anonymisées, aucune donnée personnelle directe
- [x] Tests : `tests/test_monitoring.py` (appel réussi loggé, erreur loggée, monitoring désactivé → rien écrit) — **23 tests verts**

### 3.3 Analyse du data drift — `scripts/analyze_drift.py` (Evidently 0.7.21)
- [x] Référence = `monitoring/reference_sample.parquet` (X_val du Projet 6, 100 lignes) **re-scoré** par le modèle déployé
- [x] Distribution des **features** prod (2000 logs) vs référence — `DataDriftPreset`, tests K-S / khi-deux / Z-test auto par type
- [x] Distribution des **scores prédits** : colonne `prediction` surveillée comme une feature (prod = proba loggée, réf = re-score)
- [x] Rapport **Evidently** : `monitoring/reports/drift_report.html` (+ `drift_report.json` exploitable par le dashboard 3.4)
- [x] **Points de vigilance** : 4 colonnes dérivantes / 31 → `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH` et le **score prédit** (p<0.05). Drift injecté partiellement capté (dilution : 30 % de trafic perturbé + référence de 100 lignes)
- [ ] *(option)* `DataSummaryPreset` pour la **qualité des données** (valeurs manquantes, hors-bornes) — non ajouté pour l'instant

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
