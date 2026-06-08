# Rapport — Étude du data drift en production (Étape 3.5)

> Restitution de l'analyse de dérive des données de l'API de scoring crédit.
> Synthétise les artefacts produits aux tâches 3.1 → 3.4.

## 1. Contexte et objectif

L'API de scoring crédit (Projet 6, déployée au Projet 8) attribue à chaque demande une
**probabilité de défaut** et une **décision** (accordé / refusé, seuil métier **0,49**).
Une fois en production, la population de demandeurs peut **s'éloigner** de celle vue à
l'entraînement (*data drift*), ce qui dégrade silencieusement la qualité des décisions.

Objectif : **détecter cette dérive** à partir des données réellement traitées par l'API,
et en tirer des **points de vigilance** et des **recommandations** opérationnelles.

## 2. Données et méthode

| Élément | Choix | Justification |
|---|---|---|
| **Référence** | `monitoring/reference_sample.parquet` (X_val du Projet 6, 100 lignes), **re-scorée** par le modèle déployé | distribution « saine » des features **et** des scores attendus |
| **Production** | 2000 appels journalisés en base SQLite (`production_logs.db`), appels réussis | données réellement vues par le modèle (features post-enrichissement) |
| **Périmètre** | 30 features surveillées (`MONITORED_FEATURES`, top importance) + le **score prédit** | features déterminantes pour la décision = pertinentes pour le drift |
| **Outil** | **Evidently 0.7.21**, `DataDriftPreset` | tests automatiques par type : K-S (numérique), khi-deux / Z-test (catégoriel), seuil p < 0,05 |
| **Trafic** | `scripts/simulate_traffic.py` : 30 % d'appels **perturbés** (population plus risquée) | en l'absence de trafic réel, injecte un drift contrôlé pour valider la détection |

**Artefacts** : `scripts/analyze_drift.py` (analyse), `monitoring/reports/drift_report.html`
(rapport Evidently détaillé), `monitoring/reports/drift_report.json` (résultats bruts),
`monitoring/dashboard.py` (dashboard Streamlit).

## 3. Résultats

### 3.1 Santé opérationnelle (aucun problème détecté)
- **Volume** : 2000 appels, **0 erreur** d'inférence (taux d'erreur 0 %).
- **Latence** : moyenne **3,4 ms**, p95 **5,2 ms** — très en deçà de toute cible raisonnable.
- **Décisions** : 1278 accordés / 722 refusés → **taux de refus 36 %**, cohérent avec une
  population d'entrée plus risquée.

### 3.2 Data drift : 4 colonnes dérivantes sur 31

| Colonne | Test | p-value | Drift |
|---|---|---|---|
| `EXT_SOURCE_3` | K-S | 0,0048 | ⚠️ **OUI** |
| `EXT_SOURCE_2` | K-S | 0,0050 | ⚠️ **OUI** |
| `DAYS_BIRTH` | K-S | 0,0065 | ⚠️ **OUI** |
| **`prediction`** (score prédit) | K-S | 0,0164 | ⚠️ **OUI** |
| `EXT_SOURCE_1` | K-S | 0,163 | non |
| `DAYS_EMPLOYED`, `AMT_*`, … (27 autres) | K-S / khi-deux / Z | > 0,05 | non |

**Le signal le plus important est la dérive du score prédit** (p = 0,0164) : la distribution
des probabilités de défaut s'est décalée vers le **haut risque**. C'est le résultat
**actionnable** — même quand certaines features individuelles ne déclenchent pas, la **sortie
du modèle**, elle, change, et c'est elle qui pilote les décisions accordé/refusé.

### 3.3 Pourquoi seulement une partie du drift injecté ressort
La perturbation portait sur `EXT_SOURCE_1/2/3`, `AMT_CREDIT/ANNUITY/GOODS`, `DAYS_BIRTH`,
`DAYS_EMPLOYED`. Seules 3 features (+ le score) sont détectées. C'est **attendu et réaliste** :

1. **Dilution** : seuls **30 % des appels** sont perturbés → le signal est noyé pour les
   features à forte variance.
2. **Référence petite** (100 lignes) → faible puissance statistique.
3. **Nature des features** :
   - `EXT_SOURCE_2/3` bornées [0,1] → un décalage ×0,6 est proportionnellement **très visible**.
   - `EXT_SOURCE_1` est **très lacunaire** (beaucoup de valeurs manquantes) → signal dilué.
   - `AMT_*` très asymétriques à forte variance → un ×1,3 sur 30 % des lignes est **absorbé**.
   - `DAYS_EMPLOYED` : dispersion énorme (anomalie 365243) → ×0,5 noyé.

## 4. Points de vigilance

1. **`EXT_SOURCE_*` = à la fois top features du modèle ET dérivantes** → priorité de surveillance n°1.
2. **Dérive du score prédit** → indicateur de synthèse à surveiller en premier (résume l'impact aval).
3. **Drift des entrées ≠ baisse de performance** : sans **labels réels** (défauts constatés) en
   production, on surveille un **proxy** (distribution des entrées/sorties), pas la performance
   réelle. C'est suffisant pour **alerter**, pas pour **mesurer** la dégradation.
4. **Sensibilité de la détection** dépend de la taille et de la fraîcheur de la **référence**.

## 5. Recommandations

### Surveillance & alertes
- **Planifier** `scripts/analyze_drift.py` à cadence régulière (ex. quotidienne) sur une
  **fenêtre glissante** de production.
- **Seuils d'alerte** :
  - drift du **score prédit** (p < 0,05) → alerte prioritaire ;
  - drift d'une **feature top-importance** (`EXT_SOURCE_*`, `DAYS_BIRTH`) → alerte ;
  - **part de colonnes dérivantes** > seuil (ex. > 30 %) → alerte globale ;
  - garde-fous **opérationnels** : taux d'erreur > 1 %, latence p95 > cible.

### Ré-entraînement
- Déclencher une **revue de modèle** dès que le **score prédit dérive** **et/ou** que plusieurs
  features déterminantes dérivent simultanément.
- Conditionner le ré-entraînement à la **disponibilité de labels** (issues de crédit réelles)
  pour mesurer la vraie performance, pas seulement le drift d'entrée.

### Gouvernance des données
- Utiliser une **référence figée et plus large** (jeu de validation complet, pas 100 lignes) ;
  la **rafraîchir** après chaque ré-entraînement.
- **Collecter les issues réelles** (défaut / non-défaut) pour passer du *data drift* à la mesure
  de **performance drift** (AUC, métrique métier dans le temps).
- *(Amélioration)* ajouter `DataSummaryPreset` (valeurs manquantes, hors-bornes) pour couvrir la
  **qualité des données** en plus de la dérive.

## 6. Conclusion

La chaîne de monitoring est **fonctionnelle de bout en bout** : l'API journalise chaque appel,
la base de production est analysable, Evidently détecte la dérive et un dashboard la restitue.
Sur le trafic simulé, le système **capte correctement** une population plus risquée — via les
features externes et, surtout, via la **dérive du score prédit**. Côté opérationnel, **aucun
problème** (0 erreur, latence faible). Les leviers d'industrialisation sont l'**automatisation
des contrôles**, la **collecte de labels réels** et une **référence robuste**.
