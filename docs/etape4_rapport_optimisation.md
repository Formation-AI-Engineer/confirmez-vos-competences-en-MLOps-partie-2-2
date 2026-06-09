# Rapport — Optimisation du temps d'inférence (Étape 4.5)

> Restitution de l'analyse et de l'optimisation des performances de l'API de scoring crédit.
> Synthétise les artefacts produits aux tâches 4.1 → 4.4.

## 1. Contexte et objectif

L'API de scoring crédit (Projet 6, déployée au Projet 8) attribue à chaque demande une
**probabilité de défaut** et une **décision** (accordé / refusé, seuil métier **0,49**), à partir
d'un `LGBMClassifier` de **804 features**. À partir des données de monitoring (étape 3), l'objectif
est d'**identifier le goulot d'étranglement** du temps d'inférence, de **tester des stratégies
d'optimisation**, de **prouver l'absence de régression**, puis de **re-déployer** via la CI/CD.

Point de départ (monitoring, étape 3) : latence serveur moyenne **3,4 ms**, p95 **5,2 ms**, **0 erreur**
sur 2000 appels. L'API est déjà rapide ; l'enjeu est de réduire la part **purement applicative** de
l'inférence, mesurable et reproductible hors réseau.

## 2. Méthode

| Élément | Choix | Justification |
|---|---|---|
| **Profiling** | `cProfile` + timing par étape (`scripts/profile_inference.py`) | localise le coût fonction par fonction, pas seulement le total |
| **Benchmark** | `scripts/benchmark_optimization.py`, 100 clients réels, 5000 itérations | mesure reproductible moy / médiane / p95 + écart de score vs baseline |
| **Données** | clients réels de `monitoring/reference_sample.parquet` | payloads représentatifs d'appels API réalistes |
| **Critère** | gain de latence **ET** écart de score max vs implémentation initiale | une optimisation n'est retenue que **sans régression** |

## 3. Goulot d'étranglement identifié (tâche 4.1)

Décomposition d'un appel `predict()` (2000 itérations) :

| Étape | Temps | Part |
|---|---|---|
| `add_derived_features` (dict + `CREDIT_TERM`) | 0,001 ms | négligeable |
| **Préparation** (DataFrame 1×804 + `reindex` + `astype`) | ~1,1 ms | glue pandas |
| **Inférence** `predict_proba` | ~4,4 ms | dont reconversion pandas→numpy |

**Constat** : le goulot **n'est pas le calcul des arbres**, mais le **glue pandas↔numpy**. `cProfile`
révèle que LightGBM, recevant un DataFrame, appelle `_is_allowed_numpy_dtype` **1 608 000×**
(804 features × 2000 appels) pour valider le dtype **colonne par colonne** via `_data_from_pandas`.
S'ajoute la construction du DataFrame (`frame.__init__`) et le `reindex` côté préparation.
⇒ **~75 % du temps est de la plomberie pandas**, pas de l'inférence.

## 4. Stratégies testées et résultats (tâches 4.2 / 4.3)

Trois alternatives produisant **la même** probabilité, mesurées sur 5000 itérations :

| Stratégie | Inférence (moy) | vs baseline | Écart de score max | Dépendance ajoutée |
|---|---|---|---|---|
| `baseline` — DataFrame + `predict_proba` | 1,97 ms | — | — | — |
| `numpy` — vecteur pré-aligné + `predict_proba` | 1,19 ms | 1,7× | **0** | aucune |
| **`booster` — vecteur + `booster_.predict`** ✅ | **0,05 ms** | **~38×** | **0** | aucune |
| `onnx` — ONNX Runtime (float32) | 0,02 ms | ~87× | 1,8 × 10⁻⁷ | `onnxruntime` |

**Lecture** :
- Passer un **vecteur numpy pré-aligné** (au lieu d'un DataFrame) supprime déjà le coût de préparation
  et la validation par colonne (`numpy` : 1,7×).
- Appeler directement `booster_.predict` **court-circuite en plus** le wrapper sklearn `predict_proba`
  (contrôles + assemblage des deux classes) : **~38×**, scores **identiques au bit près**.
- **ONNX** est le plus rapide (~87×) mais introduit un écart float32 (1,8 × 10⁻⁷) et une **dépendance
  runtime** supplémentaire.

**Non-régression (tâche 4.3)** — `tests/test_non_regression.py` (4 tests, dans la CI) :
la stratégie retenue rend des scores **bit-identiques** à l'implémentation initiale sur **les 100 clients**
(écart max = 0). Corollaire mathématique : **même AUC, même précision, même biais, mêmes décisions** à
tout seuil — sans avoir besoin de labels. Aucune décision ne bascule au seuil 0,49.

## 5. Configuration finale retenue et justification

**On retient la stratégie `booster` (numpy pré-aligné + `booster_.predict`)**, implémentée dans
`app/predictor.py`. ONNX est **écarté**, pour des raisons d'ingénierie :

| Critère | `booster` (retenu) | `onnx` (écarté) |
|---|---|---|
| Exactitude | scores **identiques** (max\|Δ\| = 0) | écart float32 (~1e-7) |
| Dépendance runtime | **aucune** (LightGBM déjà requis) | `onnxruntime` en prod |
| Image Docker | **inchangée** | + runtime + artefact |
| Gain *absolu* | 0,05 ms | 0,02 ms (Δ ≈ 0,03 ms) |

Le surcroît de vitesse d'ONNX (0,03 ms) est **négligeable** face au temps de réponse réel, dominé par
FastAPI + réseau (**p95 serveur 5,2 ms**). Il ne justifie pas une dépendance supplémentaire ni un risque
de régression numérique. ONNX reste **documenté et reproductible** (`scripts/export_onnx.py`, extra `onnx`)
pour justifier ce choix par des chiffres, mais hors production.

**Configuration finale (logiciel / matériel)** :

| Couche | Choix | Justification |
|---|---|---|
| **Librairies** | LightGBM 4.6.0, NumPy 2.2.6, scikit-learn 1.7.2 (versions épinglées sur le Projet 6) | désérialisation fiable du modèle + zéro nouvelle dépendance |
| **Inférence** | `Booster.predict` sur `np.ndarray` float64 1×804 | court-circuite pandas + wrapper sklearn |
| **Software** | Python 3.10, FastAPI/Uvicorn, image `python:3.10-slim` + `libgomp1` | runtime OpenMP de LightGBM, image légère |
| **Hardware** | CPU (HF Spaces, gratuit) — mesures sur Intel i7-1165G7 | modèle d'arbres : **pas de gain GPU**, CPU suffisant et économique |

Threads / quantification non nécessaires : après le gain de ~38×, l'inférence (0,05 ms) n'est plus le
goulot — l'effort serait sans effet mesurable sur le temps de réponse.

## 6. Amélioration démontrée et re-déploiement (tâche 4.4)

- **Avant** : 1,97 ms / appel (préparation DataFrame + `predict_proba`).
- **Après** : 0,05 ms / appel → **~38× plus rapide**, sans aucune régression de score.
- **Intégration** : l'optimisation est dans le code de l'API (`app/predictor.py`) — **aucun nouvel
  artefact**, image Docker identique.
- **CI/CD** (`.github/workflows/ci-cd.yml`) : `lint → test (dont non-régression) → build → deploy (HF Spaces)`.
  Le re-déploiement se déclenche au merge `dev → main`.
- **Re-test de l'URL publique** : `scripts/smoke_test.py` confirme en ligne que le client de référence
  rend toujours **0,367 / accordé** (non-régression en production).

## 7. Conclusion

Le profiling a montré que le coût d'inférence venait du **glue pandas**, pas du modèle. Une réécriture
**sans dépendance** du chemin d'inférence (vecteur numpy + `booster_.predict`) apporte un gain de **~38×**
avec des scores **strictement identiques**, validé par 4 tests de non-régression dans la CI. ONNX,
testé et chiffré, est écarté car son avantage est négligeable au regard du coût (dépendance + écart
numérique). La version optimisée est re-déployable telle quelle via le pipeline existant, et vérifiable
en ligne par un smoke test.
