# Étape 4 — Analysez et optimisez les performances du modèle

## Objectif
À partir des données de monitoring (étape 3), analyser les performances en production, identifier les goulots
d'étranglement, tester des stratégies d'**optimisation** du temps d'inférence, puis **re-déployer** la version
optimisée via le pipeline CI/CD.

## Résultats attendus (énoncé)
- Un rapport détaillant les tests d'optimisation, les résultats et les goulots d'étranglement identifiés.
- Une version optimisée du modèle déployée via le pipeline CI/CD.
- Une justification de la configuration finale (librairies, software, hardware).
- L'amélioration du temps d'inférence / de réponse est démontrée.

## Tâches

### 4.1 Analyse des performances — `scripts/profile_inference.py`
- [x] Données de monitoring (étape 3) : latence serveur moy **3,4 ms** / p95 **5,2 ms**, 0 erreur — cohérent avec le profiling ci-dessous
- [x] **Profiling** décomposé (`cProfile` + timing par étape) sur 2000 itérations :
  - `add_derived_features` : **0,001 ms** (négligeable)
  - préparation (DataFrame 1×804 + `reindex` + `astype`) : **1,1 ms**
  - inférence `predict_proba` : **4,4 ms** → `predict()` complet ≈ **5,1 ms**
- [x] **Goulot identifié = surcoût pandas, pas le calcul des arbres** :
  - dans LightGBM, `_data_from_pandas` reconvertit le DataFrame en numpy avec un **contrôle de dtype par colonne** : `_is_allowed_numpy_dtype` appelé **1 608 000×** (804 features × 2000) → ~2,8 ms/appel
  - + construction du DataFrame (`frame.__init__`) et `reindex` côté préparation
  - ⇒ ~75 % du temps est du **glue pandas↔numpy**, pas l'inférence elle-même
  - **Piste 4.2** : passer un **numpy array pré-aligné** (ordre des 804 features) au lieu d'un DataFrame, et/ou export **ONNX**

### 4.2 Stratégies d'optimisation
- [x] **Optimisation de code** : vecteur **numpy pré-aligné** (804 features, ordre du modèle) +
  `booster_.predict` → court-circuite la construction du DataFrame **et** le wrapper sklearn
  `predict_proba`. Appliqué dans `app/predictor.py`.
- [x] **Export ONNX + ONNX Runtime** testé (`scripts/export_onnx.py`, extra `onnx`) — mesuré, **non adopté** (cf. décision).
- [x] **Benchmark reproductible** `scripts/benchmark_optimization.py` (5000 itérations, 100 clients réels) :

  | stratégie | inférence (moy) | vs baseline | écart de score max |
  |---|---|---|---|
  | `baseline` (DataFrame + `predict_proba`) | 1,97 ms | — | 0 |
  | `numpy` (vecteur + `predict_proba`) | 1,19 ms | 1,7× | **0** (identique) |
  | **`booster` (vecteur + `booster_.predict`)** | **0,05 ms** | **~38×** | **0** (identique) |
  | `onnx` (ONNX Runtime, float32) | 0,02 ms | ~87× | 1,8e-07 |

- [x] **Décision : on retient `booster` (numpy + `booster_.predict`)**, pas ONNX :
  - scores **bit-identiques** à la baseline (max|Δ| = 0) → **zéro risque de régression** ; ONNX introduit un écart float32 (~1,8e-07).
  - **zéro dépendance runtime** ajoutée (LightGBM est déjà requis) → image Docker inchangée, déploiement simple ; ONNX imposerait `onnxruntime` en prod.
  - le gain *absolu* d'ONNX (0,02 vs 0,05 ms) est **négligeable** : le temps de réponse est dominé par FastAPI + réseau (p95 serveur **5,2 ms**, étape 3), pas par l'inférence.
  - ⇒ ONNX (`scripts/export_onnx.py`, extra `onnx`, artefact `models/*.onnx` gitignoré) reste **documenté et reproductible** pour justifier le choix, mais hors production.
- [x] Threads / quantification : non nécessaires — l'inférence n'est plus le goulot après ce gain de ~38×.

### 4.3 Non-régression — `tests/test_non_regression.py` (4 tests)
- [x] **Mêmes scores / même AUC** : `test_scores_bit_identical` reconstruit l'ancienne implémentation
  (DataFrame + `predict_proba`) et la compare au chemin optimisé (`booster_.predict`) sur **les 100 clients
  réels** de la référence → écart de score max **= 0** (identique au bit près).
- [x] **Précision / biais** : scores bit-identiques ⇒ **même courbe ROC, même AUC, même précision et même biais
  à n'importe quel seuil** (corollaire mathématique : aucun label requis). `test_decisions_unchanged` le confirme
  côté métier — **aucune décision ne bascule** au seuil 0,49.
- [x] **Cohérence de l'endpoint** : `test_endpoint_matches_legacy` vérifie que `/predict` (chemin optimisé)
  reproduit exactement le score arrondi de l'ancienne implémentation.
- [x] **Compatibilité environnement de prod** : `test_inference_path_has_no_pandas` garantit que l'inférence
  ne dépend plus de pandas (100 % numpy + LightGBM) → mêmes versions épinglées qu'au Projet 6, image plus légère.
- [x] Suite complète : **27 tests passent** (23 → 27), ruff propre.

### 4.4 Re-déploiement
- [x] **Version optimisée intégrée** au dépôt : l'optimisation vit dans `app/predictor.py` (le code de
  l'API lui-même) — aucun nouvel artefact ni dépendance runtime. ONNX écarté ⇒ image Docker **inchangée**
  (toujours `pip install .` sans extra). `.dockerignore` étendu : le `.onnx` de benchmark ne peut pas entrer dans l'image.
- [x] **Pipeline CI/CD** (`.github/workflows/ci-cd.yml`) : `lint (ruff)` → `test (pytest, dont les 4 tests de
  non-régression 4.3 + `--check` artefacts)` → `build (docker)` → `deploy (HF Spaces via scripts/deploy_hf.py)`.
  Le déploiement ne part que sur `main`/tags ⇒ le re-déploiement se déclenche au **merge `dev` → `main`**.
- [x] **Re-test de l'URL publique** : `scripts/smoke_test.py` interroge le Space déployé
  (`/health`, `/model/info`, `/predict`) et vérifie que le **client de référence rend toujours 0,367 / accordé**
  → preuve de **non-régression en ligne**. Validé en local (54 ms aller-retour) ; à relancer après le déploiement HF :
  `python scripts/smoke_test.py`.

### 4.5 Rapport & justification
- [ ] Rapport : tests effectués, goulots identifiés, résultats chiffrés
- [ ] Justification de la **configuration finale** (librairies, software, hardware)
- [ ] Démonstration de l'amélioration du temps d'inférence/réponse

## Clôture du projet
- [ ] **README final** : comment lancer l'API + comment interpréter le monitoring
- [ ] Fiche d'**auto-évaluation** OpenClassrooms complétée
- [ ] Préparation de la **soutenance** (livrables + choix techniques justifiés)
- [ ] Vérifier l'historique de commits (livrable « historique des versions »)

## Points de vigilance
- Baser les hypothèses d'optimisation sur les **données de monitoring réelles**.
- Documenter **rigoureusement** l'impact sur performance ET précision.
- Pas de **régression** introduite par l'optimisation.

## Outils & ressources
- `cProfile`, ONNX Runtime.
- Documentation cProfile, ONNX Runtime Home.

## Statut : À FAIRE
