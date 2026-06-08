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
- [ ] Tester l'export du modèle en **ONNX** + **ONNX Runtime** pour l'inférence
- [ ] Optimisation de code (vectorisation, pré-allocation, éviter les conversions inutiles)
- [ ] (Optionnel) quantification / réglage du nombre de threads
- [ ] Mesurer le gain : temps d'inférence **avant / après** (benchmark reproductible)

### 4.3 Non-régression
- [ ] Vérifier que l'optimisation **n'altère pas** les prédictions (mêmes scores / même AUC)
- [ ] Vérifier l'absence de régression de **précision / biais**
- [ ] Valider la **compatibilité** avec l'environnement de production

### 4.4 Re-déploiement
- [ ] Intégrer la version optimisée au dépôt
- [ ] Laisser le **pipeline CI/CD** la tester, builder et déployer
- [ ] Confirmer le déploiement et re-tester l'URL publique

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
