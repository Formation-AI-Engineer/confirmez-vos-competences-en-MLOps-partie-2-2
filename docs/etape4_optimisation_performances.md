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

### 4.1 Analyse des performances
- [ ] Exploiter les données de monitoring (latence, temps d'inférence, CPU/mémoire) de l'étape 3
- [ ] **Profiler** l'API et la prédiction (`cProfile`, timing par étape : preprocessing vs inférence vs sérialisation)
- [ ] Identifier les **goulots d'étranglement** (chargement, alignement des 804 features, prédiction…)

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
