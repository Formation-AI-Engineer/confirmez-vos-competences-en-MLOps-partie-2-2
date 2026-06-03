# Étape 1 — Mettez en place le contrôle de version et le dépôt

## Objectif
Initialiser un dépôt Git **public** structuré clairement, avec un historique de commits explicite, et le pousser
sur GitHub. C'est le socle qui rendra la CI/CD (étape 2) possible.

## Résultats attendus (énoncé)
- Un lien vers un dépôt Git **public** (ex : GitHub) contenant le code structuré.
- Un historique de commits **clair et pertinent**.

## Tâches

### 1.1 Initialisation du dépôt
- [x] `git init` à la racine de `Projet 8/`
- [x] Créer le dépôt distant sur GitHub (`Formation-AI-Engineer/confirmez-vos-competences-en-MLOps-partie-2-2`)
- [x] `git remote add origin …`

### 1.2 Fichier `.gitignore`
- [x] Ignorer : `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`
- [x] Ignorer les données sensibles / volumineuses : `data/`, `*.csv`, logs de prod
- [x] Ignorer les secrets : `.env` (committer un `.env.example`)
- [x] Sort des artefacts `models/*.joblib` : **versionnés directement** dans Git (~2 Mo au total, sous les limites GitHub) → pas de LFS

### 1.3 Structure claire du projet
- [x] Vérifier l'arborescence : `app/`, `tests/`, `monitoring/`, `notebooks/`, `models/`, `scripts/`, `docs/`, `.github/`
- [x] `README.md` initial (sera enrichi en étape 4)
- [x] Fichier de dépendances présent (`pyproject.toml` + `uv.lock`)

### 1.4 Premiers commits
- [x] Commit initial : structure + docs + dépendances
- [x] Commits explicites au fil de l'eau (un commit = une intention claire)
- [x] Stratégie de branche : `main` + branche de travail `dev` (les deux poussées sur `origin`)
- [x] `git push -u origin main`

### 1.5 Vérification
- [x] Le dépôt est bien **public** et accessible
- [x] Aucune donnée sensible / aucun secret n'est présent dans l'historique (scan `git log --all` OK ; seul `.env.example`)
- [x] Les commits racontent la construction du projet (= livrable « historique des versions »)

## Points de vigilance
- **Ne jamais committer** de données sensibles ni de secrets (vérifier l'historique, pas juste le dernier commit).
- **Dépôt public obligatoire** (exigence d'évaluation).
- Messages de commit **explicites** : ils constituent le premier livrable de la mission.

## Outils & ressources
- Git, GitHub. Git LFS (si artefacts versionnés). Documentation Git, Quickstart GitHub.

## Statut : TERMINÉ
