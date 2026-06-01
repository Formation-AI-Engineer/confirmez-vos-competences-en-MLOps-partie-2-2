# Étape 1 — Mettez en place le contrôle de version et le dépôt

## Objectif
Initialiser un dépôt Git **public** structuré clairement, avec un historique de commits explicite, et le pousser
sur GitHub. C'est le socle qui rendra la CI/CD (étape 2) possible.

## Résultats attendus (énoncé)
- Un lien vers un dépôt Git **public** (ex : GitHub) contenant le code structuré.
- Un historique de commits **clair et pertinent**.

## Tâches

### 1.1 Initialisation du dépôt
- [ ] `git init` à la racine de `Projet 8/`
- [ ] Créer le dépôt distant sur GitHub (**public**)
- [ ] `git remote add origin …`

### 1.2 Fichier `.gitignore`
- [ ] Ignorer : `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`
- [ ] Ignorer les données sensibles / volumineuses : `data/`, `*.csv`, logs de prod
- [ ] Ignorer les secrets : `.env` (committer un `.env.example`)
- [ ] Décider du sort des artefacts `models/*.joblib` (Git LFS, ou exclus + script de téléchargement)

### 1.3 Structure claire du projet
- [ ] Vérifier l'arborescence : `app/`, `tests/`, `monitoring/`, `notebooks/`, `models/`, `docs/`, `.github/`
- [ ] `README.md` initial (sera enrichi en étape 4)
- [ ] Fichier de dépendances présent (`pyproject.toml` / `requirements.txt`)

### 1.4 Premiers commits
- [ ] Commit initial : structure + docs + dépendances
- [ ] Commits explicites au fil de l'eau (un commit = une intention claire)
- [ ] Adopter une stratégie de branche si pertinent (ex. `main` + branches de feature)
- [ ] `git push -u origin main`

### 1.5 Vérification
- [ ] Le dépôt est bien **public** et accessible
- [ ] Aucune donnée sensible / aucun secret n'est présent dans l'historique
- [ ] Les commits racontent la construction du projet (= livrable « historique des versions »)

## Points de vigilance
- **Ne jamais committer** de données sensibles ni de secrets (vérifier l'historique, pas juste le dernier commit).
- **Dépôt public obligatoire** (exigence d'évaluation).
- Messages de commit **explicites** : ils constituent le premier livrable de la mission.

## Outils & ressources
- Git, GitHub. Git LFS (si artefacts versionnés). Documentation Git, Quickstart GitHub.

## Statut : À FAIRE
