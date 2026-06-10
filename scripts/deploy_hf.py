"""Déploiement des Spaces Hugging Face via l'API `huggingface_hub`.

Utilisé par le job `deploy` de la CI. Deux cibles, sélectionnées en argument :
- ``api``       : l'API FastAPI (racine du dépôt, fichiers inutiles au Space exclus) ;
- ``dashboard`` : le dashboard Streamlit (dossier ``streamlit_space/``).

`upload_folder` contourne le rejet des binaires par le git de HF (gère LFS/Xet
pour le modèle `.joblib`) et n'envoie que les fichiers **modifiés** : si rien n'a
changé, aucun commit n'est créé → pas de rebuild du Space.

Usage : ``python scripts/deploy_hf.py api`` (ou ``dashboard``, ou les deux ;
sans argument, déploie toutes les cibles).
"""

import os
import sys

from huggingface_hub import HfApi

# Ce qui n'est PAS nécessaire au build/run du Space API (image Docker = Dockerfile
# + pyproject + README + app/ + models/).
_API_IGNORE = [
    "*.pdf",
    ".git*",
    ".venv/*",
    ".github/*",
    ".idea/*",
    ".vscode/*",
    "notebooks/*",
    "docs/*",
    "tests/*",
    "scripts/*",
    "monitoring/*",
    "streamlit_space/*",  # contenu du Space Streamlit dédié, pas de l'API
    "docker-compose.yml",
    "**/__pycache__/*",
    ".pytest_cache/*",
    ".ruff_cache/*",
    "*.log",
    ".env",
    "uv.lock",
]

# Cibles de déploiement : chaque Space a son repo HF et son contenu.
TARGETS = {
    "api": {
        "repo_id": os.environ.get("HF_SPACE_ID", "lcamara/scoring-credit-pret-a-depenser"),
        "folder_path": ".",
        "ignore_patterns": _API_IGNORE,
    },
    "dashboard": {
        "repo_id": os.environ.get("HF_DASHBOARD_SPACE_ID", "lcamara/scoring-credit-stream"),
        "folder_path": "streamlit_space",
        # Le Space autonome a besoin de tout son dossier (dont uv.lock pour le build).
        "ignore_patterns": ["**/__pycache__/*"],
    },
}


def deploy(target: str) -> None:
    """Pousse une cible vers son Space Hugging Face."""
    cfg = TARGETS[target]
    api = HfApi(token=os.environ["HF_TOKEN"])
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    api.upload_folder(
        repo_id=cfg["repo_id"],
        repo_type="space",
        folder_path=cfg["folder_path"],
        ignore_patterns=cfg["ignore_patterns"],
        commit_message=f"Deploy from CI ({sha})",
    )
    print(f"Déployé ({target}) sur https://huggingface.co/spaces/{cfg['repo_id']}")


def main() -> None:
    targets = sys.argv[1:] or list(TARGETS)
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise SystemExit(f"Cible(s) inconnue(s) : {unknown}. Choix possibles : {list(TARGETS)}")
    for target in targets:
        deploy(target)


if __name__ == "__main__":
    main()
