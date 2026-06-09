"""Déploiement de l'API sur Hugging Face Spaces via l'API `huggingface_hub`.

Utilisé par le job `deploy` de la CI. Contourne le rejet des fichiers binaires
par le serveur git de HF (`pre-receive hook declined`) : `upload_folder` gère
nativement LFS/Xet pour le modèle (`.joblib`). Seuls les fichiers utiles à
l'image Docker du Space sont envoyés (le reste est exclu).
"""

import os

from huggingface_hub import HfApi

SPACE_ID = os.environ.get("HF_SPACE_ID", "lcamara/scoring-credit-pret-a-depenser")

# Ce qui n'est PAS nécessaire au build/run du Space (image Docker = Dockerfile +
# pyproject + README + app/ + models/).
IGNORE_PATTERNS = [
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


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path=".",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"Deploy from CI ({sha})",
    )
    print(f"Déployé sur https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
