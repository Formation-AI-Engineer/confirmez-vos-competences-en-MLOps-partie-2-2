"""Smoke test de l'API déployée (étape 4.4) — re-test de l'URL publique.

Après un re-déploiement par la CI/CD, vérifie que le Space Hugging Face répond
et que la version **optimisée** (étape 4.2) ne régresse pas en production :

1. ``GET /health``      → l'API est vivante ;
2. ``GET /model/info``  → bon modèle (LGBMClassifier, 804 features, seuil 0,49) ;
3. ``POST /predict``    → le **client de référence** rend toujours **0.367 / accordé**
   (même score qu'en local et qu'avant l'optimisation — non-régression en ligne).

Stdlib uniquement pour le HTTP (``urllib``) → exécutable sans dépendance dev. ::

    python scripts/smoke_test.py
    python scripts/smoke_test.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "monitoring" / "reference_sample.parquet"
DEFAULT_URL = "https://lcamara-scoring-credit-pret-a-depenser.hf.space"

EXPECTED_PROBA = 0.367  # client 0 de la référence (cf. tests/test_predict.py)
EXPECTED_DECISION = "accordé"


def get_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> tuple[int, dict]:
    """GET (ou POST si ``payload``) → (status_http, corps JSON)."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


def reference_client() -> dict:
    """Client 0 de la référence (valeurs réelles, NaN exclus)."""
    row = pd.read_parquet(REFERENCE_PATH).iloc[0].to_dict()
    return {k: float(v) for k, v in row.items() if pd.notna(v)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_URL, help="URL de base de l'API")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"Cible : {base}\n")
    ok = True

    # 1. Santé
    status, body = get_json(f"{base}/health")
    alive = status == 200 and body.get("status") == "ok"
    print(f"[{'OK ' if alive else 'KO '}] GET  /health      → {status} {body}")
    ok &= alive

    # 2. Métadonnées du modèle
    status, info = get_json(f"{base}/model/info")
    good_model = (
        status == 200
        and info.get("model_type") == "LGBMClassifier"
        and info.get("n_features") == 804
        and info.get("decision_threshold") == 0.49
    )
    print(f"[{'OK ' if good_model else 'KO '}] GET  /model/info  → {status} {info}")
    ok &= good_model

    # 3. Prédiction du client de référence (non-régression en ligne) + latence
    t0 = time.perf_counter()
    status, pred = get_json(f"{base}/predict", payload={"features": reference_client()})
    latency_ms = (time.perf_counter() - t0) * 1000
    no_reg = (
        status == 200
        and pred.get("probability") == EXPECTED_PROBA
        and pred.get("decision") == EXPECTED_DECISION
    )
    print(
        f"[{'OK ' if no_reg else 'KO '}] POST /predict     → {status} "
        f"proba={pred.get('probability')} décision={pred.get('decision')} "
        f"({latency_ms:.0f} ms aller-retour)"
    )
    print(f"        attendu : proba={EXPECTED_PROBA} décision={EXPECTED_DECISION}")
    ok &= no_reg

    verdict = "✅ Déploiement confirmé, aucune régression." if ok else "❌ Échec du smoke test."
    print("\n" + verdict)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
