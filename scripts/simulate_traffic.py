"""Simulation de trafic de production contre l'API (étape 3).

En l'absence de trafic réel, ce script génère des données de production
réalistes : il rejoue des clients de l'échantillon de référence (Projet 6) et,
pour une fraction d'entre eux, **décale les features les plus déterminantes** afin
de provoquer un *data drift* visible. Chaque requête passe par ``POST /predict``,
donc l'API la **journalise** dans la base PostgreSQL de monitoring (tâche 3.1/3.2) —
ce sont ces logs qu'analysera Evidently (tâche suivante).

Prérequis : l'API doit tourner et écrire dans la base (``DATABASE_URL``) — en local,
le Postgres Docker (``docker compose up -d db``) ::

    uvicorn app.main:app --port 7860            # dans un terminal
    python scripts/simulate_traffic.py --n 2000 --drift-ratio 0.3

Le drift simule une population plus risquée : scores externes ↓, montants ↑,
âge et ancienneté ↓. Les features non décalées restent stables — Evidently doit
distinguer les unes des autres.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # rend le paquet ``app`` importable hors install

from app.monitoring import MONITORED_FEATURES  # noqa: E402  (après ajout de ROOT au path)

REFERENCE_PATH = ROOT / "monitoring" / "reference_sample.parquet"

# Features décalées pour simuler une dérive de population (plus risquée, plus
# endettée, plus jeune). Valeur = facteur multiplicatif appliqué à la référence.
DRIFT_SHIFTS = {
    "EXT_SOURCE_1": 0.6,  # scores externes ↓ (population plus risquée)
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.6,
    "AMT_CREDIT": 1.3,  # montants ↑ (crédits plus gros)
    "AMT_ANNUITY": 1.3,
    "AMT_GOODS_PRICE": 1.3,
    "DAYS_BIRTH": 0.7,  # plus proche de 0 = plus jeune
    "DAYS_EMPLOYED": 0.5,  # moins d'ancienneté
}


def build_payload(row: dict, drift: bool, features: list[str], rng: np.random.Generator) -> dict:
    """Construit le dict de features d'un client (réel ou perturbé).

    Les features de ``DRIFT_SHIFTS`` reçoivent un peu de bruit (spread réaliste)
    et, si ``drift``, le facteur de décalage. Les autres sont envoyées telles
    quelles, donc leur distribution reste stable.
    """
    feats: dict[str, float] = {}
    for key in features:
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        value = float(value)
        if key in DRIFT_SHIFTS:
            factor = DRIFT_SHIFTS[key] if drift else 1.0
            value *= factor * float(rng.normal(1.0, 0.05 if drift else 0.02))
            if key.startswith("EXT_SOURCE"):
                value = min(max(value, 0.0), 1.0)  # scores externes bornés à [0, 1]
        feats[key] = value
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:7860", help="URL de base de l'API")
    parser.add_argument("--n", type=int, default=2000, help="Nombre total de requêtes")
    parser.add_argument(
        "--drift-ratio", type=float, default=0.3, help="Fraction de requêtes perturbées (0–1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire (reproductibilité)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout HTTP (s)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # Préflight : l'API doit répondre, sinon rien ne sera journalisé.
    try:
        httpx.get(f"{base_url}/health", timeout=5).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(
            f"API injoignable sur {base_url} ({exc}).\n"
            "Démarre-la d'abord, p. ex. : uvicorn app.main:app --port 7860",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    ref = pd.read_parquet(REFERENCE_PATH)
    features = [f for f in MONITORED_FEATURES if f in ref.columns]

    n_drift = int(args.n * args.drift_ratio)
    drift_flags = rng.permutation([True] * n_drift + [False] * (args.n - n_drift))

    endpoint = f"{base_url}/predict"
    ok = errors = refused = 0
    latencies: list[float] = []

    print(f"Envoi de {args.n} requêtes vers {endpoint} ({n_drift} perturbées)…")
    with httpx.Client(timeout=args.timeout) as http:
        for i, drift in enumerate(drift_flags):
            row = ref.iloc[int(rng.integers(len(ref)))].to_dict()
            payload = {"features": build_payload(row, bool(drift), features, rng)}
            try:
                t0 = time.perf_counter()
                resp = http.post(endpoint, json=payload)
                latencies.append((time.perf_counter() - t0) * 1000)
                if resp.status_code == 200:
                    ok += 1
                    if resp.json()["decision"] == "refusé":
                        refused += 1
                else:
                    errors += 1
            except httpx.HTTPError as exc:
                errors += 1
                if errors <= 3:
                    print(f"  [warn] échec requête {i} : {exc}", file=sys.stderr)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{args.n} envoyées…")

    lat = np.array(latencies) if latencies else np.array([0.0])
    refus_rate = f"{refused / ok:.1%}" if ok else "n/a"
    print(
        f"\nTerminé.\n"
        f"  Requêtes OK        : {ok}/{args.n}\n"
        f"  Perturbées (drift) : {n_drift} ({args.drift_ratio:.0%})\n"
        f"  Erreurs            : {errors}\n"
        f"  Taux de refus      : {refus_rate}\n"
        f"  Latence client     : moy {lat.mean():.1f} ms | p95 {np.percentile(lat, 95):.1f} ms\n"
        "  → logs écrits dans PostgreSQL via DATABASE_URL (côté API)."
    )


if __name__ == "__main__":
    main()
