"""Profiling du temps d'inférence de l'API (étape 4.1).

Décompose le coût d'un appel ``predictor.predict`` pour localiser le **goulot
d'étranglement** :
1. ``add_derived_features`` — enrichissement (dict + ``CREDIT_TERM``) ;
2. **préparation** — construction du DataFrame 1×804, ``reindex`` sur l'ordre du
   modèle, ``astype(float64)`` ;
3. **inférence** — ``model.predict_proba``.

Sorties : un tableau de timings (moyenne / médiane / p95) par étape sur N
itérations, puis un ``cProfile`` cumulé des fonctions les plus coûteuses.

Le payload de test est une vraie ligne de la référence (sous-ensemble des
features surveillées), représentative d'un appel API réaliste. ::

    python scripts/profile_inference.py --iters 2000
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # rend le paquet ``app`` importable hors install

from app import predictor  # noqa: E402
from app.monitoring import MONITORED_FEATURES  # noqa: E402

REFERENCE_PATH = ROOT / "monitoring" / "reference_sample.parquet"


def sample_payload() -> dict:
    """Une ligne réelle de la référence, limitée aux features surveillées."""
    ref = pd.read_parquet(REFERENCE_PATH)
    row = ref.iloc[0].to_dict()
    return {k: float(row[k]) for k in MONITORED_FEATURES if k in row and pd.notna(row[k])}


def timed(label: str, fn, iters: int) -> tuple[str, np.ndarray]:
    """Chronomètre ``fn`` sur ``iters`` itérations -> (label, latences en ms)."""
    lat = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        lat[i] = (time.perf_counter() - t0) * 1000
    return label, lat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=2000, help="Itérations par étape")
    parser.add_argument("--warmup", type=int, default=50, help="Itérations de chauffe")
    args = parser.parse_args()

    payload = sample_payload()
    model = predictor.load_model()  # singleton chargé hors mesure
    names = predictor.get_feature_names()

    # Étapes isolées (reproduisent predictor.predict, voir app/predictor.py)
    def step_derive():
        predictor.add_derived_features(payload)

    def step_prepare():
        feats = predictor.add_derived_features(payload)
        pd.DataFrame([feats]).reindex(columns=names).astype("float64")

    def step_infer():
        row = pd.DataFrame([predictor.add_derived_features(payload)])
        row = row.reindex(columns=names).astype("float64")
        model.predict_proba(row)

    def step_full():
        predictor.predict(payload)

    for _ in range(args.warmup):
        step_full()

    print(f"Payload : {len(payload)} features | itérations : {args.iters}\n")
    steps = [
        ("1. add_derived_features", step_derive),
        ("2. préparation (DataFrame+reindex+astype)", step_prepare),
        ("3. inférence (predict_proba)", step_infer),
        ("predict() complet", step_full),
    ]
    print(f"{'étape':<44}{'moy (ms)':>10}{'méd (ms)':>10}{'p95 (ms)':>10}")
    for label, fn in steps:
        _, lat = timed(label, fn, args.iters)
        p95 = np.percentile(lat, 95)
        print(f"{label:<44}{lat.mean():>10.3f}{np.median(lat):>10.3f}{p95:>10.3f}")

    # cProfile cumulé sur predict() complet -> où passe le temps, fonction par fonction.
    print("\n--- cProfile (predict() complet, fonctions les plus coûteuses) ---")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.iters):
        predictor.predict(payload)
    profiler.disable()
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
