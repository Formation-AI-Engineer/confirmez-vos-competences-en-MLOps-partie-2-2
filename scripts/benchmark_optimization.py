"""Benchmark des stratégies d'inférence (étape 4.2).

Compare, sur des clients réels de la référence, plusieurs façons de produire la
**même** probabilité de défaut, pour choisir la plus rapide **sans régression** :

- ``baseline`` : DataFrame pandas + ``predict_proba`` (implémentation initiale) ;
- ``numpy`` : vecteur numpy pré-aligné + ``predict_proba`` (supprime le glue pandas) ;
- ``booster`` : vecteur numpy + ``booster_.predict`` (court-circuite le wrapper sklearn) ;
- ``onnx`` *(si disponible)* : ONNX Runtime sur le modèle exporté.

Pour chaque stratégie : vérifie l'**écart de score max** vs baseline (non-régression)
puis mesure la latence (moy / médiane / p95). ::

    python scripts/benchmark_optimization.py --iters 3000
"""

from __future__ import annotations

import argparse
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
ONNX_PATH = ROOT / "models" / "lgbm_final.onnx"


def sample_payloads(n: int) -> list[dict]:
    """n clients réels (features surveillées), comme reçus par l'API."""
    ref = pd.read_parquet(REFERENCE_PATH)
    payloads = []
    for _, row in ref.head(n).iterrows():
        d = row.to_dict()
        payloads.append({k: float(d[k]) for k in MONITORED_FEATURES if k in d and pd.notna(d[k])})
    return payloads


def build_row(features: dict, names: list[str], index: dict[str, int]) -> np.ndarray:
    """Vecteur numpy 1×804 aligné sur l'ordre du modèle (manquantes -> NaN)."""
    row = np.full((1, len(names)), np.nan, dtype=np.float64)
    for key, value in features.items():
        idx = index.get(key)
        if idx is not None and value is not None:
            row[0, idx] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iters", type=int, default=3000, help="Itérations de timing par stratégie"
    )
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    model = predictor.load_model()
    names = predictor.get_feature_names()
    index = {name: i for i, name in enumerate(names)}
    payloads = sample_payloads(100)

    # --- Stratégies : payload -> probabilité (classe positive) ---
    def f_baseline(p):
        row = (
            pd.DataFrame([predictor.add_derived_features(p)])
            .reindex(columns=names)
            .astype("float64")
        )
        return float(model.predict_proba(row)[0, 1])

    def f_numpy(p):
        row = build_row(predictor.add_derived_features(p), names, index)
        return float(model.predict_proba(row)[0, 1])

    def f_booster(p):
        row = build_row(predictor.add_derived_features(p), names, index)
        return float(model.booster_.predict(row)[0])

    strategies = {"baseline": f_baseline, "numpy": f_numpy, "booster": f_booster}

    # --- ONNX (optionnel) ---
    onnx_sess = None
    if ONNX_PATH.exists():
        try:
            import onnxruntime as ort

            onnx_sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
            in_name = onnx_sess.get_inputs()[0].name
            # 2e sortie = probabilités (zipmap désactivé à l'export -> tableau)
            prob_name = onnx_sess.get_outputs()[-1].name

            def f_onnx(p):
                row = build_row(predictor.add_derived_features(p), names, index).astype(np.float32)
                out = onnx_sess.run([prob_name], {in_name: row})[0]
                return float(out[0][1])

            strategies["onnx"] = f_onnx
        except Exception as exc:  # noqa: BLE001
            print(f"[onnx] indisponible : {exc}")

    # --- Non-régression : écart de score max vs baseline ---
    base_scores = np.array([f_baseline(p) for p in payloads])
    print("Non-régression (écart de score max vs baseline, sur 100 clients) :")
    for name, fn in strategies.items():
        scores = np.array([fn(p) for p in payloads])
        print(f"  {name:<10} max|Δ| = {np.max(np.abs(scores - base_scores)):.2e}")

    # --- Timing ---
    payload = payloads[0]
    print(f"\nLatence par stratégie ({args.iters} itérations) :")
    print(f"{'stratégie':<12}{'moy (ms)':>10}{'méd (ms)':>10}{'p95 (ms)':>10}{'vs baseline':>13}")
    base_mean = None
    for name, fn in strategies.items():
        for _ in range(args.warmup):
            fn(payload)
        lat = np.empty(args.iters)
        for i in range(args.iters):
            t0 = time.perf_counter()
            fn(payload)
            lat[i] = (time.perf_counter() - t0) * 1000
        mean = lat.mean()
        base_mean = base_mean or mean
        speedup = f"{base_mean / mean:.2f}×" if name != "baseline" else "—"
        print(
            f"{name:<12}{mean:>10.3f}{np.median(lat):>10.3f}"
            f"{np.percentile(lat, 95):>10.3f}{speedup:>13}"
        )


if __name__ == "__main__":
    main()
