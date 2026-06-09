"""Export du modèle LightGBM en ONNX (étape 4.2 — benchmark seulement).

Convertit le ``LGBMClassifier`` déployé en ``models/lgbm_final.onnx`` afin de
**comparer** ONNX Runtime aux stratégies numpy/booster dans
``scripts/benchmark_optimization.py``.

⚠️ ONNX **n'est pas adopté en production** (cf. ``docs/etape4_optimisation_performances.md``) :
le gain absolu vs ``booster_.predict`` est négligeable (~0,03 ms) alors qu'ONNX
ajoute une dépendance runtime et une conversion float32 (écart de score ~1e-7).
Ce script et l'artefact ``.onnx`` ne servent qu'à **justifier ce choix par des
chiffres reproductibles**.

Prérequis : ``uv sync --extra onnx``. ::

    python scripts/export_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # rend le paquet ``app`` importable hors install

import onnx  # noqa: E402
from onnxmltools.convert import convert_lightgbm  # noqa: E402
from onnxmltools.convert.common.data_types import FloatTensorType  # noqa: E402

from app import predictor  # noqa: E402

ONNX_PATH = ROOT / "models" / "lgbm_final.onnx"


def main() -> None:
    model = predictor.load_model()
    n_features = model.n_features_

    # zipmap=False : sortie "probabilities" en tableau dense (pas une liste de
    # dicts), directement indexable par le benchmark.
    onx = convert_lightgbm(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        zipmap=False,
    )
    onnx.save_model(onx, str(ONNX_PATH))

    print(f"ONNX exporté : {ONNX_PATH} ({n_features} features)")
    print(f"Sorties : {[o.name for o in onx.graph.output]}")


if __name__ == "__main__":
    main()
