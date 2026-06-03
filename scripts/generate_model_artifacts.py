"""Génère les artefacts JSON du modèle à partir du bundle d'entraînement.

Source de vérité : ``models/lgbm_final_bundle.joblib`` (produit par le Projet 6).
Ce script en dérive trois fichiers, pour que l'API/monitoring n'aient pas besoin
de charger un .joblib juste pour connaître le contrat d'entrée ou les seuils :

  - models/feature_names.json          : liste ordonnée des 804 features attendues
  - models/model_meta.json             : carte d'identité légère (seuils, version…)
  - monitoring/reference_sample.parquet : échantillon de validation figé (100 × 804),
                                          référence pour la détection de drift

Reproductibilité MLOps : ces deux JSON ne doivent jamais être édités à la main,
mais régénérés via ce script. Utiliser ``--check`` en CI pour vérifier qu'ils
sont bien synchronisés avec le bundle.

Usage :
    python scripts/generate_model_artifacts.py          # (re)génère les JSON
    python scripts/generate_model_artifacts.py --check  # vérifie sans écrire (exit 1 si désync)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MONITORING_DIR = ROOT_DIR / "monitoring"
BUNDLE_PATH = MODELS_DIR / "lgbm_final_bundle.joblib"
FEATURES_PATH = MODELS_DIR / "feature_names.json"
META_PATH = MODELS_DIR / "model_meta.json"
REFERENCE_PATH = MONITORING_DIR / "reference_sample.parquet"


def build_artifacts():
    """Charge le bundle et construit les artefacts dérivés.

    Retourne ``(features, meta, reference)`` où ``reference`` est l'échantillon
    de validation figé (DataFrame 100 × 804) servant de référence drift.
    """
    bundle = joblib.load(BUNDLE_PATH)

    reference = bundle["X_val_sample"].reset_index(drop=True)
    features = list(reference.columns)

    meta = {
        "model_type": "LGBMClassifier",
        "n_features": len(features),
        "best_threshold": bundle["best_threshold"],
        "val_best_threshold": bundle["val_best_threshold"],
        "best_score_cv": bundle["best_score_cv"],
        "dataset_version": bundle["dataset_version"],
        "train_size": bundle["train_size"],
    }
    return features, meta, reference


def _dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Vérifie que les JSON sur disque sont à jour, sans rien écrire.",
    )
    args = parser.parse_args()

    if not BUNDLE_PATH.exists():
        print(f"[ERREUR] Bundle introuvable : {BUNDLE_PATH}", file=sys.stderr)
        return 1

    features, meta, reference = build_artifacts()
    new_features = _dumps(features)
    new_meta = _dumps(meta)

    if args.check:
        ok = True
        for path, expected in ((FEATURES_PATH, new_features), (META_PATH, new_meta)):
            current = path.read_text(encoding="utf-8") if path.exists() else None
            if current != expected:
                print(f"[DÉSYNC] {path.name} n'est pas à jour avec le bundle.")
                ok = False

        # Le parquet se compare sur le contenu (DataFrame), pas sur les octets.
        if not REFERENCE_PATH.exists():
            print(f"[DÉSYNC] {REFERENCE_PATH.name} est absent.")
            ok = False
        elif not pd.read_parquet(REFERENCE_PATH).reset_index(drop=True).equals(reference):
            print(f"[DÉSYNC] {REFERENCE_PATH.name} n'est pas à jour avec le bundle.")
            ok = False

        if ok:
            print(f"[OK] Artefacts synchronisés ({len(features)} features).")
            return 0
        print("→ Relancer sans --check pour régénérer.", file=sys.stderr)
        return 1

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.write_text(new_features, encoding="utf-8")
    META_PATH.write_text(new_meta, encoding="utf-8")
    reference.to_parquet(REFERENCE_PATH)
    print(f"[OK] {FEATURES_PATH.name} écrit ({len(features)} features).")
    print(f"[OK] {META_PATH.name} écrit (dataset_version={meta['dataset_version']}).")
    print(f"[OK] {REFERENCE_PATH.name} écrit ({reference.shape[0]} × {reference.shape[1]}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
