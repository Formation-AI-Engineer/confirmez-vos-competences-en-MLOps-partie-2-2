"""Analyse du data drift (étape 3.3) — référence (Projet 6) vs production (API).

Compare, avec **Evidently 0.7**, deux distributions :
- les **features surveillées** (``MONITORED_FEATURES``) telles que reçues en prod
  vs l'échantillon de validation du Projet 6 ;
- le **score prédit** (``prediction``) : probabilité journalisée en prod vs
  référence re-scorée par le modèle déployé.

Sorties (dans ``monitoring/reports/``) :
- ``drift_report.html`` : rapport Evidently interactif (livrable) ;
- ``drift_report.json`` : résultats bruts (exploitables par le dashboard 3.4) ;
- un **résumé console** : nombre de features dérivantes + liste triée par
  intensité de drift (points de vigilance).

Prérequis : base de prod remplie (``scripts/simulate_traffic.py``) et extra
``monitoring`` installé (``uv sync --extra monitoring``). ::

    python scripts/analyze_drift.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # rend le paquet ``app`` importable hors install

from app import config, predictor  # noqa: E402
from app.monitoring import MONITORED_FEATURES  # noqa: E402
from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

REFERENCE_PATH = ROOT / "monitoring" / "reference_sample.parquet"
REPORT_DIR = ROOT / "monitoring" / "reports"
PREDICTION_COL = "prediction"  # colonne du score prédit, surveillée comme une feature


def load_reference() -> pd.DataFrame:
    """Référence = échantillon de validation, re-scoré par le modèle déployé.

    On garde les features surveillées + le score prédit, pour comparer aussi la
    distribution des **sorties** (et pas seulement des entrées).
    """
    ref = pd.read_parquet(REFERENCE_PATH)
    names = predictor.get_feature_names()
    model = predictor.load_model()
    X = ref.reindex(columns=names).astype("float64")
    ref = ref.copy()
    ref[PREDICTION_COL] = model.predict_proba(X)[:, 1]
    cols = [f for f in MONITORED_FEATURES if f in ref.columns] + [PREDICTION_COL]
    return ref[cols]


def load_production() -> pd.DataFrame:
    """Production = features + proba journalisées (appels réussis uniquement)."""
    with sqlite3.connect(config.LOG_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT features, probability FROM predictions WHERE http_status = 200"
        ).fetchall()
    if not rows:
        raise SystemExit(
            "Aucune prédiction en base. Lance d'abord scripts/simulate_traffic.py."
        )
    feats = pd.json_normalize([json.loads(r[0]) for r in rows])
    feats = feats.reindex(columns=MONITORED_FEATURES)  # aligne + manquantes -> NaN
    feats[PREDICTION_COL] = [r[1] for r in rows]
    cols = [f for f in MONITORED_FEATURES if f in feats.columns] + [PREDICTION_COL]
    return feats[cols]


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numériques vs catégorielles : faible cardinalité (<=10) -> catégorielle.

    Après le feature engineering du Projet 6 tout est encodé en nombres, mais
    les binaires (``CODE_GENDER``, ``*_Married``…) doivent être traités en
    catégoriel pour qu'Evidently applique un test du khi-deux plutôt que K-S.
    """
    numerical, categorical = [], []
    for c in df.columns:
        if c == PREDICTION_COL:
            numerical.append(c)
            continue
        (categorical if df[c].nunique(dropna=True) <= 10 else numerical).append(c)
    return numerical, categorical


def summarise(result, html_path: Path) -> None:
    """Imprime un résumé lisible : colonnes dérivantes triées par intensité.

    Chaque ``ValueDrift`` porte sa ``value`` et sa ``config`` (colonne, méthode,
    seuil). Pour les méthodes *p_value* (K-S, khi-deux, Z-test) le drift est
    détecté si ``value < seuil`` ; pour les méthodes de distance, si
    ``value > seuil``.
    """
    columns = []
    for metric in result.dict().get("metrics", []):
        cfg = metric.get("config", {})
        if not cfg.get("type", "").endswith("ValueDrift"):
            continue
        method, threshold, value = cfg.get("method", ""), cfg["threshold"], metric["value"]
        drifted = value < threshold if "p_value" in method else value > threshold
        columns.append((drifted, cfg["column"], method, threshold, value))

    # Dérivantes d'abord, puis par p-value croissante (plus c'est bas, plus c'est net).
    columns.sort(key=lambda r: (not r[0], r[4]))
    n_drift = sum(c[0] for c in columns)

    print(f"\nRapport HTML : {html_path}")
    print(f"Colonnes dérivantes : {n_drift}/{len(columns)}")
    print(f"\n{'DRIFT':<6}{'colonne':<42}{'test':<18}{'seuil':<7}p-value")
    for drifted, column, method, threshold, value in columns:
        tag = " (score prédit)" if column == PREDICTION_COL else ""
        print(f"{'OUI' if drifted else 'non':<6}{column + tag:<42}{method:<18}{threshold:<7}{round(value, 4)}")


def main() -> None:
    ref = load_reference()
    cur = load_production()
    common = [c for c in ref.columns if c in cur.columns]
    ref, cur = ref[common], cur[common]
    print(f"Référence : {ref.shape[0]} lignes | Production : {cur.shape[0]} lignes")
    print(f"Colonnes comparées : {len(common)} (dont le score 'prediction')")

    numerical, categorical = split_columns(pd.concat([ref, cur]))
    data_def = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    ref_ds = Dataset.from_pandas(ref, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur, data_definition=data_def)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(current_data=cur_ds, reference_data=ref_ds)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORT_DIR / "drift_report.html"
    result.save_html(str(html_path))
    (REPORT_DIR / "drift_report.json").write_text(result.json())

    summarise(result, html_path)


if __name__ == "__main__":
    main()
