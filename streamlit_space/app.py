"""Dashboard de monitoring de production — Space Streamlit (lit Neon).

Application **autonome** déployée sur un Space Hugging Face dédié
(`scoring-credit-stream`). Elle lit la base **PostgreSQL Neon** alimentée en
production par l'API de scoring (chaque appel à `/predict` y est journalisé) et
affiche :
- des **KPI** (volume, taux d'erreur, taux de refus, latence moy / p95) ;
- la **distribution des scores** prédits (avec le seuil métier) ;
- la **latence** et le **débit** dans le temps ;
- à la demande, le **data drift** (Evidently) vs une référence pré-scorée embarquée.

Configuration : le secret **`DATABASE_URL`** (URL Neon) doit être défini dans les
*Settings → Secrets* du Space. Aucun modèle n'est nécessaire ici : la référence est
déjà scorée (`reference_scored.parquet`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import psycopg
import streamlit as st

HERE = Path(__file__).resolve().parent
REFERENCE_PATH = HERE / "reference_scored.parquet"
DATABASE_URL = os.getenv("DATABASE_URL", "")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.49"))
PREDICTION_COL = "prediction"


@st.cache_data(ttl=30)
def load_logs() -> pd.DataFrame:
    """KPI / séries temporelles : ts, latence, statut, proba, décision (cache 30 s)."""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            df = pd.read_sql(
                "SELECT ts, latency_ms, http_status, probability, decision "
                "FROM predictions ORDER BY ts",
                conn,
            )
    except Exception as exc:
        st.error(f"Connexion à la base impossible : {exc}")
        return pd.DataFrame()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


@st.cache_data(ttl=30)
def load_production_features() -> pd.DataFrame:
    """Features + proba des appels réussis, pour le calcul de drift (cache 30 s)."""
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT features, probability FROM predictions WHERE http_status = 200"
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    feats = pd.json_normalize([r[0] for r in rows])  # JSONB -> dict
    feats[PREDICTION_COL] = [r[1] for r in rows]
    return feats


def compute_drift(reference: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    """Drift par colonne via Evidently (K-S / khi-deux, p < 0,05)."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    common = [c for c in reference.columns if c in production.columns]
    ref, cur = reference[common], production[common]

    numerical, categorical = [], []
    for c in common:
        if c == PREDICTION_COL or pd.concat([ref[c], cur[c]]).nunique(dropna=True) > 10:
            numerical.append(c)
        else:
            categorical.append(c)

    data_def = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    ref_ds = Dataset.from_pandas(ref, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur, data_definition=data_def)
    result = Report(metrics=[DataDriftPreset()]).run(current_data=cur_ds, reference_data=ref_ds)

    rows = []
    for metric in result.dict().get("metrics", []):
        cfg = metric.get("config", {})
        if not cfg.get("type", "").endswith("ValueDrift"):
            continue
        method, threshold, value = cfg.get("method", ""), cfg["threshold"], metric["value"]
        drifted = value < threshold if "p_value" in method else value > threshold
        rows.append(
            {"colonne": cfg["column"], "test": method, "p_value": round(value, 4), "drift": drifted}
        )
    rows.sort(key=lambda r: (not r["drift"], r["p_value"]))
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Monitoring — Scoring crédit", layout="wide")
    st.title("📊 Monitoring de production — API de scoring crédit")

    # Les données sont mises en cache 30 s ; ce bouton force un rechargement immédiat.
    if st.button("🔄 Rafraîchir les données"):
        st.cache_data.clear()
        st.rerun()

    if not DATABASE_URL:
        st.error("Le secret `DATABASE_URL` (URL Neon) n'est pas défini dans les Settings du Space.")
        st.stop()

    df = load_logs()
    if df.empty:
        st.warning(
            "Aucune donnée en base. L'API doit recevoir du trafic (`/predict`) pour remplir Neon."
        )
        st.stop()

    ok = df[df["http_status"] == 200]
    st.caption(
        f"{len(df):,} appels — du {df['ts'].min():%Y-%m-%d %H:%M} "
        f"au {df['ts'].max():%Y-%m-%d %H:%M} (UTC) · seuil de décision = {DECISION_THRESHOLD}"
    )

    # --- KPI ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Appels", f"{len(df):,}")
    c2.metric("Taux d'erreur", f"{(df['http_status'] != 200).mean():.1%}")
    c3.metric("Taux de refus", f"{(ok['decision'] == 'refusé').mean() if len(ok) else 0:.1%}")
    c4.metric("Latence moy.", f"{ok['latency_ms'].mean():.1f} ms")
    c5.metric("Latence p95", f"{ok['latency_ms'].quantile(0.95):.1f} ms")
    st.divider()

    # --- Distribution des scores & latence ---
    left, right = st.columns(2)
    with left:
        st.subheader("Distribution des scores prédits")
        fig = px.histogram(ok, x="probability", color="decision", nbins=50, opacity=0.8)
        fig.add_vline(
            x=DECISION_THRESHOLD,
            line_dash="dash",
            line_color="red",
            annotation_text=f"seuil {DECISION_THRESHOLD}",
        )
        fig.update_layout(xaxis_title="Probabilité de défaut", yaxis_title="Nombre d'appels")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Latence d'inférence")
        fig = px.histogram(ok, x="latency_ms", nbins=50)
        fig.add_vline(
            x=ok["latency_ms"].quantile(0.95),
            line_dash="dash",
            line_color="orange",
            annotation_text="p95",
        )
        fig.update_layout(xaxis_title="Latence (ms)", yaxis_title="Nombre d'appels")
        st.plotly_chart(fig, use_container_width=True)

    # --- Débit dans le temps ---
    st.subheader("Débit et erreurs dans le temps")
    per_min = (
        df.set_index("ts")
        .assign(erreur=lambda d: d["http_status"] != 200)
        .resample("1min")
        .agg(appels=("http_status", "size"), erreurs=("erreur", "sum"))
        .reset_index()
    )
    fig = px.bar(per_min, x="ts", y="appels")
    fig.update_layout(xaxis_title="Temps (UTC)", yaxis_title="Appels / minute")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Data drift (à la demande : Evidently est coûteux à charger) ---
    st.subheader("Indicateurs de data drift")
    st.caption("Comparaison production (Neon) vs référence du Projet 6 (pré-scorée).")
    if st.button("Calculer le drift maintenant"):
        if not REFERENCE_PATH.exists():
            st.info("Référence absente (`reference_scored.parquet`).")
        else:
            production = load_production_features()
            if production.empty:
                st.info("Pas encore d'appels réussis à analyser.")
            else:
                with st.spinner("Analyse Evidently en cours…"):
                    try:
                        table = compute_drift(pd.read_parquet(REFERENCE_PATH), production)
                    except Exception as exc:
                        st.error(f"Drift indisponible : {exc}")
                        table = pd.DataFrame()
                if not table.empty:
                    n_drift = int(table["drift"].sum())
                    d1, d2 = st.columns([1, 3])
                    d1.metric("Colonnes dérivantes", f"{n_drift}/{len(table)}")
                    pred = table[table["colonne"] == PREDICTION_COL]
                    if not pred.empty and bool(pred.iloc[0]["drift"]):
                        d1.error("⚠️ Le score prédit a dérivé : ré-entraînement à envisager.")
                    elif not pred.empty:
                        d1.success("Score prédit stable.")
                    table["drift"] = table["drift"].map({True: "⚠️ OUI", False: "non"})
                    d2.dataframe(table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
