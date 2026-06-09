"""Dashboard de monitoring de production (étape 3.4) — Streamlit.

Lit la base PostgreSQL des appels journalisés (``DATABASE_URL``) et le rapport de
drift (``monitoring/reports/drift_report.json``) pour afficher :
- des **KPI** : volume d'appels, taux d'erreur, taux de refus, latence (moy / p95) ;
- la **distribution des scores prédits** (avec le seuil de décision métier) ;
- la **latence** d'inférence ;
- le **débit** et les **erreurs** dans le temps ;
- les **indicateurs de drift** (colonnes dérivantes, score prédit).

Lancement (extra ``monitoring`` installé, base remplie) ::

    streamlit run monitoring/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import psycopg
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # rend le paquet ``app`` importable hors install

from app import config  # noqa: E402

REPORT_JSON = ROOT / "monitoring" / "reports" / "drift_report.json"
REPORT_HTML = ROOT / "monitoring" / "reports" / "drift_report.html"
PREDICTION_COL = "prediction"


@st.cache_data(ttl=30)
def load_logs() -> pd.DataFrame:
    """Charge les appels journalisés depuis PostgreSQL (cache 30 s)."""
    try:
        with psycopg.connect(config.DATABASE_URL) as conn:
            df = pd.read_sql(
                "SELECT ts, latency_ms, http_status, probability, decision "
                "FROM predictions ORDER BY ts",
                conn,
            )
    except Exception:
        # Base inaccessible (non démarrée, mauvaise URL) -> dashboard vide + message.
        return pd.DataFrame()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


@st.cache_data(ttl=30)
def load_drift() -> tuple[list[dict], float | None]:
    """Lit le rapport de drift JSON -> (colonnes, part de colonnes dérivantes).

    Drift par colonne déduit de la ``value`` et du ``config`` : pour les méthodes
    *p_value* (K-S, khi-deux, Z-test), drift si ``value < seuil``.
    """
    if not REPORT_JSON.exists():
        return [], None
    data = json.loads(REPORT_JSON.read_text())
    columns, share = [], None
    for metric in data.get("metrics", []):
        cfg = metric.get("config", {})
        mtype = cfg.get("type", "")
        if mtype.endswith("DriftedColumnsCount"):
            share = metric.get("value", {}).get("share")
        elif mtype.endswith("ValueDrift"):
            method, threshold, value = cfg.get("method", ""), cfg["threshold"], metric["value"]
            drifted = value < threshold if "p_value" in method else value > threshold
            columns.append(
                {"colonne": cfg["column"], "test": method, "p_value": value, "drift": drifted}
            )
    columns.sort(key=lambda c: (not c["drift"], c["p_value"]))
    return columns, share


def main() -> None:
    st.set_page_config(page_title="Monitoring — Scoring crédit", layout="wide")
    st.title("📊 Monitoring de production — API de scoring crédit")

    df = load_logs()
    if df.empty:
        st.warning(
            "Aucune donnée (base PostgreSQL vide ou inaccessible). Vérifie `DATABASE_URL`, "
            "lance l'API puis `python scripts/simulate_traffic.py` pour remplir la base."
        )
        st.stop()

    ok = df[df["http_status"] == 200]
    threshold = config.DECISION_THRESHOLD

    st.caption(
        f"{len(df):,} appels — du {df['ts'].min():%Y-%m-%d %H:%M:%S} "
        f"au {df['ts'].max():%Y-%m-%d %H:%M:%S} (UTC) · seuil de décision = {threshold}"
    )

    # --- KPI ---
    error_rate = (df["http_status"] != 200).mean()
    refus_rate = (ok["decision"] == "refusé").mean() if len(ok) else 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Appels", f"{len(df):,}")
    c2.metric("Taux d'erreur", f"{error_rate:.1%}")
    c3.metric("Taux de refus", f"{refus_rate:.1%}")
    c4.metric("Latence moy.", f"{ok['latency_ms'].mean():.1f} ms")
    c5.metric("Latence p95", f"{ok['latency_ms'].quantile(0.95):.1f} ms")

    st.divider()

    # --- Distribution des scores & latence ---
    left, right = st.columns(2)
    with left:
        st.subheader("Distribution des scores prédits")
        fig = px.histogram(ok, x="probability", color="decision", nbins=50, opacity=0.8)
        fig.add_vline(
            x=threshold, line_dash="dash", line_color="red", annotation_text=f"seuil {threshold}"
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

    # --- Débit & erreurs dans le temps (par seconde : trafic en rafale) ---
    st.subheader("Débit et erreurs dans le temps")
    per_sec = (
        df.set_index("ts")
        .assign(erreur=lambda d: d["http_status"] != 200)
        .resample("1s")
        .agg(appels=("http_status", "size"), erreurs=("erreur", "sum"))
        .reset_index()
    )
    fig = px.bar(per_sec, x="ts", y="appels")
    fig.update_layout(xaxis_title="Temps (UTC)", yaxis_title="Appels / seconde")
    st.plotly_chart(fig, use_container_width=True)
    if per_sec["erreurs"].sum() == 0:
        st.success("Aucune erreur d'inférence sur la période.")
    else:
        st.error(f"{int(per_sec['erreurs'].sum())} erreur(s) d'inférence (HTTP 500).")

    st.divider()

    # --- Indicateurs de drift ---
    st.subheader("Indicateurs de data drift")
    columns, share = load_drift()
    if not columns:
        st.info(
            "Pas de rapport de drift. Lance `python scripts/analyze_drift.py` "
            "pour générer `monitoring/reports/drift_report.json`."
        )
    else:
        n_drift = sum(c["drift"] for c in columns)
        d1, d2 = st.columns([1, 3])
        d1.metric(
            "Colonnes dérivantes",
            f"{n_drift}/{len(columns)}",
            delta=f"{share:.0%} du total" if share is not None else None,
        )
        pred = next((c for c in columns if c["colonne"] == PREDICTION_COL), None)
        if pred and pred["drift"]:
            d1.error(
                "⚠️ Le **score prédit** a dérivé : "
                "revue de performance / ré-entraînement à envisager."
            )
        elif pred:
            d1.success("Score prédit stable.")
        table = pd.DataFrame(columns)
        table["drift"] = table["drift"].map({True: "⚠️ OUI", False: "non"})
        table["p_value"] = table["p_value"].round(4)
        d2.dataframe(table, use_container_width=True, hide_index=True)
        if REPORT_HTML.exists():
            st.caption(f"Rapport Evidently détaillé : `{REPORT_HTML.relative_to(ROOT)}`")


if __name__ == "__main__":
    main()
