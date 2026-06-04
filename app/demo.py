"""Interface de démonstration Gradio — montée par-dessus l'API FastAPI.

Petit frontend interactif pour la tâche 2.1 (optionnel) : saisir/charger les
features d'un client puis obtenir la **probabilité de défaut** et la **décision
métier**. Tourne dans le même process que l'API (appel direct de ``predictor``,
pas d'aller-retour HTTP) et est exposé sous ``/demo`` du Space Docker.
"""

import json
from pathlib import Path

import gradio as gr

from app import config, predictor

_EXAMPLES_PATH = Path(__file__).resolve().parent / "demo_examples.json"


def _load_examples() -> dict[str, dict]:
    """Charge quelques clients réels d'exemple (profils variés). Vide si absent."""
    if _EXAMPLES_PATH.exists():
        return json.loads(_EXAMPLES_PATH.read_text(encoding="utf-8"))
    return {}


def _fill_example(label: str, examples: dict[str, dict]) -> str:
    """Renvoie le JSON (indenté) des features du client d'exemple sélectionné."""
    feats = examples.get(label, {})
    return json.dumps(feats, ensure_ascii=False, indent=2)


def _predict_from_json(features_json: str) -> tuple[dict, str]:
    """Parse le JSON saisi, appelle le modèle et formate le résultat.

    Returns un couple (objet pour ``gr.Label``, markdown récapitulatif). Les
    erreurs (JSON invalide, type non numérique) sont renvoyées proprement à l'UI.
    """
    try:
        features = json.loads(features_json)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"JSON invalide : {exc}") from exc

    if not isinstance(features, dict) or not features:
        raise gr.Error("Fournir un objet JSON non vide {nom_feature: valeur}.")

    try:
        features = {k: float(v) for k, v in features.items()}
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"Toutes les valeurs doivent être numériques : {exc}") from exc

    result = predictor.predict(features)
    proba = result["probability"]
    decision = result["decision"]

    # gr.Label attend {classe: confiance} : on montre les deux décisions possibles.
    label_obj = {
        "refusé": proba,
        "accordé": round(1.0 - proba, 4),
    }
    recap = (
        f"**Décision : {decision.upper()}**\n\n"
        f"- Probabilité de défaut : **{proba:.1%}**\n"
        f"- Seuil de refus : **{result['threshold']:.0%}** "
        f"(refus si proba ≥ seuil)\n"
        f"- Features reconnues : **{result['n_features_received']} / "
        f"{result['n_features_expected']}** (les manquantes → NaN)"
    )
    return label_obj, recap


def build_demo() -> gr.Blocks:
    """Construit l'interface Gradio (Blocks) de démonstration de l'API."""
    examples = _load_examples()
    example_labels = list(examples.keys())
    default_json = (
        _fill_example(example_labels[0], examples)
        if example_labels
        else json.dumps(
            {"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 180000, "DAYS_BIRTH": -12000},
            ensure_ascii=False,
            indent=2,
        )
    )

    with gr.Blocks(title="Démo — Scoring crédit Prêt à Dépenser") as demo:
        gr.Markdown(
            "# 💳 Démo — Scoring crédit *Prêt à Dépenser*\n"
            "Frontend de démonstration de l'API de scoring "
            f"(modèle LightGBM, **{predictor.get_model_info()['n_features']} features**, "
            f"seuil métier **{config.DECISION_THRESHOLD:.0%}**). "
            "La classe positive est le **défaut de paiement** : proba ≥ seuil → **refus**.\n\n"
            "👉 API & documentation Swagger : [`/docs`](/docs)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                if example_labels:
                    example_dd = gr.Dropdown(
                        choices=example_labels,
                        value=example_labels[0],
                        label="Charger un client d'exemple",
                        info="Profils réels issus de l'échantillon de référence.",
                    )
                features_in = gr.Code(
                    value=default_json,
                    language="json",
                    label="Features du client (JSON éditable)",
                )
                predict_btn = gr.Button("Prédire", variant="primary")
            with gr.Column(scale=1):
                decision_out = gr.Label(label="Décision (confiance par classe)")
                recap_out = gr.Markdown(label="Détail")

        if example_labels:
            example_dd.change(
                fn=lambda label: _fill_example(label, examples),
                inputs=example_dd,
                outputs=features_in,
            )
        predict_btn.click(
            fn=_predict_from_json,
            inputs=features_in,
            outputs=[decision_out, recap_out],
        )

    return demo
