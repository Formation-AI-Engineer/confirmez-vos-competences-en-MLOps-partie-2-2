"""Interface de démonstration Gradio — montée par-dessus l'API FastAPI.

Frontend interactif pour la tâche 2.1 (optionnel) : saisir un client via un
formulaire des features **les plus déterminantes** du modèle (scores externes
`EXT_SOURCE_*`, montants, âge/ancienneté en années), charger un client réel
d'exemple, ou passer par le JSON avancé pour atteindre n'importe quelle feature ;
puis obtenir la **probabilité de défaut** et la **décision métier**. Tourne dans
le même process que l'API (appel direct de ``predictor``), sous ``/demo``.

Comme l'endpoint ``/predict``, chaque prédiction est **journalisée** (best-effort)
pour le monitoring : les tests via la démo apparaissent donc dans le dashboard.
"""

import json
import time
from pathlib import Path

import gradio as gr

from app import config, monitoring, predictor

_EXAMPLES_PATH = Path(__file__).resolve().parent / "demo_examples.json"
_NEW_CLIENT = "➕ Nouveau client (saisie libre)"
_DAYS_PER_YEAR = 365.25

# Champs du formulaire = features les plus influentes du modèle (cf. importances).
# Types de widget : "slider" (EXT_SOURCE 0–1, ↑ = moins risqué), "number" (montants),
# "years" (saisi en années par l'utilisateur → converti en jours négatifs DAYS_*).
# (clé modèle, label, valeur par défaut affichée, type)
_FORM_FIELDS = [
    ("EXT_SOURCE_1", "Score externe 1 (0–1, ↑ = moins risqué)", 0.5, "slider"),
    ("EXT_SOURCE_2", "Score externe 2 (0–1, ↑ = moins risqué)", 0.5, "slider"),
    ("EXT_SOURCE_3", "Score externe 3 (0–1, ↑ = moins risqué)", 0.5, "slider"),
    ("AMT_CREDIT", "Montant du crédit (€)", 500000.0, "number"),
    ("AMT_ANNUITY", "Annuité annuelle (€) — sert au CREDIT_TERM", 25000.0, "number"),
    ("AMT_INCOME_TOTAL", "Revenu annuel total (€)", 180000.0, "number"),
    ("DAYS_BIRTH", "Âge (années)", 33.0, "years"),
    ("DAYS_EMPLOYED", "Ancienneté dans l'emploi (années)", 5.5, "years"),
]
_FORM_KEYS = [key for key, _, _, _ in _FORM_FIELDS]
_DEFAULTS = [default for _, _, default, _ in _FORM_FIELDS]
_DRIVERS = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}


def _to_display(value: float | None, kind: str) -> float | None:
    """Valeur modèle → valeur affichée (jours négatifs → années pour ``years``)."""
    if value is None:
        return None
    return round(-value / _DAYS_PER_YEAR, 1) if kind == "years" else value


def _to_model(value: float, kind: str) -> float:
    """Valeur saisie → valeur modèle (années → jours négatifs pour ``years``)."""
    return -value * _DAYS_PER_YEAR if kind == "years" else value


def _load_examples() -> dict[str, dict]:
    """Charge quelques clients réels d'exemple (profils variés). Vide si absent."""
    if _EXAMPLES_PATH.exists():
        return json.loads(_EXAMPLES_PATH.read_text(encoding="utf-8"))
    return {}


def _fill_form(label: str, examples: dict[str, dict]) -> tuple:
    """Renvoie l'état de base + les valeurs des champs pour la sélection.

    - « Nouveau client » → base vide + valeurs par défaut.
    - Client d'exemple → features complètes en base + champs pré-remplis (âge/
      ancienneté reconvertis en années pour l'affichage).
    """
    if label == _NEW_CLIENT or label not in examples:
        return ({}, *_DEFAULTS, "")
    feats = examples[label]
    values = (_to_display(feats.get(key), kind) for key, _, _, kind in _FORM_FIELDS)
    return (feats, *values, "")


def _predict(base: dict | None, *form_and_json) -> tuple[dict, str]:
    """Assemble les features (base + formulaire + JSON avancé), prédit et formate.

    Précédence : client de base (exemple) < champs du formulaire < JSON avancé.
    La feature dérivée ``CREDIT_TERM`` est calculée côté ``predictor`` (partagée
    avec l'API), donc la démo et Swagger réagissent de façon identique.
    """
    *form_values, advanced_json = form_and_json
    feats: dict[str, float] = dict(base or {})

    # Champs du formulaire (ignorés si laissés vides), avec conversion années → jours.
    for (key, _, _, kind), value in zip(_FORM_FIELDS, form_values, strict=True):
        if value is not None:
            feats[key] = float(_to_model(value, kind))

    # JSON avancé optionnel (override final) pour atteindre n'importe quelle feature.
    if advanced_json and advanced_json.strip():
        try:
            extra = json.loads(advanced_json)
        except json.JSONDecodeError as exc:
            raise gr.Error(f"JSON avancé invalide : {exc}") from exc
        if not isinstance(extra, dict):
            raise gr.Error("Le JSON avancé doit être un objet {feature: valeur}.")
        try:
            feats.update({k: float(v) for k, v in extra.items()})
        except (TypeError, ValueError) as exc:
            raise gr.Error(f"JSON avancé : valeurs non numériques ({exc}).") from exc

    if not feats:
        raise gr.Error("Renseigner au moins une feature.")

    # Même chemin journalisé que POST /predict : on mesure la latence et on
    # enregistre l'appel (best-effort) pour que la démo alimente le monitoring.
    start = time.perf_counter()
    result = predictor.predict(feats)
    monitoring.log_prediction(
        features=predictor.add_derived_features(feats),
        latency_ms=(time.perf_counter() - start) * 1000,
        http_status=200,
        result=result,
    )
    proba = result["probability"]
    decision = result["decision"]

    label_obj = {"refusé": proba, "accordé": round(1.0 - proba, 4)}

    # Avertissement de fiabilité : sans aucun score externe, les features les plus
    # déterminantes sont NaN → la décision reste proche du taux de base.
    warning = ""
    if not _DRIVERS & feats.keys():
        warning = (
            "> ⚠️ **Prédiction indicative** : aucun score externe `EXT_SOURCE_*` "
            "(features les plus déterminantes) n'est fourni, la décision reste donc "
            "proche du taux de base. Renseignez-les ou chargez un **client d'exemple**.\n\n"
        )

    recap = warning + (
        f"**Décision : {decision.upper()}**\n\n"
        f"- Probabilité de défaut : **{proba:.1%}**\n"
        f"- Seuil de refus : **{result['threshold']:.0%}** (refus si proba ≥ seuil)\n"
        f"- Features fournies au modèle : **{result['n_features_received']} / "
        f"{result['n_features_expected']}** (les manquantes → NaN, gérées par LightGBM)"
    )
    return label_obj, recap


def build_demo() -> gr.Blocks:
    """Construit l'interface Gradio (Blocks) de démonstration de l'API."""
    examples = _load_examples()
    choices = list(examples.keys())
    init_label = next(iter(examples), _NEW_CLIENT)
    init = _fill_form(init_label, examples)

    with gr.Blocks(title="Démo — Scoring crédit Prêt à Dépenser") as demo:
        gr.Markdown(
            "# 💳 Démo — Scoring crédit *Prêt à Dépenser*\n"
            "Saisissez un client (ou chargez un profil réel d'exemple) pour obtenir la "
            f"probabilité de défaut et la décision. Modèle LightGBM "
            f"(**{predictor.get_model_info()['n_features']} features**), "
            f"seuil métier **{config.DECISION_THRESHOLD:.0%}** — proba ≥ seuil → **refus**.\n\n"
            "Le formulaire expose les features **les plus déterminantes** "
            "(`EXT_SOURCE_*`, montants, âge/ancienneté en années) ; les ~796 autres "
            "restent NaN (gérées par LightGBM) ou se renseignent via *Avancé*.\n\n"
            "👉 API & documentation Swagger : [`/docs`](/docs)."
        )

        base_state = gr.State(init[0])

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    selector = gr.Dropdown(
                        choices=choices,
                        value=init_label,
                        label="Client d'exemple",
                        info="Profils réels issus de l'échantillon de référence.",
                        scale=4,
                    )
                    new_btn = gr.Button("➕ Nouveau client", scale=1)

                def _make_field(spec, val):
                    _, lbl, _, kind = spec
                    if kind == "slider":
                        return gr.Slider(0.0, 1.0, value=val, step=0.01, label=lbl)
                    return gr.Number(value=val, label=lbl)

                # Champs disposés 2 par ligne.
                fields = []
                form_vals = init[1:-1]
                for i in range(0, len(_FORM_FIELDS), 2):
                    with gr.Row():
                        for spec, val in zip(
                            _FORM_FIELDS[i : i + 2], form_vals[i : i + 2], strict=True
                        ):
                            fields.append(_make_field(spec, val))
                with gr.Accordion("Avancé — autres features (JSON)", open=False):
                    advanced = gr.Code(
                        value="",
                        language="json",
                        label='Override JSON, ex. {"CODE_GENDER": 1, "OWN_CAR_AGE": 5}',
                    )
                predict_btn = gr.Button("Prédire", variant="primary")
            with gr.Column(scale=1):
                decision_out = gr.Label(label="Décision (confiance par classe)")
                recap_out = gr.Markdown()

        selector.change(
            fn=lambda label: _fill_form(label, examples),
            inputs=selector,
            outputs=[base_state, *fields, advanced],
        )
        new_btn.click(
            fn=lambda: _fill_form(_NEW_CLIENT, examples),
            outputs=[base_state, *fields, advanced],
        )
        predict_btn.click(
            fn=_predict,
            inputs=[base_state, *fields, advanced],
            outputs=[decision_out, recap_out],
        )

    return demo
