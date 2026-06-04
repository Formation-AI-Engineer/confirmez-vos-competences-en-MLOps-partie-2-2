"""Tests de l'interface de démo Gradio montée sous /demo."""

import gradio as gr
import pytest

from app.demo import _DEFAULTS, _load_examples, _predict

# 8 champs de formulaire → None pour « rien saisi ».
_NONE_FORM = [None] * len(_DEFAULTS)


def test_demo_mounted(client):
    """L'interface Gradio est servie sous /demo."""
    r = client.get("/demo")
    assert r.status_code == 200


def test_demo_predict_from_form(client):
    """Une prédiction depuis le formulaire (features influentes) est cohérente."""
    label, recap = _predict({}, *_DEFAULTS, "")
    assert set(label) == {"accordé", "refusé"}
    assert abs(label["accordé"] + label["refusé"] - 1.0) < 1e-6
    assert "Décision" in recap
    # 8 champs saisis + CREDIT_TERM dérivée (annuité/crédit) = 9 features.
    assert "9 / 804" in recap


def test_demo_ext_source_drives_decision(client):
    """Les scores externes pilotent la décision : bas = refusé, haut = accordé."""
    low = list(_DEFAULTS)
    low[0] = low[1] = low[2] = 0.1
    high = list(_DEFAULTS)
    high[0] = high[1] = high[2] = 0.9
    proba_low = _predict({}, *low, "")[0]["refusé"]
    proba_high = _predict({}, *high, "")[0]["refusé"]
    assert proba_low > proba_high


def test_demo_advanced_json_override(client):
    """Le JSON avancé est fusionné par-dessus les champs du formulaire."""
    label, recap = _predict({}, *_NONE_FORM, '{"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 180000}')
    assert "2 / 804" in recap  # seules les 2 features du JSON sont fournies


def test_demo_invalid_advanced_json_raises(client):
    """Un JSON avancé invalide lève une gr.Error (message propre côté UI)."""
    with pytest.raises(gr.Error):
        _predict({}, *_NONE_FORM, "{pas du json}")


def test_demo_empty_input_raises(client):
    """Aucune feature renseignée → gr.Error."""
    with pytest.raises(gr.Error):
        _predict({}, *_NONE_FORM, "")


def test_demo_examples_present():
    """Les clients d'exemple sont bien embarqués pour la démo."""
    examples = _load_examples()
    assert len(examples) >= 1
    first = next(iter(examples.values()))
    assert isinstance(first, dict) and len(first) > 0
