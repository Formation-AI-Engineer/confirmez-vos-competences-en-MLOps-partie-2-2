"""Tests de l'interface de démo Gradio montée sous /demo."""

import json

import gradio as gr
import pytest

from app.demo import _load_examples, _predict_from_json


def test_demo_mounted(client):
    """L'interface Gradio est servie sous /demo."""
    r = client.get("/demo")
    assert r.status_code == 200


def test_demo_predict_from_json(client):
    """Une prédiction depuis le JEU JSON renvoie un label et un récap cohérents."""
    features = json.dumps({"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 180000})
    label, recap = _predict_from_json(features)
    assert set(label) == {"accordé", "refusé"}
    assert abs(label["accordé"] + label["refusé"] - 1.0) < 1e-6
    assert "Décision" in recap


def test_demo_invalid_json_raises(client):
    """Un JSON invalide lève une gr.Error (message propre côté UI)."""
    with pytest.raises(gr.Error):
        _predict_from_json("{pas du json}")


def test_demo_examples_present():
    """Les clients d'exemple sont bien embarqués pour la démo."""
    examples = _load_examples()
    assert len(examples) >= 1
    first = next(iter(examples.values()))
    assert isinstance(first, dict) and len(first) > 0
