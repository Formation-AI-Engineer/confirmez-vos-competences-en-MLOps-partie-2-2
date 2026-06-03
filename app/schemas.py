"""Schémas Pydantic pour l'entrée/sortie de l'API de scoring crédit."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Données d'un client pour la prédiction.

    Le modèle attend 804 features déjà encodées (issues du feature engineering du
    Projet 6). On accepte un **sous-ensemble** : les features absentes sont traitées
    comme NaN, ce que LightGBM gère nativement. Les clés inconnues sont ignorées.
    """

    features: dict[str, float] = Field(
        ...,
        min_length=1,
        description=(
            "Dictionnaire {nom_feature: valeur} parmi les 804 features du modèle. "
            "Sous-ensemble accepté ; les manquantes sont traitées comme NaN."
        ),
        examples=[
            {
                "AMT_CREDIT": 500000.0,
                "AMT_INCOME_TOTAL": 180000.0,
                "AMT_ANNUITY": 25000.0,
                "DAYS_BIRTH": -12000.0,
                "DAYS_EMPLOYED": -2000.0,
            }
        ],
    )


class PredictionOutput(BaseModel):
    """Résultat de la prédiction de risque de défaut."""

    probability: float = Field(
        ..., ge=0, le=1, description="Probabilité de défaut de paiement (classe 1)"
    )
    decision: Literal["accordé", "refusé"] = Field(
        ..., description="Décision métier (refus si probabilité >= seuil)"
    )
    threshold: float = Field(
        ..., ge=0, le=1, description="Seuil de décision utilisé (refus si proba >= seuil)"
    )
    n_features_received: int = Field(
        ..., description="Nombre de features reconnues parmi celles attendues par le modèle"
    )
    n_features_expected: int = Field(
        ..., description="Nombre total de features attendues par le modèle (804)"
    )


class ModelInfo(BaseModel):
    """Métadonnées du modèle déployé."""

    model_config = {"protected_namespaces": ()}

    model_type: str
    n_features: int
    decision_threshold: float
    dataset_version: str
    train_size: int
