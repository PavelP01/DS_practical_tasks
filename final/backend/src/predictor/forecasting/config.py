"""Shared constants and ``ModelConfig`` for training, CV, and inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

HOLDOUT_DEFAULT = 12
FUTURE_HORIZON_DEFAULT = 12
CV_MIN_TRAIN_MONTHS = 36
CV_N_FOLDS = 4
DATE_COL = "Date"

CITY_GROUP_COLS = ["Country", "State", "City"]
CLUSTER_DRIVERS = [
    "Wind_Speed (km/h)",
    "Humidity (%)",
    "Deforestation_Rate_%",
    "Industry_Growth_%",
    "CO2_Emission_MT",
    "Population_Density_per_SqKm",
]
N_CLUSTERS_DEFAULT = 8
N_KERNEL_PCA_DEFAULT = 3
KERNEL_PCA_KERNEL = "rbf"
KERNEL_PCA_GAMMA = 0.06

PANEL_MODEL_CANDIDATES = (
    "SeasonalNaiveVector",
    "HoltWintersVector",
    "RidgeMultiOutput",
    "WLSRidgeMultiOutput",
    "GradientBoostingVector",
)


@dataclass(frozen=True)
class ModelConfig:
    """Immutable model family name plus sorted hyperparameters for CV and artifacts."""

    model_type: str
    params: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, model_type: str, params: dict[str, Any] | None = None) -> ModelConfig:
        items = tuple(sorted((params or {}).items()))
        return cls(model_type=model_type, params=items)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def label(self) -> str:
        if not self.params:
            return "default"
        return ", ".join(f"{k}={v}" for k, v in self.params)

    def to_json(self) -> str:
        return json.dumps({"model_type": self.model_type, "params": self.as_dict()}, sort_keys=True)


def default_model_config(model_type: str) -> ModelConfig:
    """Return default hyperparameters for a supported ``model_type``."""
    defaults: dict[str, dict[str, Any]] = {
        "SeasonalNaiveVector": {"season_lag": 12, "fallback": "last"},
        "HoltWintersVector": {"trend": "add", "seasonal": "add", "damped_trend": False},
        "RidgeMultiOutput": {"alpha": 1.0},
        "WLSRidgeMultiOutput": {"alpha": 1.0, "vol_window": 6},
        "GradientBoostingVector": {
            "max_depth": 5,
            "max_iter": 250,
            "learning_rate": 0.08,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
    }
    return ModelConfig.from_dict(model_type, defaults.get(model_type, {}))


def model_config_from_json(raw: str | dict[str, Any] | None, model_type: str) -> ModelConfig:
    """Parse ``best_params_json`` from selection tables or artifact metadata."""
    if not raw:
        return default_model_config(model_type)
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return default_model_config(model_type)
    else:
        payload = raw
    if isinstance(payload, dict) and payload.get("model_type"):
        return ModelConfig.from_dict(str(payload["model_type"]), payload.get("params", {}))
    if isinstance(payload, dict):
        return ModelConfig.from_dict(model_type, payload)
    return default_model_config(model_type)


def hyperparam_grids() -> dict[str, list[ModelConfig]]:
    """Short grids for per-cluster tuning (experiments.ipynb §6.4.1)."""
    return {
        "SeasonalNaiveVector": [
            ModelConfig.from_dict("SeasonalNaiveVector", {"season_lag": 12, "fallback": "last"}),
            ModelConfig.from_dict("SeasonalNaiveVector", {"season_lag": 12, "fallback": "mean12"}),
        ],
        "HoltWintersVector": [
            ModelConfig.from_dict(
                "HoltWintersVector",
                {"trend": "add", "seasonal": "add", "damped_trend": False},
            ),
            ModelConfig.from_dict(
                "HoltWintersVector",
                {"trend": "add", "seasonal": "add", "damped_trend": True},
            ),
            ModelConfig.from_dict(
                "HoltWintersVector",
                {"trend": None, "seasonal": "add", "damped_trend": False},
            ),
            ModelConfig.from_dict(
                "HoltWintersVector",
                {"trend": "add", "seasonal": "mul", "damped_trend": False},
            ),
        ],
        "RidgeMultiOutput": [
            ModelConfig.from_dict("RidgeMultiOutput", {"alpha": a})
            for a in (0.01, 0.1, 1.0, 10.0, 100.0)
        ],
        "WLSRidgeMultiOutput": [
            ModelConfig.from_dict("WLSRidgeMultiOutput", {"alpha": alpha, "vol_window": window})
            for alpha in (0.1, 1.0, 10.0)
            for window in (3, 6, 12)
        ],
        "GradientBoostingVector": [
            ModelConfig.from_dict(
                "GradientBoostingVector",
                {
                    "learning_rate": lr,
                    "max_depth": depth,
                    "max_iter": n_iter,
                    "min_samples_leaf": 20,
                    "l2_regularization": l2,
                },
            )
            for lr, depth, n_iter, l2 in (
                (0.08, 5, 250, 0.0),
                (0.05, 3, 200, 0.0),
                (0.05, 5, 300, 0.0),
                (0.12, 5, 150, 0.0),
                (0.03, 5, 400, 0.0),
                (0.08, 5, 250, 0.1),
            )
        ],
    }
