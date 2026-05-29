"""Train-time artifact building and notebook forecast helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cluster_forecast.config import ModelConfig, default_model_config
from cluster_forecast.inference import format_future_months, forecast_from_artifact
from cluster_forecast.features import make_panel_features
from cluster_forecast.intervals import interval_horizon_behavior
from cluster_forecast.models.holt_winters import fit_holt_winters_submodels
from cluster_forecast.models.lag import ridge_residual_std
from cluster_forecast.models.seasonal_naive import estimate_seasonal_naive_residual_std
from cluster_forecast.predict import fit_lag_model, predict_intervals_with_config


def forecast_panel_with_intervals(
    Y: pd.DataFrame,
    model_type: str,
    horizon: int,
    target_columns: list[str] | None = None,
    config: ModelConfig | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Fit on full panel Y and forecast ``horizon`` months with mean/lo/hi."""
    target_columns = target_columns or list(Y.columns)
    cfg = config or default_model_config(model_type)
    future_index = pd.date_range(
        Y.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    mean, lo, hi = predict_intervals_with_config(Y, future_index, cfg, target_columns)
    return future_index, mean, lo, hi


def build_cluster_artifact(
    Y: pd.DataFrame,
    model_type: str | ModelConfig,
    target_columns: list[str],
    future_horizon: int,
) -> dict[str, Any]:
    """Fit on full panel and return a joblib-serializable artifact dict."""
    config = (
        model_type if isinstance(model_type, ModelConfig) else default_model_config(model_type)
    )
    model_type_name = config.model_type
    params = config.as_dict()
    future_index = pd.date_range(
        Y.index.max() + pd.offsets.MonthBegin(1), periods=future_horizon, freq="MS"
    )

    mean, lo, hi = predict_intervals_with_config(Y, future_index, config, target_columns)

    if model_type_name == "SeasonalNaiveVector":
        season_lag = int(params.get("season_lag", 12))
        sn_std = estimate_seasonal_naive_residual_std(Y, season_lag=season_lag)
        artifact: dict[str, Any] = {
            "model_type": model_type_name,
            "residual_std": sn_std.tolist(),
            "hyperparams": params,
        }
    elif model_type_name == "HoltWintersVector":
        submodels = fit_holt_winters_submodels(
            Y,
            target_columns,
            trend=params.get("trend", "add"),
            seasonal=str(params.get("seasonal", "add")),
            damped_trend=bool(params.get("damped_trend", False)),
        )
        artifact = {
            "model_type": model_type_name,
            "submodels": submodels,
            "hyperparams": params,
        }
    elif model_type_name in ("RidgeMultiOutput", "GradientBoostingVector", "WLSRidgeMultiOutput"):
        X_full, y_full = make_panel_features(Y)
        if len(X_full) < 24:
            raise ValueError("Not enough history for lag-based models (need >= 24 valid rows).")
        model = fit_lag_model(config, X_full, y_full)
        r_std = ridge_residual_std(model, X_full, y_full)
        artifact = {
            "model_type": model_type_name,
            "model": model,
            "residual_std": r_std.tolist(),
            "feature_builder": "panel_lag_rolling_calendar_v1",
            "hyperparams": params,
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type_name}")

    future_months = format_future_months(future_index, mean, lo, hi, target_columns)
    artifact.update(
        {
            "target_columns": target_columns,
            "n_targets": len(target_columns),
            "full_points": int(len(Y)),
            "future_horizon": int(future_horizon),
            "training_panel": Y,
            "future_months": future_months,
            "selected_model": model_type_name,
            "interval_horizon": interval_horizon_behavior(model_type_name),
            "model_config_json": config.to_json(),
        }
    )
    return artifact


__all__ = [
    "build_cluster_artifact",
    "forecast_from_artifact",
    "forecast_panel_with_intervals",
    "format_future_months",
]
