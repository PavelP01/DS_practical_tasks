"""Unified point and interval prediction for all model types and configs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cluster_forecast.config import ModelConfig, default_model_config
from cluster_forecast.features import make_panel_features, recursive_vector_predict
from cluster_forecast.intervals import (
    ridge_vector_intervals,
    seasonal_naive_vector_intervals,
)
from cluster_forecast.models.holt_winters import holt_winters_vector_forecast
from cluster_forecast.models.lag import (
    build_boosting_pipeline,
    build_ridge_pipeline,
    fit_wls_ridge_pipeline,
    ridge_residual_std,
)
from cluster_forecast.models.seasonal_naive import (
    estimate_seasonal_naive_residual_std,
    seasonal_naive_vector_forecast,
)


def fit_lag_model(config: ModelConfig, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    """Fit Ridge, WLS-Ridge, or gradient boosting pipeline from ``config``."""
    params = config.as_dict()
    if config.model_type == "RidgeMultiOutput":
        model = build_ridge_pipeline(alpha=float(params.get("alpha", 1.0)))
        model.fit(X_train, y_train)
        return model
    if config.model_type == "GradientBoostingVector":
        model = build_boosting_pipeline(
            max_depth=int(params.get("max_depth", 5)),
            max_iter=int(params.get("max_iter", 250)),
            learning_rate=float(params.get("learning_rate", 0.08)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            l2_regularization=float(params.get("l2_regularization", 0.0)),
        )
        model.fit(X_train, y_train)
        return model
    if config.model_type == "WLSRidgeMultiOutput":
        return fit_wls_ridge_pipeline(
            X_train,
            y_train,
            alpha=float(params.get("alpha", 1.0)),
            vol_window=int(params.get("vol_window", 6)),
        )
    raise ValueError(f"Not a lag-based model: {config.model_type}")


def predict_with_config(
    train: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    config: ModelConfig,
) -> np.ndarray:
    """Point forecasts only (no intervals)."""
    params = config.as_dict()
    steps = len(test_index)
    if config.model_type == "SeasonalNaiveVector":
        return seasonal_naive_vector_forecast(
            train,
            test_index,
            season_lag=int(params.get("season_lag", 12)),
            fallback=str(params.get("fallback", "last")),
        )
    if config.model_type == "HoltWintersVector":
        mean, _, _ = holt_winters_vector_forecast(
            train,
            steps,
            trend=params.get("trend", "add"),
            seasonal=str(params.get("seasonal", "add")),
            damped_trend=bool(params.get("damped_trend", False)),
        )
        return mean
    X_train, y_train = make_panel_features(train)
    if len(X_train) < 24:
        raise ValueError("Not enough history for lag-based models")
    model = fit_lag_model(config, X_train, y_train)
    return recursive_vector_predict(model, train.copy(), test_index)


def predict_intervals_with_config(
    train: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    config: ModelConfig,
    target_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point forecasts with min/max bands for all targets."""
    del target_columns  # column order follows train.columns
    params = config.as_dict()
    steps = len(test_index)
    if config.model_type == "SeasonalNaiveVector":
        season_lag = int(params.get("season_lag", 12))
        mean = seasonal_naive_vector_forecast(
            train,
            test_index,
            season_lag=season_lag,
            fallback=str(params.get("fallback", "last")),
        )
        sn_std = estimate_seasonal_naive_residual_std(train, season_lag=season_lag)
        lo, hi = seasonal_naive_vector_intervals(mean, sn_std)
        return mean, lo, hi
    if config.model_type == "HoltWintersVector":
        return holt_winters_vector_forecast(
            train,
            steps,
            trend=params.get("trend", "add"),
            seasonal=str(params.get("seasonal", "add")),
            damped_trend=bool(params.get("damped_trend", False)),
        )
    X_train, y_train = make_panel_features(train)
    model = fit_lag_model(config, X_train, y_train)
    mean = recursive_vector_predict(model, train.copy(), test_index)
    r_std = ridge_residual_std(model, X_train, y_train)
    lo, hi = ridge_vector_intervals(mean, r_std, steps)
    return mean, lo, hi


def holdout_vector_predict(
    train: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    model_type: str,
    target_columns: list[str],
    config: ModelConfig | None = None,
) -> np.ndarray:
    """Hold-out point forecast for a named model type (optional tuned config)."""
    del target_columns
    cfg = config or default_model_config(model_type)
    return predict_with_config(train, test_index, cfg)
