"""Holt–Winters exponential smoothing per target (statsmodels, period=12)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from predictor.forecasting.intervals import holt_winters_mean_intervals
from predictor.forecasting.metrics import clip_targets


def _fit_holt_winters_series(
    series: pd.Series,
    *,
    trend: str | None = "add",
    seasonal: str = "add",
    damped_trend: bool = False,
) -> ExponentialSmoothing:
    kwargs: dict[str, Any] = {
        "seasonal_periods": 12,
        "initialization_method": "estimated",
    }
    if trend is not None:
        kwargs["trend"] = trend
        if damped_trend:
            kwargs["damped_trend"] = True
    if seasonal is not None:
        kwargs["seasonal"] = seasonal
    return ExponentialSmoothing(series, **kwargs).fit(optimized=True)


def fit_holt_winters_submodels(
    Y: pd.DataFrame,
    target_columns: list[str],
    *,
    trend: str | None = "add",
    seasonal: str = "add",
    damped_trend: bool = False,
) -> dict[str, Any]:
    """Fit one Holt–Winters model per column for artifact persistence."""
    submodels: dict[str, Any] = {}
    for col in target_columns:
        submodels[col] = _fit_holt_winters_series(
            Y[col],
            trend=trend,
            seasonal=seasonal,
            damped_trend=damped_trend,
        )
    return submodels


def holt_winters_forecast_from_submodels(
    submodels: dict[str, Any],
    steps: int,
    target_columns: list[str] | None = None,
    history: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast from pre-fitted submodels (inference / saved artifacts)."""
    mean_cols, lo_cols, hi_cols = [], [], []
    for col in target_columns or list(submodels.keys()):
        fit = submodels[col]
        series = history[col] if history is not None else fit.data
        mean, lo, hi = holt_winters_mean_intervals(fit, series, steps)
        mean_cols.append(mean)
        lo_cols.append(lo)
        hi_cols.append(hi)
    mean = clip_targets(np.column_stack(mean_cols))
    lo = clip_targets(np.column_stack(lo_cols))
    hi = clip_targets(np.column_stack(hi_cols))
    return mean, lo, hi


def holt_winters_vector_forecast(
    history: pd.DataFrame,
    steps: int,
    *,
    trend: str | None = "add",
    seasonal: str = "add",
    damped_trend: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit submodels on ``history`` and forecast ``steps`` months ahead."""
    submodels = fit_holt_winters_submodels(
        history,
        list(history.columns),
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped_trend,
    )
    return holt_winters_forecast_from_submodels(
        submodels, steps, list(history.columns), history=history
    )
