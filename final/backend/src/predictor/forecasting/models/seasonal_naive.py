"""Seasonal naive vector forecast: value from the same calendar month one year ago."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predictor.forecasting.metrics import clip_targets


def estimate_seasonal_naive_residual_std(
    history: pd.DataFrame, season_lag: int = 12
) -> np.ndarray:
    """Per-target residual std from in-sample seasonal naive errors."""
    residuals: list[np.ndarray] = []
    for i in range(season_lag, len(history)):
        ref = history.index[i] - pd.DateOffset(months=season_lag)
        if ref in history.index:
            residuals.append(history.iloc[i].values - history.loc[ref].values)
    if not residuals:
        return np.full(history.shape[1], np.nan)
    return np.nanstd(np.vstack(residuals), axis=0, ddof=1)


def seasonal_naive_vector_forecast(
    history: pd.DataFrame,
    future_index: pd.DatetimeIndex,
    *,
    season_lag: int = 12,
    fallback: str = "last",
) -> np.ndarray:
    """Forecast each target using lag ``season_lag`` with optional fallback."""
    preds = []
    for dt in future_index:
        ref = dt - pd.DateOffset(months=season_lag)
        if ref in history.index:
            preds.append(history.loc[ref].values)
        elif fallback == "mean12" and len(history) >= season_lag:
            preds.append(history.iloc[-season_lag:].mean().values)
        else:
            preds.append(history.iloc[-1].values)
    return clip_targets(np.vstack(preds))
