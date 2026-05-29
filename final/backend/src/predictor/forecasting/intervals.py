"""Min/max forecast bands and horizon scaling rules (const vs sqrt(h))."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from predictor.air_quality_config import INTERVAL_Z
from predictor.forecasting.metrics import clip_targets


def interval_horizon_behavior(model_type: str) -> str:
    """How band half-width scales with forecast step h (months from history end)."""
    if model_type == "SeasonalNaiveVector":
        return "const"
    if model_type in (
        "HoltWintersVector",
        "RidgeMultiOutput",
        "WLSRidgeMultiOutput",
        "GradientBoostingVector",
    ):
        return "sqrt_h"
    return "—"


def seasonal_naive_vector_intervals(
    mean: np.ndarray, residual_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    std = np.where(np.isfinite(residual_std), residual_std, 0.0)
    lo = clip_targets(mean - INTERVAL_Z * std)
    hi = clip_targets(mean + INTERVAL_Z * std)
    return lo, hi


def holt_winters_residual_std(fit: Any, series: pd.Series) -> float:
    fitted = fit.fittedvalues
    resid = series.reindex(fitted.index) - fitted
    if len(resid) < 2:
        return 0.0
    return float(resid.std(ddof=1))


def holt_winters_mean_intervals(
    fit: Any, series: pd.Series, steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point forecast via Holt–Winters; intervals from in-sample residual σ."""
    mean = np.asarray(fit.forecast(steps), dtype=float)
    std = holt_winters_residual_std(fit, series)
    scale = np.sqrt(np.arange(1, steps + 1))
    lo = mean - INTERVAL_Z * std * scale
    hi = mean + INTERVAL_Z * std * scale
    return mean, lo, hi


def ridge_vector_intervals(
    mean: np.ndarray, residual_std: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.arange(1, steps + 1))[:, None]
    std = np.where(np.isfinite(residual_std), residual_std, 0.0) * scale
    lo = clip_targets(mean - INTERVAL_Z * std)
    hi = clip_targets(mean + INTERVAL_Z * std)
    return lo, hi
