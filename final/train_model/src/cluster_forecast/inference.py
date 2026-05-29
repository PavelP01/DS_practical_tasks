"""Load saved cluster artifacts and build API-ready forecasts (inference only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from air_quality_config import AQI_COLUMN, aqi_to_bucket
from cluster_forecast.config import FUTURE_HORIZON_DEFAULT
from cluster_forecast.features import recursive_vector_predict
from cluster_forecast.intervals import ridge_vector_intervals, seasonal_naive_vector_intervals
from cluster_forecast.models.holt_winters import (
    holt_winters_forecast_from_submodels,
    holt_winters_vector_forecast,
)
from cluster_forecast.models.seasonal_naive import seasonal_naive_vector_forecast


def format_future_months(
    future_index: pd.DatetimeIndex,
    mean: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    target_columns: list[str],
) -> list[dict[str, Any]]:
    """Build JSON-friendly month list with pollutants, AQI, and min/max bands."""
    months: list[dict[str, Any]] = []
    aqi_idx = target_columns.index(AQI_COLUMN)

    for i, dt in enumerate(future_index):
        pollutants: dict[str, Any] = {}
        for j, col in enumerate(target_columns):
            if col == AQI_COLUMN:
                continue
            pollutants[col] = {
                "value": float(mean[i, j]),
                "min": float(lo[i, j]),
                "max": float(hi[i, j]),
            }

        aqi_value = float(mean[i, aqi_idx])
        months.append(
            {
                "date": str(dt.date()),
                "pollutants": pollutants,
                "aqi": aqi_value,
                "aqi_min": float(lo[i, aqi_idx]),
                "aqi_max": float(hi[i, aqi_idx]),
                "aqi_bucket": aqi_to_bucket(aqi_value),
            }
        )
    return months


def history_from_artifact(
    artifact: dict[str, Any], year: int = 2025
) -> list[dict[str, Any]]:
    """Factual cluster panel for ``year`` (no min/max bands)."""
    Y = artifact["training_panel"]
    if not isinstance(Y, pd.DataFrame):
        Y = pd.DataFrame(Y)
        Y.index = pd.to_datetime(Y.index)

    target_columns: list[str] = list(artifact["target_columns"])
    Y = Y.loc[Y.index.year == year].sort_index()
    months: list[dict[str, Any]] = []

    for dt, row in Y.iterrows():
        pollutants: dict[str, float] = {}
        for col in target_columns:
            if col == AQI_COLUMN:
                continue
            pollutants[col] = float(row[col])

        aqi_value = float(row[AQI_COLUMN])
        months.append(
            {
                "date": str(pd.Timestamp(dt).date()),
                "pollutants": pollutants,
                "aqi": aqi_value,
                "aqi_bucket": aqi_to_bucket(aqi_value),
            }
        )
    return months


def forecast_from_artifact(
    artifact: dict[str, Any], horizon_months: int | None = None
) -> list[dict[str, Any]]:
    """Build API-ready month list from a saved cluster ``.joblib`` artifact."""
    Y = artifact["training_panel"]
    if not isinstance(Y, pd.DataFrame):
        Y = pd.DataFrame(Y)
        Y.index = pd.to_datetime(Y.index)

    steps = int(horizon_months or artifact.get("future_horizon", FUTURE_HORIZON_DEFAULT))
    future_index = pd.date_range(
        Y.index.max() + pd.offsets.MonthBegin(1), periods=steps, freq="MS"
    )

    model_type = artifact["model_type"]
    if model_type == "HoltWintersVector":
        if artifact.get("submodels"):
            mean, lo, hi = holt_winters_forecast_from_submodels(
                artifact["submodels"],
                steps,
                artifact["target_columns"],
                history=Y,
            )
        else:
            mean, lo, hi = holt_winters_vector_forecast(Y, steps)
    elif model_type in (
        "RidgeMultiOutput",
        "WLSRidgeMultiOutput",
        "GradientBoostingVector",
    ):
        model = artifact["model"]
        mean = recursive_vector_predict(model, Y.copy(), future_index)
        lo, hi = ridge_vector_intervals(mean, np.array(artifact["residual_std"]), steps)
    else:
        mean = seasonal_naive_vector_forecast(Y, future_index)
        lo, hi = seasonal_naive_vector_intervals(mean, np.array(artifact["residual_std"]))

    return format_future_months(
        future_index, mean, lo, hi, artifact["target_columns"]
    )
