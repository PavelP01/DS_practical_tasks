"""Vector MAE/RMSE helpers and non-negative clipping for pollutant targets."""

from __future__ import annotations

from typing import Any

import numpy as np


def clip_targets(arr: np.ndarray) -> np.ndarray:
    """Clip forecasts to non-negative values (concentrations and AQI)."""
    return np.maximum(arr, 0.0)


def eval_vector_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, target_columns: list[str]
) -> dict[str, Any]:
    """Mean MAE/RMSE across targets plus per-column breakdown."""
    mae_per_target = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_per_target = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return {
        "mae_mean": float(np.mean(mae_per_target)),
        "rmse_mean": float(np.mean(rmse_per_target)),
        "mae_per_target": {p: float(v) for p, v in zip(target_columns, mae_per_target)},
        "rmse_per_target": {p: float(v) for p, v in zip(target_columns, rmse_per_target)},
    }
