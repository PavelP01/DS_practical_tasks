"""Expanding-window cross-validation helpers for monthly cluster panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cluster_forecast.config import ModelConfig
from cluster_forecast.metrics import eval_vector_metrics
from cluster_forecast.predict import predict_with_config


def linear_fold_weights(n_folds: int) -> np.ndarray:
    """Linear weights 1..F normalized to sum 1 (recent folds weigh more)."""
    w = np.arange(1, n_folds + 1, dtype=float)
    return w / w.sum()


def expanding_cv_folds(
    Y: pd.DataFrame, holdout: int, min_train: int, n_folds: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding window; last fold ends at the final holdout block."""
    n = len(Y)
    last_start = n - holdout
    if last_start < min_train:
        return []
    if n_folds <= 1:
        starts = np.array([last_start], dtype=int)
    else:
        starts = np.linspace(min_train, last_start, n_folds).astype(int)
        starts = np.unique(starts)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for ts in starts:
        if ts + holdout > n:
            continue
        train = Y.iloc[:ts]
        if len(train) < min_train:
            continue
        test = Y.iloc[ts : ts + holdout]
        if len(test) < holdout:
            continue
        folds.append((train, test))
    return folds


def eval_config_on_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ModelConfig,
    target_columns: list[str],
) -> tuple[float, float]:
    try:
        pred = predict_with_config(train, test.index, config)
        metrics = eval_vector_metrics(test.values, pred, target_columns)
        return float(metrics["mae_mean"]), float(metrics["rmse_mean"])
    except Exception:
        return np.nan, np.nan


def weighted_cv_for_config(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    fold_weights: np.ndarray,
    config: ModelConfig,
    target_columns: list[str],
) -> tuple[float, float, float]:
    """Weighted MAE/RMSE across folds; third value is MAE on the last fold only."""
    fold_maes: list[float] = []
    fold_rmses: list[float] = []
    last_fold_mae = np.nan
    for fold_i, (train, test) in enumerate(folds):
        mae, rmse = eval_config_on_fold(train, test, config, target_columns)
        fold_maes.append(mae)
        fold_rmses.append(rmse)
        if fold_i == len(folds) - 1 and np.isfinite(mae):
            last_fold_mae = mae
    if not fold_maes or not all(np.isfinite(x) for x in fold_maes):
        return np.nan, np.nan, np.nan
    return (
        float(np.dot(fold_weights, fold_maes)),
        float(np.dot(fold_weights, fold_rmses)),
        float(last_fold_mae),
    )
