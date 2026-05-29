"""Lag/rolling/calendar features and recursive multi-step vector prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from predictor.forecasting.metrics import clip_targets


def make_panel_features(Y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build supervised matrix X aligned with Y (drops rows with NaN lags)."""
    parts = []
    for col in Y.columns:
        s = Y[col]
        feat = pd.DataFrame(index=Y.index)
        feat[f"{col}__lag_1"] = s.shift(1)
        feat[f"{col}__lag_2"] = s.shift(2)
        feat[f"{col}__lag_3"] = s.shift(3)
        feat[f"{col}__lag_6"] = s.shift(6)
        feat[f"{col}__lag_12"] = s.shift(12)
        feat[f"{col}__roll_mean_3"] = s.shift(1).rolling(3).mean()
        feat[f"{col}__roll_mean_6"] = s.shift(1).rolling(6).mean()
        parts.append(feat)

    cal = pd.DataFrame(index=Y.index)
    cal["month_sin"] = np.sin(2 * np.pi * Y.index.month / 12)
    cal["month_cos"] = np.cos(2 * np.pi * Y.index.month / 12)
    cal["time_idx"] = np.arange(len(Y))
    parts.append(cal)

    X = pd.concat(parts, axis=1)
    valid_idx = X.dropna().index
    return X.loc[valid_idx], Y.loc[valid_idx]


def feature_row_from_panel_history(history: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:
    row: dict[str, float] = {}
    for col in history.columns:
        vals = history[col]
        row[f"{col}__lag_1"] = float(vals.iloc[-1])
        row[f"{col}__lag_2"] = float(vals.iloc[-2])
        row[f"{col}__lag_3"] = float(vals.iloc[-3])
        row[f"{col}__lag_6"] = float(vals.iloc[-6])
        row[f"{col}__lag_12"] = float(vals.iloc[-12])
        row[f"{col}__roll_mean_3"] = float(vals.iloc[-3:].mean())
        row[f"{col}__roll_mean_6"] = float(vals.iloc[-6:].mean())
    row["month_sin"] = float(np.sin(2 * np.pi * current_date.month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * current_date.month / 12))
    row["time_idx"] = float(len(history))
    return pd.DataFrame([row], index=[current_date])


def recursive_vector_predict(
    model: Pipeline, history: pd.DataFrame, future_index: pd.DatetimeIndex
) -> np.ndarray:
    """Multi-step forecast feeding predicted values back into lag features."""
    hist = history.copy()
    preds = []
    for dt in future_index:
        x_next = feature_row_from_panel_history(hist, dt)
        y_hat = model.predict(x_next)[0]
        preds.append(y_hat)
        hist.loc[dt] = y_hat
    return clip_targets(np.vstack(preds))
