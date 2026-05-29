"""Sklearn pipelines for multi-output lag models (Ridge, WLS-Ridge, boosting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_pipeline(alpha: float = 1.0, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=random_state)),
        ]
    )


def build_boosting_pipeline(
    random_state: int = 42,
    *,
    max_depth: int = 5,
    max_iter: int = 250,
    learning_rate: float = 0.08,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_depth=max_depth,
                        max_iter=max_iter,
                        learning_rate=learning_rate,
                        min_samples_leaf=min_samples_leaf,
                        l2_regularization=l2_regularization,
                        random_state=random_state,
                    )
                ),
            ),
        ]
    )


def panel_row_weights(y_train: pd.DataFrame, vol_window: int = 6) -> np.ndarray:
    """Inverse-variance weights for WLS: quieter months weigh more."""
    vol = y_train.std(axis=1).rolling(vol_window, min_periods=3).mean()
    vol = vol.fillna(float(vol.mean())).clip(lower=1e-6)
    w = 1.0 / np.square(vol.values)
    return w / np.mean(w)


def fit_wls_ridge_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    alpha: float = 1.0,
    random_state: int = 42,
    vol_window: int = 6,
) -> Pipeline:
    pipe = build_ridge_pipeline(alpha=alpha, random_state=random_state)
    pipe.fit(X_train, y_train, model__sample_weight=panel_row_weights(y_train, vol_window))
    return pipe


def ridge_residual_std(model: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame) -> np.ndarray:
    y_hat = model.predict(X_train)
    return np.std(y_train.values - y_hat, axis=0, ddof=1)
