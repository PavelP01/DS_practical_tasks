"""Hyperparameter tuning and weighted CV model selection per cluster."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from air_quality_config import AQI_COLUMN
from cluster_forecast.config import (
    HOLDOUT_DEFAULT,
    CV_MIN_TRAIN_MONTHS,
    CV_N_FOLDS,
    PANEL_MODEL_CANDIDATES,
    ModelConfig,
    default_model_config,
    hyperparam_grids,
)
from cluster_forecast.cv import (
    expanding_cv_folds,
    linear_fold_weights,
    weighted_cv_for_config,
)
from cluster_forecast.intervals import interval_horizon_behavior
from cluster_forecast.metrics import eval_vector_metrics
from cluster_forecast.panel import build_cluster_panel
from cluster_forecast.logutil import LOGGER
from cluster_forecast.predict import holdout_vector_predict, predict_intervals_with_config


def config_for_cluster_type(
    tuned_by_cluster: dict[int, dict[str, ModelConfig]] | None,
    cluster_id: int,
    model_type: str,
) -> ModelConfig:
    if tuned_by_cluster and cluster_id in tuned_by_cluster:
        return tuned_by_cluster[cluster_id].get(model_type, default_model_config(model_type))
    return default_model_config(model_type)


def tune_hyperparams_weighted_cv(
    df: pd.DataFrame,
    target_columns: list[str],
    *,
    holdout: int = HOLDOUT_DEFAULT,
    cv_min_train_months: int = CV_MIN_TRAIN_MONTHS,
    cv_n_folds: int = CV_N_FOLDS,
    model_candidates: list[str] | None = None,
    grids: dict[str, list[ModelConfig]] | None = None,
) -> tuple[pd.DataFrame, dict[int, dict[str, ModelConfig]]]:
    """Per-cluster hyperparameter tuning for each model type (§6.4.1)."""
    model_candidates = list(model_candidates or PANEL_MODEL_CANDIDATES)
    grids = grids or hyperparam_grids()
    tuning_rows: list[dict[str, Any]] = []
    tuned_by_cluster: dict[int, dict[str, ModelConfig]] = {}
    cluster_ids = sorted(df["cluster_id"].dropna().astype(int).unique())
    n_grid = sum(len(grids.get(m, [default_model_config(m)])) for m in model_candidates)
    LOGGER.info(
        "6.4.1: clusters=%s, model types=%s, configs≈%s per cluster",
        len(cluster_ids),
        len(model_candidates),
        n_grid,
    )

    for cid in cluster_ids:
        Y = build_cluster_panel(df[df["cluster_id"] == cid], target_columns)
        tuned_by_cluster[cid] = {}
        min_required = cv_min_train_months + holdout
        folds = (
            expanding_cv_folds(Y, holdout, cv_min_train_months, cv_n_folds)
            if len(Y) >= min_required
            else []
        )
        fold_weights = linear_fold_weights(len(folds)) if folds else np.array([])
        LOGGER.info(
            "  cluster %s: months=%s, CV folds=%s",
            cid,
            len(Y),
            len(folds),
        )

        for model_type in model_candidates:
            candidates = grids.get(model_type, [default_model_config(model_type)])
            best_cfg: ModelConfig | None = None
            best_key = (np.inf, np.inf, np.inf)
            best_scores = (np.nan, np.nan)

            if folds:
                LOGGER.info(
                    "    %s: search over %s configs (CV)",
                    model_type,
                    len(candidates),
                )
                for cfg in candidates:
                    w_mae, w_rmse, last_mae = weighted_cv_for_config(
                        folds, fold_weights, cfg, target_columns
                    )
                    if not np.isfinite(w_mae):
                        continue
                    key = (
                        w_mae,
                        last_mae if np.isfinite(last_mae) else np.inf,
                        w_rmse,
                    )
                    if key < best_key:
                        best_key = key
                        best_cfg = cfg
                        best_scores = (w_mae, w_rmse)

            if best_cfg is None:
                best_cfg = default_model_config(model_type)
            else:
                tuned_by_cluster[cid][model_type] = best_cfg

            tuning_rows.append(
                {
                    "cluster_id": cid,
                    "model_type": model_type,
                    "best_hyperparams": best_cfg.label(),
                    "best_params_json": best_cfg.to_json(),
                    "cv_wmae": best_scores[0],
                    "cv_wrmse": best_scores[1],
                    "cv_n_folds": len(folds),
                }
            )
            if folds and best_cfg is not None:
                LOGGER.info(
                    "    %s: best hp=%s, cv_wmae=%.3f",
                    model_type,
                    best_cfg.label(),
                    best_scores[0],
                )

    tuning_df = pd.DataFrame(tuning_rows).set_index(["cluster_id", "model_type"])
    return tuning_df, tuned_by_cluster


def holdout_aqi_band_halfwidth(
    train: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    model_name: str,
    target_columns: list[str],
    config: ModelConfig | None = None,
) -> float:
    cfg = config or default_model_config(model_name)
    aqi_j = target_columns.index(AQI_COLUMN)
    try:
        _, lo, hi = predict_intervals_with_config(train, test_index, cfg, target_columns)
    except Exception:
        return np.nan
    return float((hi[-1, aqi_j] - lo[-1, aqi_j]) / 2)


def select_models_weighted_cv(
    df: pd.DataFrame,
    target_columns: list[str],
    *,
    residual_variance_group: dict[int, tuple[str, str]] | None = None,
    holdout: int = HOLDOUT_DEFAULT,
    cv_min_train_months: int = CV_MIN_TRAIN_MONTHS,
    cv_n_folds: int = CV_N_FOLDS,
    model_candidates: list[str] | None = None,
    tuned_by_cluster: dict[int, dict[str, ModelConfig]] | None = None,
    include_per_type_scores: bool = False,
) -> pd.DataFrame:
    """Weighted expanding-window CV model selection (experiments.ipynb §6.4.2)."""
    model_candidates = list(model_candidates or PANEL_MODEL_CANDIDATES)
    residual_variance_group = residual_variance_group or {}
    eval_rows: list[dict[str, Any]] = []
    cluster_ids = sorted(df["cluster_id"].dropna().astype(int).unique())
    LOGGER.info("6.4.2: model-type selection, clusters=%s", len(cluster_ids))

    for cid in cluster_ids:
        grp_code, grp_label = residual_variance_group.get(cid, ("?", "неизвестно"))
        row: dict[str, Any] = {
            "cluster_id": cid,
            "группа_остатков": grp_code,
            "остатки_дисперсия": grp_label,
        }

        Y = build_cluster_panel(df[df["cluster_id"] == cid], target_columns)
        row["n_months"] = len(Y)

        min_required = cv_min_train_months + holdout
        if len(Y) < min_required:
            row.update(
                {
                    "best_model": "SKIPPED_SHORT_SERIES",
                    "best_hyperparams": "",
                    "best_params_json": "",
                    "best_mae_mean": np.nan,
                    "best_rmse_mean": np.nan,
                    "mae_last_holdout": np.nan,
                    "cv_n_folds": 0,
                    "cv_weights": "",
                    "ДИ_на_горизонте": "—",
                    "aqi_±_h12": np.nan,
                }
            )
            eval_rows.append(row)
            continue

        folds = expanding_cv_folds(Y, holdout, cv_min_train_months, cv_n_folds)
        if not folds:
            row.update(
                {
                    "best_model": "FAILED_NO_CV_FOLDS",
                    "best_hyperparams": "",
                    "best_params_json": "",
                    "best_mae_mean": np.nan,
                    "best_rmse_mean": np.nan,
                    "mae_last_holdout": np.nan,
                    "cv_n_folds": 0,
                    "cv_weights": "",
                    "ДИ_на_горизонте": "—",
                    "aqi_±_h12": np.nan,
                }
            )
            eval_rows.append(row)
            continue

        fold_weights = linear_fold_weights(len(folds))
        row["cv_n_folds"] = len(folds)
        row["cv_weights"] = ", ".join(f"{w:.2f}" for w in fold_weights)

        scored: list[tuple[str, ModelConfig, float, float, float]] = []

        for model_name in model_candidates:
            config = config_for_cluster_type(tuned_by_cluster, cid, model_name)
            w_mae, w_rmse, last_mae = weighted_cv_for_config(
                folds, fold_weights, config, target_columns
            )
            if not np.isfinite(w_mae):
                if include_per_type_scores:
                    row[f"MAE_{model_name}"] = np.nan
                    row[f"RMSE_{model_name}"] = np.nan
                continue
            if include_per_type_scores:
                row[f"MAE_{model_name}"] = w_mae
                row[f"RMSE_{model_name}"] = w_rmse
            scored.append((model_name, config, w_mae, w_rmse, last_mae))

        if not scored:
            row["best_model"] = "FAILED_ALL_MODELS"
            row["best_hyperparams"] = ""
            row["best_params_json"] = ""
            row["best_mae_mean"] = row["best_rmse_mean"] = np.nan
            row["mae_last_holdout"] = np.nan
        else:
            best_name, best_cfg, best_mae, best_rmse, last_mae = sorted(
                scored,
                key=lambda x: (x[2], x[4] if np.isfinite(x[4]) else np.inf, x[3]),
            )[0]
            row["best_model"] = best_name
            row["best_hyperparams"] = best_cfg.label()
            row["best_params_json"] = best_cfg.to_json()
            row["best_mae_mean"] = best_mae
            row["best_rmse_mean"] = best_rmse
            row["mae_last_holdout"] = last_mae
            LOGGER.info(
                "  cluster %s: winner=%s, cv_wmae=%.3f, group=%s",
                cid,
                best_name,
                best_mae,
                grp_code,
            )

        best_m = row.get("best_model")
        row["ДИ_на_горизонте"] = (
            interval_horizon_behavior(best_m) if best_m in model_candidates else "—"
        )
        if best_m in model_candidates:
            best_cfg = next((cfg for name, cfg, *_ in scored if name == best_m), None)
            if best_cfg is None:
                best_cfg = config_for_cluster_type(tuned_by_cluster, cid, best_m)
            last_train, last_test = folds[-1]
            try:
                row["aqi_±_h12"] = holdout_aqi_band_halfwidth(
                    last_train,
                    last_test.index,
                    best_m,
                    target_columns,
                    config=best_cfg,
                )
            except Exception:
                row["aqi_±_h12"] = np.nan
        else:
            row["aqi_±_h12"] = np.nan

        eval_rows.append(row)

    return pd.DataFrame(eval_rows).set_index("cluster_id")


def evaluate_holdout_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_columns: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Score all model types on a single hold-out block (fallback when no preset winner)."""
    test_index = test.index
    scored: list[tuple[str, dict[str, Any]]] = []

    try:
        from cluster_forecast.models.seasonal_naive import seasonal_naive_vector_forecast

        sn_pred = seasonal_naive_vector_forecast(train, test_index)
        scored.append(
            ("SeasonalNaiveVector", eval_vector_metrics(test.values, sn_pred, target_columns))
        )
    except Exception:
        pass

    try:
        from cluster_forecast.models.holt_winters import holt_winters_vector_forecast

        hw_pred, _, _ = holt_winters_vector_forecast(train, len(test))
        scored.append(
            ("HoltWintersVector", eval_vector_metrics(test.values, hw_pred, target_columns))
        )
    except Exception:
        pass

    try:
        from cluster_forecast.features import make_panel_features

        X_train, y_train = make_panel_features(train)
        if len(X_train) >= 24:
            for name in ("RidgeMultiOutput", "WLSRidgeMultiOutput", "GradientBoostingVector"):
                try:
                    pred = holdout_vector_predict(train, test_index, name, target_columns)
                    scored.append(
                        (name, eval_vector_metrics(test.values, pred, target_columns))
                    )
                except Exception:
                    pass
    except Exception:
        pass

    return scored


def pick_best_from_scored(
    scored: list[tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    valid = [(n, m) for n, m in scored if np.isfinite(m.get("mae_mean", np.nan))]
    if not valid:
        return None, None
    best_name, best_metrics = sorted(valid, key=lambda x: (x[1]["mae_mean"], x[1]["rmse_mean"]))[0]
    return best_name, best_metrics
