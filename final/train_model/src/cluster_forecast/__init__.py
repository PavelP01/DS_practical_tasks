"""Per-cluster multi-target air-quality forecasting (package facade).

Subpackages: clustering, panel, models, CV/selection, artifacts, training.
"""

from cluster_forecast.artifacts import (
    build_cluster_artifact,
    forecast_from_artifact,
    forecast_panel_with_intervals,
    format_future_months,
)
from cluster_forecast.clustering import (
    assign_city_clusters,
    build_city_profile,
    compute_residual_variance_groups,
)
from cluster_forecast.config import (
    CITY_GROUP_COLS,
    CLUSTER_DRIVERS,
    CV_MIN_TRAIN_MONTHS,
    CV_N_FOLDS,
    DATE_COL,
    FUTURE_HORIZON_DEFAULT,
    HOLDOUT_DEFAULT,
    KERNEL_PCA_GAMMA,
    KERNEL_PCA_KERNEL,
    N_CLUSTERS_DEFAULT,
    N_KERNEL_PCA_DEFAULT,
    PANEL_MODEL_CANDIDATES,
    ModelConfig,
    default_model_config,
    hyperparam_grids,
    model_config_from_json,
)
from cluster_forecast.interpret import (
    interpret_cluster_model_selection,
    interpret_model_selection_table,
)
from cluster_forecast.intervals import interval_horizon_behavior
from cluster_forecast.metrics import eval_vector_metrics
from cluster_forecast.panel import build_cluster_panel
from cluster_forecast.selection import (
    pick_best_from_scored,
    select_models_weighted_cv,
    tune_hyperparams_weighted_cv,
)
from cluster_forecast.training import train_cluster_models

__all__ = [
    "CITY_GROUP_COLS",
    "CLUSTER_DRIVERS",
    "CV_MIN_TRAIN_MONTHS",
    "CV_N_FOLDS",
    "DATE_COL",
    "FUTURE_HORIZON_DEFAULT",
    "HOLDOUT_DEFAULT",
    "KERNEL_PCA_GAMMA",
    "KERNEL_PCA_KERNEL",
    "ModelConfig",
    "N_CLUSTERS_DEFAULT",
    "N_KERNEL_PCA_DEFAULT",
    "PANEL_MODEL_CANDIDATES",
    "assign_city_clusters",
    "build_city_profile",
    "build_cluster_artifact",
    "build_cluster_panel",
    "compute_residual_variance_groups",
    "default_model_config",
    "eval_vector_metrics",
    "forecast_from_artifact",
    "forecast_panel_with_intervals",
    "format_future_months",
    "hyperparam_grids",
    "interpret_cluster_model_selection",
    "interpret_model_selection_table",
    "interval_horizon_behavior",
    "model_config_from_json",
    "pick_best_from_scored",
    "select_models_weighted_cv",
    "train_cluster_models",
    "tune_hyperparams_weighted_cv",
]
