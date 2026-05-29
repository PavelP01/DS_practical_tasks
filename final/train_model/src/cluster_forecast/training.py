"""Train per-cluster models, save joblib artifacts and routing JSON maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from air_quality_config import TARGET_COLUMNS
from cluster_forecast.artifacts import build_cluster_artifact
from cluster_forecast.config import (
    FUTURE_HORIZON_DEFAULT,
    HOLDOUT_DEFAULT,
    PANEL_MODEL_CANDIDATES,
    ModelConfig,
    default_model_config,
    model_config_from_json,
)
from cluster_forecast.intervals import interval_horizon_behavior
from cluster_forecast.metrics import eval_vector_metrics
from cluster_forecast.panel import build_cluster_panel
from cluster_forecast.predict import holdout_vector_predict
from cluster_forecast.logutil import LOGGER
from cluster_forecast.selection import evaluate_holdout_candidates, pick_best_from_scored


def model_config_from_selection(
    model_by_cluster: Any, cid: int, model_type: str
) -> ModelConfig:
    if not isinstance(model_by_cluster, pd.DataFrame) or cid not in model_by_cluster.index:
        return default_model_config(model_type)
    row = model_by_cluster.loc[cid]
    if str(row.get("best_model", "")) != model_type:
        return default_model_config(model_type)
    if "best_params_json" in row.index and pd.notna(row.get("best_params_json")):
        return model_config_from_json(str(row["best_params_json"]), model_type)
    return default_model_config(model_type)


def model_name_from_selection(model_by_cluster: Any, cid: int) -> str | None:
    if model_by_cluster is None:
        return None
    if isinstance(model_by_cluster, pd.DataFrame):
        if "best_model" in model_by_cluster.columns and cid in model_by_cluster.index:
            return str(model_by_cluster.loc[cid, "best_model"])
    return None


def metadata_from_selection(model_by_cluster: Any, cid: int) -> dict[str, Any]:
    if not isinstance(model_by_cluster, pd.DataFrame) or cid not in model_by_cluster.index:
        return {}
    row = model_by_cluster.loc[cid]
    meta: dict[str, Any] = {}
    if "группа_остатков" in row.index and pd.notna(row["группа_остатков"]):
        meta["residual_variance_group"] = str(row["группа_остатков"])
    for src_col in ("ДИ_на_горизонте", "interval_horizon"):
        if src_col in row.index and pd.notna(row[src_col]):
            meta["interval_horizon"] = str(row[src_col])
            break
    return meta


def holdout_metrics_from_selection(model_by_cluster: Any, cid: int) -> dict[str, Any] | None:
    if not isinstance(model_by_cluster, pd.DataFrame) or cid not in model_by_cluster.index:
        return None
    row = model_by_cluster.loc[cid]
    if not np.isfinite(row.get("best_mae_mean", np.nan)):
        return None
    return {
        "mae_mean": float(row["best_mae_mean"]),
        "rmse_mean": float(row.get("best_rmse_mean", np.nan)),
        "mae_per_target": {},
        "rmse_per_target": {},
    }


def train_cluster_models(
    df: pd.DataFrame,
    city_profile: pd.DataFrame,
    models_dir: Path,
    target_columns: list[str] | None = None,
    holdout: int = HOLDOUT_DEFAULT,
    future_horizon: int = FUTURE_HORIZON_DEFAULT,
    model_by_cluster: Any | None = None,
) -> pd.DataFrame:
    """Train and save one multi-output model per cluster (winners from §6.4 when provided)."""
    target_columns = target_columns or [c for c in TARGET_COLUMNS if c in df.columns]
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    cluster_ids = sorted(df["cluster_id"].dropna().astype(int).unique().tolist())
    LOGGER.info("Saving models: clusters=%s → %s", len(cluster_ids), models_dir)

    for cid in cluster_ids:
        LOGGER.info("  cluster %s: building panel...", cid)
        cluster_df = df[df["cluster_id"] == cid].copy()
        Y = build_cluster_panel(cluster_df, target_columns)

        if len(Y) < (holdout + 24):
            all_results.append(
                {
                    "cluster_id": cid,
                    "model": "SKIPPED_SHORT_SERIES",
                    "mae_mean": np.nan,
                    "rmse_mean": np.nan,
                    "n_points": len(Y),
                    "interval_horizon": "—",
                }
            )
            continue

        train = Y.iloc[:-holdout]
        test = Y.iloc[-holdout:]
        preset_name = model_name_from_selection(model_by_cluster, cid)

        if preset_name in ("SKIPPED_SHORT_SERIES", "FAILED_ALL_MODELS"):
            all_results.append(
                {
                    "cluster_id": cid,
                    "model": preset_name,
                    "mae_mean": np.nan,
                    "rmse_mean": np.nan,
                    "n_points": len(Y),
                    "interval_horizon": "—",
                }
            )
            continue

        if preset_name and preset_name in PANEL_MODEL_CANDIDATES:
            best_name = preset_name
            best_config = model_config_from_selection(model_by_cluster, cid, best_name)
            best_metrics = holdout_metrics_from_selection(model_by_cluster, cid)
            if best_metrics is None:
                try:
                    pred = holdout_vector_predict(
                        train,
                        test.index,
                        best_name,
                        target_columns,
                        config=best_config,
                    )
                    best_metrics = eval_vector_metrics(test.values, pred, target_columns)
                except Exception:
                    best_metrics = {
                        "mae_mean": np.nan,
                        "rmse_mean": np.nan,
                        "mae_per_target": {},
                        "rmse_per_target": {},
                    }
        else:
            scored = evaluate_holdout_candidates(train, test, target_columns)
            best_name, best_metrics = pick_best_from_scored(scored)
            best_config = default_model_config(best_name) if best_name else None
            if best_name is None or best_metrics is None:
                all_results.append(
                    {
                        "cluster_id": cid,
                        "model": "FAILED_ALL_MODELS",
                        "mae_mean": np.nan,
                        "rmse_mean": np.nan,
                        "n_points": len(Y),
                        "interval_horizon": "—",
                    }
                )
                continue

        try:
            LOGGER.info("  cluster %s: training %s on full history...", cid, best_name)
            artifact = build_cluster_artifact(
                Y, best_config or default_model_config(best_name), target_columns, future_horizon
            )
        except Exception:
            all_results.append(
                {
                    "cluster_id": cid,
                    "model": f"FAILED_FIT_{best_name}",
                    "mae_mean": np.nan,
                    "rmse_mean": np.nan,
                    "n_points": len(Y),
                    "interval_horizon": "—",
                }
            )
            continue

        artifact.update(
            {
                "cluster_id": int(cid),
                "holdout_mae_mean": best_metrics["mae_mean"],
                "holdout_rmse_mean": best_metrics["rmse_mean"],
                "holdout_mae_per_target": best_metrics.get("mae_per_target", {}),
                "holdout_rmse_per_target": best_metrics.get("rmse_per_target", {}),
                "train_points": int(len(train)),
            }
        )

        out_path = models_dir / f"cluster_{cid}__multioutput.joblib"
        joblib.dump(artifact, out_path)
        LOGGER.info("  cluster %s: saved %s", cid, out_path.name)

        result_row: dict[str, Any] = {
            "cluster_id": cid,
            "model": best_name,
            "mae_mean": best_metrics["mae_mean"],
            "rmse_mean": best_metrics["rmse_mean"],
            "n_points": int(len(Y)),
            "interval_horizon": interval_horizon_behavior(best_name),
        }
        result_row.update(metadata_from_selection(model_by_cluster, cid))
        if "interval_horizon" not in result_row or result_row["interval_horizon"] in ("—", ""):
            result_row["interval_horizon"] = interval_horizon_behavior(best_name)
        all_results.append(result_row)

    results_df = pd.DataFrame(all_results).sort_values("cluster_id")
    results_df.to_csv(models_dir / "cluster_model_selection_results.csv", index=False)
    results_df.to_json(
        models_dir / "cluster_model_selection_results.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )

    city_cluster_map = (
        city_profile[["City", "cluster_id"]]
        .drop_duplicates(subset=["City"])
        .set_index("City")["cluster_id"]
        .astype(int)
        .to_dict()
    )
    with open(models_dir / "city_to_cluster_id.json", "w", encoding="utf-8") as f:
        json.dump(city_cluster_map, f, ensure_ascii=False, indent=2)

    country_city_cluster_map = (
        city_profile.assign(country_city_key=city_profile["Country"] + "|" + city_profile["City"])
        [["country_city_key", "cluster_id"]]
        .drop_duplicates(subset=["country_city_key"])
        .set_index("country_city_key")["cluster_id"]
        .astype(int)
        .to_dict()
    )
    with open(models_dir / "country_city_to_cluster_id.json", "w", encoding="utf-8") as f:
        json.dump(country_city_cluster_map, f, ensure_ascii=False, indent=2)

    with open(models_dir / "aqi_bucket_specs.json", "w", encoding="utf-8") as f:
        from air_quality_config import AQI_BUCKET_SPECS

        json.dump(AQI_BUCKET_SPECS, f, ensure_ascii=False, indent=2)

    return results_df
