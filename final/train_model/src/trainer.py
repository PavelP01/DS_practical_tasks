"""Train per-cluster air-quality forecast models (separate from the EDA notebook).

Writes artifacts to train_model/models/ and backend/models/ (existing files overwritten).

Run from train_model/:
    poetry run python src/trainer.py
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from air_quality_config import TARGET_COLUMNS
from cluster_forecast import (
    CV_MIN_TRAIN_MONTHS,
    CV_N_FOLDS,
    DATE_COL,
    HOLDOUT_DEFAULT,
    N_CLUSTERS_DEFAULT,
    assign_city_clusters,
    compute_residual_variance_groups,
    select_models_weighted_cv,
    train_cluster_models,
    tune_hyperparams_weighted_cv,
)
from cluster_forecast.logutil import configure_logging, stage

FUTURE_HORIZON_MONTHS = 12

ARTIFACT_SUFFIXES = {".joblib", ".json", ".csv"}

LOG = logging.getLogger("trainer")


def publish_model_artifacts(source_dir: Path, *dest_dirs: Path) -> None:
    """Copy model artifacts from source_dir to each dest (overwrite existing files)."""
    source_dir.mkdir(parents=True, exist_ok=True)
    files = [
        p
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ARTIFACT_SUFFIXES
    ]
    for dest in dest_dirs:
        dest.mkdir(parents=True, exist_ok=True)
        for src_file in files:
            shutil.copy2(src_file, dest / src_file.name)
    LOG.info("Copied %s artifact file(s) → %s destination(s)", len(files), len(dest_dirs))


class Trainer:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent.parent
        self.data_path = self.root / "data" / "global_air_quality_2014_2025.csv"
        self.models_dir = self.root / "models"
        self.backend_models_dir = self.root.parent / "backend" / "models"
        self.models_dirs = (self.models_dir, self.backend_models_dir)

    @property
    def report_path(self) -> Path:
        return self.models_dir / "report.json"

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        LOG.info(
            "Data: %s rows, %s cities, period %s — %s",
            len(df),
            df[["Country", "State", "City"]].drop_duplicates().shape[0],
            df[DATE_COL].min().date() if df[DATE_COL].notna().any() else "?",
            df[DATE_COL].max().date() if df[DATE_COL].notna().any() else "?",
        )
        return df

    def train(self) -> pd.DataFrame:
        with stage("1/7 Load data"):
            df = self.load_data()

        with stage(f"2/7 Cluster cities (k={N_CLUSTERS_DEFAULT})"):
            df, city_profile = assign_city_clusters(df, n_clusters=N_CLUSTERS_DEFAULT)
            LOG.info(
                "Distribution by cluster_id:\n%s",
                city_profile["cluster_id"].value_counts().sort_index().to_string(),
            )

        target_columns = [c for c in TARGET_COLUMNS if c in df.columns]
        LOG.info("Target columns (%s): %s", len(target_columns), ", ".join(target_columns))

        with stage("3/7 Residual groups A/B (ARCH, §6.2.4)"):
            resid_groups = compute_residual_variance_groups(df)
            for cid in sorted(resid_groups):
                code, label = resid_groups[cid]
                LOG.info("  cluster %s: group %s (%s)", cid, code, label)

        with stage(
            f"4/7 Hyperparameter tuning §6.4.1 "
            f"(holdout={HOLDOUT_DEFAULT} mo., folds={CV_N_FOLDS}) — usually slowest step"
        ):
            _, tuned_by_cluster = tune_hyperparams_weighted_cv(
                df,
                target_columns,
                holdout=HOLDOUT_DEFAULT,
                cv_min_train_months=CV_MIN_TRAIN_MONTHS,
                cv_n_folds=CV_N_FOLDS,
            )

        with stage("5/7 Model-type selection §6.4.2 (weighted CV)"):
            model_selection = select_models_weighted_cv(
                df,
                target_columns,
                residual_variance_group=resid_groups,
                holdout=HOLDOUT_DEFAULT,
                cv_min_train_months=CV_MIN_TRAIN_MONTHS,
                cv_n_folds=CV_N_FOLDS,
                tuned_by_cluster=tuned_by_cluster,
            )

        with stage("6/7 Train on full history and save .joblib"):
            results = train_cluster_models(
                df=df,
                city_profile=city_profile,
                models_dir=self.models_dir,
                target_columns=target_columns,
                holdout=HOLDOUT_DEFAULT,
                future_horizon=FUTURE_HORIZON_MONTHS,
                model_by_cluster=model_selection,
            )

        with stage("7/7 Copy artifacts to backend/models"):
            publish_model_artifacts(self.models_dir, self.backend_models_dir)

        return results

    def save_report(self, results: pd.DataFrame) -> None:
        with stage("Report report.json"):
            trained_models = {
                "SeasonalNaiveVector",
                "HoltWintersVector",
                "RidgeMultiOutput",
                "WLSRidgeMultiOutput",
                "GradientBoostingVector",
            }
            summary = {
                "n_clusters_trained": int(results["model"].isin(trained_models).sum()),
                "models_per_cluster": results.set_index("cluster_id")["model"].astype(str).to_dict(),
                "mae_mean_per_cluster": {
                    str(int(r["cluster_id"])): None
                    if pd.isna(r["mae_mean"])
                    else float(r["mae_mean"])
                    for _, r in results.iterrows()
                },
            }
            self.models_dir.mkdir(parents=True, exist_ok=True)
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            publish_model_artifacts(self.models_dir, self.backend_models_dir)
            LOG.info("Report: %s", self.report_path.resolve())

    def sync_inference_to_backend(self) -> None:
        """Copy inference Python modules from cluster_forecast/ into backend/src."""
        script = self.root / "scripts" / "sync_inference_to_backend.py"
        if not script.is_file():
            LOG.warning("Sync script not found, skipping: %s", script)
            return
        with stage("Sync inference code to backend/src"):
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                for line in result.stdout.rstrip().splitlines():
                    LOG.info("  %s", line)
            if result.returncode != 0:
                if result.stderr:
                    LOG.error("%s", result.stderr.rstrip())
                raise RuntimeError(
                    f"sync_inference_to_backend.py failed with exit code {result.returncode}"
                )

    def run(self) -> pd.DataFrame:
        configure_logging()
        logging.getLogger("trainer").setLevel(logging.INFO)
        if not LOG.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
            )
            LOG.addHandler(handler)

        t0 = time.perf_counter()
        LOG.info("=== trainer.py start (root: %s) ===", self.root.resolve())

        results = self.train()
        self.save_report(results)
        self.sync_inference_to_backend()

        LOG.info("=== Done in %.1f s ===", time.perf_counter() - t0)
        print(results.to_string(index=False))
        print("\nArtifacts saved (overwrite):")
        for d in self.models_dirs:
            print(f"  - {d.resolve()}")
        return results


if __name__ == "__main__":
    Trainer().run()
