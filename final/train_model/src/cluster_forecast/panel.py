"""Monthly cluster-level panels used for forecasting and CV."""

from __future__ import annotations

import pandas as pd

from cluster_forecast.config import DATE_COL


def build_cluster_panel(cluster_df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """Aggregate city rows to one MS-indexed panel per cluster (mean over cities, drop NaN rows)."""
    panel = (
        cluster_df.set_index(DATE_COL)[target_columns]
        .groupby(level=0)
        .mean()
        .sort_index()
        .resample("MS")
        .mean()
        .dropna(how="all")
    )
    return panel.dropna(how="any")
