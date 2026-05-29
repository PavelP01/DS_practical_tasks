"""City clustering (KernelPCA + KMeans) and ARCH-based residual variance groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.seasonal import seasonal_decompose

from air_quality_config import AQI_COLUMN, TARGET_COLUMNS
from cluster_forecast.config import (
    CITY_GROUP_COLS,
    CLUSTER_DRIVERS,
    DATE_COL,
    KERNEL_PCA_GAMMA,
    KERNEL_PCA_KERNEL,
    N_CLUSTERS_DEFAULT,
    N_KERNEL_PCA_DEFAULT,
)


def _aqi_trend_slope(group: pd.DataFrame) -> float:
    g = group.sort_values(DATE_COL)
    y = g[AQI_COLUMN].astype(float).values
    x = np.arange(len(y), dtype=float)
    if len(y) < 2 or np.isclose(y.std(), 0):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def build_city_profile(df: pd.DataFrame) -> pd.DataFrame:
    """City-level features for clustering (experiments.ipynb §4.1)."""
    cluster_pollutants = [c for c in TARGET_COLUMNS if c != AQI_COLUMN and c in df.columns]
    drivers = [c for c in CLUSTER_DRIVERS if c in df.columns]

    agg_dict = {c: ["mean", "std"] for c in cluster_pollutants + [AQI_COLUMN]}
    agg_dict.update({c: ["mean"] for c in drivers})

    city_profile = df.groupby(CITY_GROUP_COLS).agg(agg_dict)
    city_profile.columns = ["__".join(col).strip("_") for col in city_profile.columns]
    city_profile = city_profile.reset_index()

    slopes = (
        df.groupby(CITY_GROUP_COLS)
        .apply(_aqi_trend_slope, include_groups=False)
        .reset_index(name="AQI_trend_slope")
    )
    city_profile = city_profile.merge(slopes, on=CITY_GROUP_COLS, how="left")

    feature_cols = [c for c in city_profile.columns if c not in CITY_GROUP_COLS]
    city_profile[feature_cols] = city_profile[feature_cols].replace([np.inf, -np.inf], np.nan)
    city_profile[feature_cols] = city_profile[feature_cols].fillna(
        city_profile[feature_cols].median()
    )
    return city_profile


def assign_city_clusters(
    df: pd.DataFrame,
    city_profile: pd.DataFrame | None = None,
    n_clusters: int = N_CLUSTERS_DEFAULT,
    n_kpca: int = N_KERNEL_PCA_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign ``cluster_id`` via StandardScaler → KernelPCA → KMeans."""
    if city_profile is None:
        city_profile = build_city_profile(df)
    else:
        city_profile = city_profile.copy()

    feature_cols = [c for c in city_profile.columns if c not in CITY_GROUP_COLS]
    x_scaled = StandardScaler().fit_transform(city_profile[feature_cols])
    x_cluster = KernelPCA(
        n_components=n_kpca,
        kernel=KERNEL_PCA_KERNEL,
        gamma=KERNEL_PCA_GAMMA,
        n_jobs=-1,
        random_state=42,
        eigen_solver="dense",
    ).fit_transform(x_scaled)

    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(x_cluster)
    city_profile["cluster_id"] = labels

    city_to_cluster = city_profile.set_index(CITY_GROUP_COLS)["cluster_id"]
    out = df.merge(city_to_cluster.rename("cluster_id"), on=CITY_GROUP_COLS, how="left")
    return out, city_profile


def compute_residual_variance_groups(
    df: pd.DataFrame,
    alpha: float = 0.05,
    min_months: int = 24,
) -> dict[int, tuple[str, str]]:
    """ARCH-based A/B groups on seasonal AQI residuals (experiments.ipynb §6.2.4)."""
    groups: dict[int, tuple[str, str]] = {}
    for cid in sorted(df["cluster_id"].dropna().astype(int).unique()):
        series = (
            df.loc[df["cluster_id"] == cid]
            .set_index(DATE_COL)[AQI_COLUMN]
            .sort_index()
            .resample("MS")
            .mean()
            .dropna()
        )
        if len(series) < min_months:
            continue
        decomp = seasonal_decompose(
            series, model="additive", period=12, extrapolate_trend="freq"
        )
        resid = decomp.resid.dropna()
        arch_p = float(het_arch(resid.values, nlags=12)[1])
        var_label = "гетероскедастичны" if arch_p < alpha else "гомоскедастичны"
        group_code = "B" if var_label == "гетероскедастичны" else "A"
        groups[cid] = (group_code, var_label)
    return groups
