"""Helpers used only by chaos experiments notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd


def section_bar(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def concise_conclusion(lines: list[str]) -> None:
    print("\nКраткий вывод:")
    for line in lines:
        print(f"- {line}")


def iqr_outlier_share(s: pd.Series) -> float:
    """
    Outlier share by the IQR rule.
    - 0.25 and 0.75 are the 25th and 75th percentiles (Q1 and Q3).
    - IQR = Q3 - Q1 is the spread of the central bulk.
    - 1.5 * IQR is the classic Tukey threshold for outliers.
    """
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((s < lo) | (s > hi)).mean())


def build_cluster_monthly_aqi(
    df: pd.DataFrame,
    date_col: str,
    aqi_col: str,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Monthly mean AQI by cluster_id (minimum step: calendar month)."""
    return (
        df.set_index(date_col)
        .groupby(cluster_col)[aqi_col]
        .resample("MS")
        .mean()
        .reset_index()
    )


def cluster_balance_table(
    df: pd.DataFrame,
    city_profile: pd.DataFrame,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Cluster sizes: city counts and row share in the panel."""
    city_group = ["Country", "State", "City"]
    n_cities = city_profile.groupby(cluster_col).size().rename("n_cities")
    n_rows = df.groupby(cluster_col).size().rename("n_rows")
    out = pd.concat([n_cities, n_rows], axis=1)
    out["share_rows"] = (out["n_rows"] / out["n_rows"].sum()).round(4)
    return out.sort_index()
