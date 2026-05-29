"""Human-readable summaries of model selection results (notebook §6.4.2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cluster_forecast.config import ModelConfig, model_config_from_json


_MODEL_TYPE_LABELS = {
    "SeasonalNaiveVector": "Сезонный наивный",
    "HoltWintersVector": "Holt–Winters",
    "RidgeMultiOutput": "Ridge на лагах",
    "WLSRidgeMultiOutput": "WLS-Ridge",
    "GradientBoostingVector": "Gradient Boosting",
}


def _interpret_sn_params(params: dict[str, Any]) -> str:
    lag = int(params.get("season_lag", 12))
    fallback = str(params.get("fallback", "last"))
    fb = (
        "если нет точки год назад — последний месяц"
        if fallback == "last"
        else "если нет точки год назад — среднее за 12 мес."
    )
    return f"опора на {lag} мес. назад; {fb}"


def _interpret_hw_params(params: dict[str, Any]) -> str:
    trend = params.get("trend", "add")
    seasonal = str(params.get("seasonal", "add"))
    damped = bool(params.get("damped_trend", False))
    parts: list[str] = []
    if trend is None:
        parts.append("без явного тренда — только уровень и сезон")
    elif trend == "add":
        parts.append(
            "аддитивный затухающий тренд" if damped else "аддитивный тренд без затухания"
        )
    else:
        parts.append(f"тренд={trend}")
    if seasonal == "add":
        parts.append("аддитивная годовая сезонность (12 мес.)")
    elif seasonal == "mul":
        parts.append("мультипликативная годовая сезонность (12 мес.)")
    return "; ".join(parts)


def _interpret_ridge_params(params: dict[str, Any]) -> str:
    alpha = float(params.get("alpha", 1.0))
    if alpha >= 10:
        reg = "сильная регуляризация — консервативный линейный прогноз"
    elif alpha <= 0.1:
        reg = "слабая регуляризация — гибкая линейная модель"
    else:
        reg = "умеренная регуляризация"
    return f"alpha={alpha:g} ({reg}); лаги 1/2/3/6/12 + календарь"


def _interpret_gb_params(params: dict[str, Any]) -> str:
    lr = float(params.get("learning_rate", 0.08))
    depth = int(params.get("max_depth", 5))
    n_iter = int(params.get("max_iter", 250))
    l2 = float(params.get("l2_regularization", 0.0))
    l2_note = f", L2={l2:g}" if l2 > 0 else ""
    return (
        f"lr={lr:g}, глубина={depth}, итераций={n_iter}{l2_note}; "
        "нелинейные связи по лагам и календарю"
    )


def _interpret_wls_params(params: dict[str, Any]) -> str:
    alpha = float(params.get("alpha", 1.0))
    window = int(params.get("vol_window", 6))
    return (
        f"alpha={alpha:g}; веса по обратной дисперсии (окно {window} мес.) — "
        "спокойные месяцы важнее для подгонки"
    )


def _interpret_hyperparams(model_type: str, params: dict[str, Any]) -> str:
    if model_type == "SeasonalNaiveVector":
        return _interpret_sn_params(params)
    if model_type == "HoltWintersVector":
        return _interpret_hw_params(params)
    if model_type == "RidgeMultiOutput":
        return _interpret_ridge_params(params)
    if model_type == "WLSRidgeMultiOutput":
        return _interpret_wls_params(params)
    if model_type == "GradientBoostingVector":
        return _interpret_gb_params(params)
    return ModelConfig.from_dict(model_type, params).label()


def _interpret_model_type(model_type: str) -> str:
    label = _MODEL_TYPE_LABELS.get(model_type, model_type)
    gloss = {
        "SeasonalNaiveVector": " — повторяет годовой профиль «как в этом месяце год назад»",
        "HoltWintersVector": " — экспоненциальное сглаживание с сезонным циклом 12 мес",
        "RidgeMultiOutput": " — линейная модель по памяти ряда (лаги всех 6 показателей)",
        "WLSRidgeMultiOutput": " — линейная модель с большим весом «тихих» месяцев",
        "GradientBoostingVector": " — деревья по тем же признакам, учитывает нелинейности",
    }
    return label + gloss.get(model_type, "")


def interpret_cluster_model_selection(row: pd.Series) -> str:
    """One-line Russian summary for a single cluster row from ``select_models_weighted_cv``."""
    cid = row.name if row.name is not None else row.get("cluster_id", "?")
    model = str(row.get("best_model", ""))
    if model in ("SKIPPED_SHORT_SERIES", "FAILED_NO_CV_FOLDS", "FAILED_ALL_MODELS", ""):
        return f"Кластер {cid}: модель не выбрана ({model or 'нет данных'})."

    grp = row.get("группа_остатков", "?")
    params = model_config_from_json(
        row.get("best_params_json") if pd.notna(row.get("best_params_json", np.nan)) else None,
        model,
    ).as_dict()
    mae = row.get("best_mae_mean", np.nan)
    mae_txt = f"{mae:.2f}" if np.isfinite(mae) else "—"
    di = row.get("ДИ_на_горизонте", "—")
    di_txt = (
        "коридор min/max постоянной ширины на горизонте"
        if di == "const"
        else "коридор min/max растёт как √h"
        if di == "sqrt_h"
        else f"коридор: {di}"
    )

    return (
        f"Кластер {cid} (группа {grp}): {_interpret_model_type(model)}. "
        f"Настройки: {_interpret_hyperparams(model, params)}. "
        f"cv_wmae≈{mae_txt}; {di_txt}."
    )


def interpret_model_selection_table(results: pd.DataFrame) -> list[str]:
    """One summary string per ``cluster_id`` in the selection table."""
    return [
        interpret_cluster_model_selection(results.loc[cid])
        for cid in sorted(results.index.astype(int))
    ]
