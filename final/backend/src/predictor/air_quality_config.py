"""AQI reference scale and forecast target columns."""

POLLUTANT_COLUMNS = [
    "PM2.5 (ug/m3)",
    "PM10 (ug/m3)",
    "NO2 (ug/m3)",
    "SO2 (ug/m3)",
    "O3 (ug/m3)",
]

AQI_COLUMN = "AQI"

TARGET_COLUMNS = POLLUTANT_COLUMNS + [AQI_COLUMN]

# AQI Reference (dataset definition)
AQI_BUCKET_SPECS = [
    {"name": "Good", "min": 0, "max": 50, "color": "#22C55E"},
    {"name": "Satisfactory", "min": 51, "max": 100, "color": "#84CC16"},
    {"name": "Moderate", "min": 101, "max": 150, "color": "#EAB308"},
    {"name": "Poor", "min": 151, "max": 200, "color": "#F97316"},
    {"name": "Very Poor", "min": 201, "max": 300, "color": "#EF4444"},
    {"name": "Severe", "min": 301, "max": None, "color": "#7F1D1D"},
]

INTERVAL_Z = 1.96  # ~95% interval from residual / HW CI

# Canonical model_type strings from train_model G4 / saved .joblib artifacts
PANEL_MODEL_CANDIDATES = (
    "SeasonalNaiveVector",
    "HoltWintersVector",
    "RidgeMultiOutput",
    "GradientBoostingVector",
    "WLSRidgeMultiOutput",
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "SeasonalNaiveVector": "Seasonal Naive",
    "HoltWintersVector": "Holt–Winters",
    "RidgeMultiOutput": "Ridge (мульти-выход)",
    "GradientBoostingVector": "Gradient Boosting",
    "WLSRidgeMultiOutput": "WLS-Ridge",
}


def model_display_name(model_type: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_type, model_type)


def interval_horizon_behavior(model_type: str) -> str:
    """How min/max band half-width scales with forecast step h (months from history end)."""
    if model_type == "SeasonalNaiveVector":
        return "const"
    if model_type in PANEL_MODEL_CANDIDATES and model_type != "SeasonalNaiveVector":
        return "sqrt_h"
    return "—"


_Z_NOTE = (
    f"**z = {INTERVAL_Z}** — квантиль стандартного нормального распределения для двустороннего "
    "интервала (~95% при нормальных остатках); min/max = прогноз ∓ z·σ (или z·σ·√h)."
)

INTERVAL_HORIZON_LABELS: dict[str, str] = {
    "const": f"постоянная ширина (±{INTERVAL_Z}·σ)",
    "sqrt_h": f"растёт как √h (±{INTERVAL_Z}·σ·√h)",
    "—": "—",
}

INTERVAL_HORIZON_NOTES: dict[str, str] = {
    "const": (
        "Полоса min/max **одинаковой ширины** на все месяцы горизонта: прогноз ± z·σ. "
        f"{_Z_NOTE} **σ** — типичная ошибка по истории (одна на весь горизонт)."
    ),
    "sqrt_h": (
        "Полуширина **растёт как √h** (h — номер месяца от конца истории): прогноз ± z·σ·√h. "
        f"{_Z_NOTE} Неопределённость рекурсивного/сглаженного прогноза накапливается — "
        "на дальних месяцах коридор шире, чем на ближних."
    ),
}

RESIDUAL_SPREAD_LABELS: dict[str, str] = {
    "A": "разброс остатков постоянен",
    "B": "разброс остатков условно постоянен",
}

RESIDUAL_SPREAD_NOTES: dict[str, str] = {
    "A": (
        "По истории кластера разброс ошибки от месяца к месяцу **примерно одинаковый** — "
        "одна σ для коридора уместна."
    ),
    "B": (
        "По истории кластера разброс ошибки **меняется по месяцам**; в коридоре используется "
        "усреднённая σ — полоса **условно постоянной** ширины и может не совпадать с фактом "
        "на отдельных месяцах."
    ),
}


def interval_horizon_label(code: str) -> str:
    return INTERVAL_HORIZON_LABELS.get(code, code)


def interval_horizon_note(code: str) -> str:
    return INTERVAL_HORIZON_NOTES.get(
        code,
        f"Коридор min/max — ориентир по остаткам обучения; {_Z_NOTE}",
    )


def residual_spread_label(variance_group: str | None) -> str | None:
    if not variance_group:
        return None
    return RESIDUAL_SPREAD_LABELS.get(variance_group.upper())


def residual_spread_note(variance_group: str | None) -> str | None:
    if not variance_group:
        return None
    return RESIDUAL_SPREAD_NOTES.get(variance_group.upper())


def aqi_to_bucket(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 150:
        return "Moderate"
    if aqi <= 200:
        return "Poor"
    if aqi <= 300:
        return "Very Poor"
    return "Severe"
