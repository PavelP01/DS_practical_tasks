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
