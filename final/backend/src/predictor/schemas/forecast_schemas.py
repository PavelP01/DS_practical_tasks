from typing import Literal

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    country: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    pollutants: list[str] | Literal["all"] = "all"
    horizon_months: int = Field(default=12, ge=1, le=12)


class ValueInterval(BaseModel):
    value: float
    min: float
    max: float


class ForecastMonth(BaseModel):
    date: str
    pollutants: dict[str, ValueInterval]
    aqi: float
    aqi_min: float
    aqi_max: float
    aqi_bucket: str


class HistoryMonth(BaseModel):
    date: str
    pollutants: dict[str, float]
    aqi: float
    aqi_bucket: str


class ForecastResponse(BaseModel):
    country: str
    city: str
    cluster_id: int
    horizon_months: int
    model_name: str
    model_display_name: str
    interval_horizon: str = Field(
        description="const — constant band width; sqrt_h — half-width grows as √h",
    )
    interval_horizon_label: str
    interval_horizon_note: str
    residual_spread_label: str | None = Field(
        default=None,
        description="Brief: constant / conditionally constant residual spread over cluster history",
    )
    residual_spread_note: str | None = None
    history_months: list[HistoryMonth] = Field(
        default_factory=list,
        description="Cluster facts for 2025 (no min/max band)",
    )
    months: list[ForecastMonth]


class ModelTypeSpec(BaseModel):
    id: str
    display_name: str


class AqiBucketSpec(BaseModel):
    name: str
    min: int
    max: int | None
    color: str
