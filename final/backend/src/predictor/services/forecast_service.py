import json
from functools import lru_cache
from pathlib import Path

import joblib

from predictor.air_quality_config import (
    MODEL_DISPLAY_NAMES,
    PANEL_MODEL_CANDIDATES,
    POLLUTANT_COLUMNS,
    interval_horizon_behavior,
    interval_horizon_label,
    interval_horizon_note,
    model_display_name,
    residual_spread_label,
    residual_spread_note,
)
from predictor.cluster_forecast import forecast_from_artifact, history_from_artifact
from predictor.schemas.forecast_schemas import (
    ForecastMonth,
    ForecastRequest,
    ForecastResponse,
    HistoryMonth,
    ValueInterval,
)


class ForecastService:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._country_city_map = self._load_json("country_city_to_cluster_id.json")
        self._selection = self._load_json("cluster_model_selection_results.json")

    @staticmethod
    def _load_json(name: str) -> dict | list:
        path = Path(__file__).resolve().parents[3] / "models" / name
        if not path.exists():
            path = Path(__file__).resolve().parents[3].parent / "train_model" / "models" / name
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def resolve_cluster_id(self, country: str, city: str) -> int:
        key = f"{country}|{city}"
        if key not in self._country_city_map:
            raise KeyError(f"Город не найден: {country} / {city}")
        return int(self._country_city_map[key])

    @lru_cache(maxsize=16)
    def load_artifact(self, cluster_id: int) -> dict:
        path = self.models_dir / f"cluster_{cluster_id}__multioutput.joblib"
        if not path.exists():
            alt = Path(__file__).resolve().parents[3].parent / "train_model" / "models" / path.name
            path = alt if alt.exists() else path
        return joblib.load(path)

    def selection_row_for_cluster(self, cluster_id: int) -> dict | None:
        for row in self._selection:
            if int(row["cluster_id"]) == cluster_id:
                return row
        return None

    def model_name_for_cluster(self, cluster_id: int) -> str:
        row = self.selection_row_for_cluster(cluster_id)
        return str(row["model"]) if row else "unknown"

    def _interval_meta(
        self, cluster_id: int, model_type: str, artifact: dict
    ) -> tuple[str, str, str, str | None, str | None]:
        row = self.selection_row_for_cluster(cluster_id)
        code = str(
            (row or {}).get("interval_horizon")
            or artifact.get("interval_horizon")
            or interval_horizon_behavior(model_type)
        )
        variance_group = None
        if row and row.get("residual_variance_group"):
            variance_group = str(row["residual_variance_group"])
        elif row and row.get("группа_остатков"):
            variance_group = str(row["группа_остатков"])
        return (
            code,
            interval_horizon_label(code),
            interval_horizon_note(code),
            residual_spread_label(variance_group),
            residual_spread_note(variance_group),
        )

    def forecast(self, req: ForecastRequest) -> ForecastResponse:
        cluster_id = self.resolve_cluster_id(req.country, req.city)
        artifact = self.load_artifact(cluster_id)
        months_raw = forecast_from_artifact(artifact, horizon_months=req.horizon_months)

        pollutant_filter: set[str] | None = None
        if req.pollutants != "all":
            pollutant_filter = set(req.pollutants)

        months: list[ForecastMonth] = []
        for m in months_raw:
            pol = m["pollutants"]
            if pollutant_filter is not None:
                pol = {k: v for k, v in pol.items() if k in pollutant_filter}

            months.append(
                ForecastMonth(
                    date=m["date"],
                    pollutants={
                        k: ValueInterval(**v) for k, v in pol.items()
                    },
                    aqi=m["aqi"],
                    aqi_min=m["aqi_min"],
                    aqi_max=m["aqi_max"],
                    aqi_bucket=m["aqi_bucket"],
                )
            )

        history_raw = history_from_artifact(artifact, year=2025)
        history_months: list[HistoryMonth] = []
        for m in history_raw:
            pol = m["pollutants"]
            if pollutant_filter is not None:
                pol = {k: v for k, v in pol.items() if k in pollutant_filter}
            history_months.append(
                HistoryMonth(
                    date=m["date"],
                    pollutants=pol,
                    aqi=m["aqi"],
                    aqi_bucket=m["aqi_bucket"],
                )
            )

        model_type = str(
            artifact.get("selected_model")
            or artifact.get("model_type")
            or self.model_name_for_cluster(cluster_id)
        )
        ih_code, ih_label, ih_note, spread_label, spread_note = self._interval_meta(
            cluster_id, model_type, artifact
        )

        return ForecastResponse(
            country=req.country,
            city=req.city,
            cluster_id=cluster_id,
            horizon_months=req.horizon_months,
            model_name=model_type,
            model_display_name=model_display_name(model_type),
            interval_horizon=ih_code,
            interval_horizon_label=ih_label,
            interval_horizon_note=ih_note,
            residual_spread_label=spread_label,
            residual_spread_note=spread_note,
            history_months=history_months,
            months=months,
        )

    def list_countries(self) -> list[str]:
        countries = {key.split("|", 1)[0] for key in self._country_city_map}
        return sorted(countries)

    def list_cities(self, country: str) -> list[str]:
        prefix = f"{country}|"
        cities = [
            key.split("|", 1)[1]
            for key in self._country_city_map
            if key.startswith(prefix)
        ]
        return sorted(cities)

    @staticmethod
    def list_pollutants() -> list[str]:
        return list(POLLUTANT_COLUMNS)

    @staticmethod
    def list_aqi_buckets() -> list[dict]:
        from predictor.air_quality_config import AQI_BUCKET_SPECS

        return AQI_BUCKET_SPECS

    @staticmethod
    def list_model_types() -> list[dict[str, str]]:
        return [
            {"id": mid, "display_name": MODEL_DISPLAY_NAMES[mid]}
            for mid in PANEL_MODEL_CANDIDATES
        ]
