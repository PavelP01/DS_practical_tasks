"""Streamlit UI for air quality monthly forecast."""

from __future__ import annotations

import os
import textwrap
from datetime import datetime
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st


# Fallback if API does not return model_display_name (older backend)
MODEL_LABELS: dict[str, str] = {
    "SeasonalNaiveVector": "Seasonal Naive",
    "HoltWintersVector": "Holt–Winters",
    "RidgeMultiOutput": "Ridge (мульти-выход)",
    "GradientBoostingVector": "Gradient Boosting",
    "WLSRidgeMultiOutput": "WLS-Ridge",
}

# История / прогноз — контрастная пара (cyan vs red)
HISTORY_LINE_COLOR = "#00FFFF"
FORECAST_LINE_COLOR = "#FF0000"
FORECAST_INTERVAL_COLOR = "#FF6666"
SERIES_LINE_WIDTH = 2.5
MARKER_BORDER_WIDTH = 2.5
MONTH_AXIS_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


class AirQualityForecastApp:
    def __init__(self, api_base_url: str = "http://0.0.0.0:8000") -> None:
        self.api_base_url = api_base_url.rstrip("/")
        st.set_page_config(
            page_title="Air Quality Forecast",
            page_icon="🌫️",
            layout="wide",
        )

    def _get(self, path: str, params: dict | None = None) -> Any:
        response = requests.get(f"{self.api_base_url}{path}", params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict) -> Any:
        response = requests.post(f"{self.api_base_url}{path}", json=payload, timeout=120)
        return response

    @staticmethod
    def _show_html(html: str) -> None:
        """Render HTML (markdown treats indented blocks as code)."""
        clean = textwrap.dedent(html).strip()
        st.html(clean)

    @st.cache_data(ttl=3600)
    def load_meta(_self) -> tuple[list[str], list[dict], dict[str, str]]:
        pollutants = _self._get("/meta/pollutants/")["pollutants"]
        buckets = _self._get("/meta/aqi-buckets/")["buckets"]
        colors = {b["name"]: b["color"] for b in buckets}
        return pollutants, buckets, colors

    @st.cache_data(ttl=3600)
    def load_countries(_self) -> list[str]:
        return _self._get("/meta/countries/")["countries"]

    @st.cache_data(ttl=3600)
    def load_cities(_self, country: str) -> list[str]:
        return _self._get("/meta/cities/", params={"country": country})["cities"]

    def run(self) -> None:
        st.title("🌫️ Прогноз качества воздуха")
        st.caption(
            "Месячный прогноз по кластеру города. На одном графике: **история** и **прогноз** "
            "на общей шкале Jan–Dec (год — во всплывающей подсказке). "
            "Прогноз: точка + коридор min/max (не строгий 95% ДИ — см. подсказку после запроса)."
        )

        try:
            pollutants, bucket_specs, bucket_colors = self.load_meta()
            countries = self.load_countries()
        except Exception as exc:
            st.error(
                f"Не удалось подключиться к API ({self.api_base_url}). "
                f"Запустите backend. Ошибка: {exc}"
            )
            return

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            country = st.selectbox("Страна", options=countries, index=0)
        with col2:
            cities = self.load_cities(country) if country else []
            city = st.selectbox("Город", options=cities, index=0 if cities else None)
        with col3:
            horizon = st.slider("Горизонт (мес.)", min_value=1, max_value=12, value=12)

        pollutant_mode = st.radio(
            "Загрязнитель",
            options=["all", "single"],
            format_func=lambda x: "Все (5 + AQI)" if x == "all" else "Один загрязнитель",
            horizontal=True,
        )
        selected_pollutants: list[str] | str = "all"
        if pollutant_mode == "single":
            selected_pollutants = [
                st.selectbox("Выберите загрязнитель", options=pollutants)
            ]

        self._render_legend(bucket_specs)

        if st.button("Получить прогноз", type="primary", use_container_width=False):
            if not city:
                st.warning("Выберите город")
                return
            self._fetch_and_render(
                country=country,
                city=city,
                horizon=horizon,
                pollutants=selected_pollutants,
                bucket_colors=bucket_colors,
                all_pollutants=pollutants,
            )

    def _render_legend(self, bucket_specs: list[dict]) -> None:
        st.markdown("**AQI Reference (категории)**")
        chips = []
        for spec in bucket_specs:
            hi = spec["max"] if spec["max"] is not None else "∞"
            chips.append(
                f'<span style="display:inline-block;margin:2px 6px 2px 0;padding:4px 10px;'
                f'border-radius:6px;background:{spec["color"]};color:#fff;font-size:12px;">'
                f'{spec["name"]} ({spec["min"]}–{hi})</span>'
            )
        self._show_html("".join(chips))

    def _fetch_and_render(
        self,
        country: str,
        city: str,
        horizon: int,
        pollutants: list[str] | str,
        bucket_colors: dict[str, str],
        all_pollutants: list[str],
    ) -> None:
        payload = {
            "country": country,
            "city": city,
            "pollutants": pollutants,
            "horizon_months": horizon,
        }
        with st.spinner("Запрос прогноза..."):
            try:
                response = self._post("/forecast/", payload)
            except Exception as exc:
                st.error(f"Ошибка сети: {exc}")
                return

        if response.status_code == 404:
            st.error(response.json().get("detail", "Город не найден"))
            return
        if response.status_code >= 400:
            st.error(f"Ошибка API ({response.status_code})")
            st.json(response.json())
            return

        data = response.json()
        model_label = data.get("model_display_name") or MODEL_LABELS.get(
            data.get("model_name", ""), data.get("model_name", "—")
        )
        spread_label = data.get("residual_spread_label")
        spread_txt = f" · **{spread_label}**" if spread_label else ""
        ih_label = data.get("interval_horizon_label") or data.get("interval_horizon", "")
        st.success(
            f"Кластер **{data['cluster_id']}** · модель **{model_label}** · "
            f"{data['horizon_months']} мес.{spread_txt} · коридор: **{ih_label}**"
        )
        ih_note = data.get("interval_horizon_note")
        spread_note = data.get("residual_spread_note")
        with st.expander("Как ведёт себя коридор min/max на горизонте"):
            st.markdown("**Ширина коридора по месяцам горизонта**")
            if ih_note:
                st.markdown(ih_note)
            else:
                st.caption("Нет описания для данного типа модели.")
            st.markdown("**Разброс остатков по истории кластера (группа A/B)**")
            if spread_note:
                st.markdown(spread_note)
            elif spread_label:
                st.markdown(spread_label)
            else:
                st.caption(
                    "Группа A/B не задана в метаданных кластера. "
                    "Переобучите модели (`trainer.py`) после диагностики остатков (§6.2.4)."
                )

        months = data["months"]
        history_months = data.get("history_months") or []
        show_all = pollutants == "all"

        st.subheader("AQI — история 2025 и прогноз")
        self._render_error_bar_plot(
            months=months,
            history_months=history_months,
            mode="aqi",
            bucket_colors=bucket_colors,
            title="AQI",
        )

        if show_all:
            for pol in all_pollutants:
                st.subheader(pol)
                self._render_error_bar_plot(
                    months=months,
                    history_months=history_months,
                    mode="pollutant",
                    pollutant=pol,
                    title=pol,
                )
        elif isinstance(pollutants, list) and pollutants:
            pol = pollutants[0]
            st.subheader(pol)
            self._render_error_bar_plot(
                months=months,
                history_months=history_months,
                mode="pollutant",
                pollutant=pol,
                title=pol,
            )

    @staticmethod
    def _parse_month_date(date_str: str) -> datetime:
        return datetime.strptime(date_str, "%Y-%m-%d")

    @staticmethod
    def _month_axis_label(dt: datetime) -> str:
        return MONTH_AXIS_LABELS[dt.month - 1]

    @staticmethod
    def _hover_date_label(dt: datetime) -> str:
        return dt.strftime("%b %Y")

    @staticmethod
    def _extract_history_series(
        history_months: list[dict],
        mode: str,
        pollutant: str | None = None,
    ) -> tuple[list[str], list[datetime], list[float], list[str]]:
        x_months: list[str] = []
        dates: list[datetime] = []
        values: list[float] = []
        labels: list[str] = []

        for month in history_months:
            dt = AirQualityForecastApp._parse_month_date(month["date"])
            if mode == "aqi":
                x_months.append(AirQualityForecastApp._month_axis_label(dt))
                dates.append(dt)
                values.append(float(month["aqi"]))
                labels.append(str(month["aqi_bucket"]))
            else:
                val = month.get("pollutants", {}).get(pollutant or "")
                if val is None or val == "":
                    continue
                x_months.append(AirQualityForecastApp._month_axis_label(dt))
                dates.append(dt)
                values.append(float(val))
                labels.append(pollutant or "")

        return x_months, dates, values, labels

    @staticmethod
    def _extract_series(
        months: list[dict],
        mode: str,
        pollutant: str | None = None,
    ) -> tuple[list[str], list[datetime], list[float], list[float], list[float], list[str]]:
        x_months: list[str] = []
        dates: list[datetime] = []
        values: list[float] = []
        y_min: list[float] = []
        y_max: list[float] = []
        labels: list[str] = []

        for month in months:
            dt = AirQualityForecastApp._parse_month_date(month["date"])
            if mode == "aqi":
                x_months.append(AirQualityForecastApp._month_axis_label(dt))
                dates.append(dt)
                values.append(float(month["aqi"]))
                y_min.append(float(month["aqi_min"]))
                y_max.append(float(month["aqi_max"]))
                labels.append(str(month["aqi_bucket"]))
            else:
                pol_data = month["pollutants"].get(pollutant or "", {})
                if not pol_data:
                    continue
                x_months.append(AirQualityForecastApp._month_axis_label(dt))
                dates.append(dt)
                values.append(float(pol_data["value"]))
                y_min.append(float(pol_data["min"]))
                y_max.append(float(pol_data["max"]))
                labels.append(pollutant or "")

        return x_months, dates, values, y_min, y_max, labels

    def _error_bar_figure(
        self,
        months: list[dict],
        mode: str,
        bucket_colors: dict[str, str] | None = None,
        pollutant: str | None = None,
        title: str = "",
        history_months: list[dict] | None = None,
    ) -> go.Figure | None:
        history_months = history_months or []
        if mode == "pollutant":
            months = [
                m
                for m in months
                if pollutant and pollutant in m.get("pollutants", {})
            ]
            history_months = [
                m
                for m in history_months
                if pollutant and pollutant in m.get("pollutants", {})
            ]
        if not months and not history_months:
            return None

        traces: list[go.Scatter] = []
        has_history = False
        has_forecast = False

        if history_months:
            hist_x, hist_dates, hist_values, hist_labels_meta = (
                self._extract_history_series(history_months, mode, pollutant)
            )
            if hist_values:
                has_history = True
                if mode == "aqi":
                    hist_marker_colors = [
                        (bucket_colors or {}).get(label, "#64748B")
                        for label in hist_labels_meta
                    ]
                    hist_hover = [
                        f"{self._hover_date_label(dt)}<br>AQI {val:.1f}<br>{label}"
                        for dt, val, label in zip(
                            hist_dates, hist_values, hist_labels_meta
                        )
                    ]
                else:
                    hist_marker_colors = [HISTORY_LINE_COLOR] * len(hist_values)
                    hist_hover = [
                        f"{self._hover_date_label(dt)}<br>{val:.1f}"
                        for dt, val in zip(hist_dates, hist_values)
                    ]

                traces.append(
                    go.Scatter(
                        x=hist_x,
                        y=hist_values,
                        mode="markers+lines",
                        showlegend=False,
                        line=dict(color=HISTORY_LINE_COLOR, width=SERIES_LINE_WIDTH),
                        marker=dict(
                            size=10,
                            color=hist_marker_colors,
                            line=dict(
                                width=MARKER_BORDER_WIDTH,
                                color=HISTORY_LINE_COLOR,
                            ),
                        ),
                        hovertext=hist_hover,
                        hoverinfo="text",
                    )
                )

        if months:
            fc_x, fc_dates, values, y_min, y_max, labels = self._extract_series(
                months, mode, pollutant
            )
            if values:
                has_forecast = True
                err_plus = [max(0.0, hi - val) for val, hi in zip(values, y_max)]
                err_minus = [max(0.0, val - lo) for val, lo in zip(values, y_min)]

                if mode == "aqi":
                    marker_colors = [
                        (bucket_colors or {}).get(label, "#64748B") for label in labels
                    ]
                    hover = [
                        f"{self._hover_date_label(dt)}<br>AQI {val:.1f}<br>{label}"
                        f"<br>min/max: {lo:.1f} – {hi:.1f}"
                        for dt, val, label, lo, hi in zip(
                            fc_dates, values, labels, y_min, y_max
                        )
                    ]
                else:
                    marker_colors = [FORECAST_LINE_COLOR] * len(values)
                    hover = [
                        f"{self._hover_date_label(dt)}<br>{val:.1f}"
                        f"<br>min/max: {lo:.1f} – {hi:.1f}"
                        for dt, val, lo, hi in zip(fc_dates, values, y_min, y_max)
                    ]

                traces.append(
                    go.Scatter(
                        x=fc_x,
                        y=values,
                        mode="markers+lines",
                        showlegend=False,
                        line=dict(color=FORECAST_LINE_COLOR, width=SERIES_LINE_WIDTH),
                        marker=dict(
                            size=12,
                            color=marker_colors,
                            line=dict(
                                width=MARKER_BORDER_WIDTH,
                                color=FORECAST_LINE_COLOR,
                            ),
                        ),
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=err_plus,
                            arrayminus=err_minus,
                            color=FORECAST_INTERVAL_COLOR,
                            thickness=1.5,
                            width=6,
                        ),
                        hovertext=hover,
                        hoverinfo="text",
                    )
                )

        if not traces:
            return None

        if has_history:
            traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name="История",
                    line=dict(color=HISTORY_LINE_COLOR, width=SERIES_LINE_WIDTH),
                )
            )
        if has_forecast:
            traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name="Прогноз",
                    line=dict(color=FORECAST_LINE_COLOR, width=SERIES_LINE_WIDTH),
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=title, x=0, font=dict(size=14)),
            xaxis_title="Месяц",
            yaxis_title=title,
            height=360,
            margin=dict(l=40, r=20, t=48, b=40),
            showlegend=has_history or has_forecast,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified",
        )
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=MONTH_AXIS_LABELS,
        )
        fig.update_yaxes(rangemode="tozero")
        return fig

    def _render_error_bar_plot(
        self,
        months: list[dict],
        mode: str,
        bucket_colors: dict[str, str] | None = None,
        pollutant: str | None = None,
        title: str = "",
        history_months: list[dict] | None = None,
    ) -> None:
        fig = self._error_bar_figure(
            months=months,
            mode=mode,
            bucket_colors=bucket_colors,
            pollutant=pollutant,
            title=title,
            history_months=history_months,
        )
        if fig is None:
            st.info("Нет данных для графика.")
            return
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    api_base_url = os.getenv("API_BASE_URL", "http://0.0.0.0:8000")
    AirQualityForecastApp(api_base_url=api_base_url).run()
