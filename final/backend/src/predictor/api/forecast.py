from fastapi import APIRouter, HTTPException, Request

from predictor.schemas.forecast_schemas import ForecastRequest, ForecastResponse

forecast_router = APIRouter()


@forecast_router.post("/forecast/", response_model=ForecastResponse)
async def forecast(request: Request, body: ForecastRequest) -> ForecastResponse:
    service = request.app.state.forecast_service
    try:
        return service.forecast(body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Модели кластеров не найдены") from exc


@forecast_router.get("/meta/countries/")
async def meta_countries(request: Request) -> dict:
    service = request.app.state.forecast_service
    return {"countries": service.list_countries()}


@forecast_router.get("/meta/cities/")
async def meta_cities(request: Request, country: str) -> dict:
    service = request.app.state.forecast_service
    return {"cities": service.list_cities(country)}


@forecast_router.get("/meta/pollutants/")
async def meta_pollutants(request: Request) -> dict:
    service = request.app.state.forecast_service
    return {"pollutants": service.list_pollutants()}


@forecast_router.get("/meta/aqi-buckets/")
async def meta_aqi_buckets(request: Request) -> dict:
    service = request.app.state.forecast_service
    return {"buckets": service.list_aqi_buckets()}


@forecast_router.get("/meta/models/")
async def meta_models(request: Request) -> dict:
    service = request.app.state.forecast_service
    return {"models": service.list_model_types()}
