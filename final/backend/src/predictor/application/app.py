from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predictor.api.forecast import forecast_router
from predictor.services.forecast_service import ForecastService


@asynccontextmanager
async def lifespan(app: FastAPI):
    local_path = Path(__file__).resolve().parent.parent.parent.parent
    models_dir = local_path / "models"
    if not models_dir.exists():
        models_dir = local_path.parent / "train_model" / "models"
    app.state.forecast_service = ForecastService(models_dir=models_dir)
    yield


app = FastAPI(title="Air Quality Forecast API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast_router)
