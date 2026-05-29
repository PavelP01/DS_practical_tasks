"""API inference facade — DO NOT EDIT; run train_model/scripts/sync_inference_to_backend.py."""

from predictor.forecasting.inference import forecast_from_artifact, history_from_artifact

__all__ = ["forecast_from_artifact", "history_from_artifact"]
