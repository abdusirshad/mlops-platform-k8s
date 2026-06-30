"""FastAPI inference service backed by the MLflow Model Registry.

The model is loaded lazily on first use (and cached) from the MLflow registry
using the ``models:/<name>/<stage-or-version>`` URI scheme. This keeps the
container start fast and lets /health respond even before a model exists.

Environment variables:
  MLFLOW_TRACKING_URI  MLflow server URL (default http://localhost:5000)
  MODEL_NAME           Registered model name (default iris-classifier)
  MODEL_STAGE          Stage or version alias to load (default latest)
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "iris-classifier")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "latest")

# Iris feature order — the model was trained on these columns.
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Iris Model Serving API", version="1.0.0")

_model: Any = None
_model_lock = Lock()


def _model_uri() -> str:
    if MODEL_STAGE.lower() == "latest":
        return f"models:/{MODEL_NAME}/latest"
    return f"models:/{MODEL_NAME}/{MODEL_STAGE}"


def get_model() -> Any:
    """Load and cache the model from the MLflow registry (thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = mlflow.pyfunc.load_model(_model_uri())
    return _model


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Iris features: sepal length, sepal width, petal length, petal width.",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )


class PredictResponse(BaseModel):
    prediction: int
    label: str
    model_name: str
    model_uri: str


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — does not require a loaded model."""
    return {"status": "healthy"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness probe — confirms the model can be loaded from the registry."""
    try:
        get_model()
    except Exception as exc:  # noqa: BLE001 - surfaced to the probe as 503
        raise HTTPException(status_code=503, detail=f"model not ready: {exc}") from exc
    return {"status": "ready", "model_uri": _model_uri()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        model = get_model()
        frame = pd.DataFrame([request.features], columns=FEATURE_NAMES)
        result = model.predict(frame)
        pred = int(result[0])
        label = CLASS_NAMES[pred] if 0 <= pred < len(CLASS_NAMES) else str(pred)
        return PredictResponse(
            prediction=pred,
            label=label,
            model_name=MODEL_NAME,
            model_uri=_model_uri(),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - return a clean 500 to the client
        raise HTTPException(status_code=500, detail=str(exc)) from exc
