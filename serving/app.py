"""FastAPI inference service backed by the MLflow Model Registry.

The model is loaded lazily on first use (and cached) from the MLflow registry
using the ``models:/<name>/<stage-or-version>`` URI scheme. This keeps the
container start fast and lets /health respond even before a model exists.

The service is instrumented for Prometheus: every request is counted and timed,
predictions are counted per iris class, and gauges track the loaded model
version and in-flight requests. Metrics are exposed at ``GET /metrics``.

Environment variables:
  MLFLOW_TRACKING_URI  MLflow server URL (default http://localhost:5000)
  MODEL_NAME           Registered model name (default iris-classifier)
  MODEL_STAGE          Stage or version alias to load (default latest)
"""

from __future__ import annotations

import os
import time
from threading import Lock
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
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

# --- Prometheus metrics -----------------------------------------------------
# A dedicated registry keeps the exposition clean and import-safe under tests.
REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "serving_requests_total",
    "Total HTTP requests handled by the serving API.",
    ["method", "endpoint", "http_status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "serving_request_latency_seconds",
    "HTTP request latency in seconds, per endpoint.",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)
PREDICTION_COUNT = Counter(
    "serving_predictions_total",
    "Total predictions made, labelled by the predicted iris class.",
    ["iris_class"],
    registry=REGISTRY,
)
MODEL_VERSION = Gauge(
    "serving_model_version",
    "Version number of the model currently loaded from the MLflow registry.",
    registry=REGISTRY,
)
INFLIGHT = Gauge(
    "serving_inflight_requests",
    "Number of HTTP requests currently being processed.",
    registry=REGISTRY,
)

_model: Any = None
_model_version: int = 0
_model_lock = Lock()


def _model_uri() -> str:
    if MODEL_STAGE.lower() == "latest":
        return f"models:/{MODEL_NAME}/latest"
    return f"models:/{MODEL_NAME}/{MODEL_STAGE}"


def _resolve_version() -> int:
    """Best-effort lookup of the concrete registry version being served."""
    try:
        client = mlflow.tracking.MlflowClient()
        if MODEL_STAGE.lower() == "latest":
            versions = client.get_latest_versions(MODEL_NAME)
            if versions:
                return max(int(v.version) for v in versions)
        elif MODEL_STAGE.isdigit():
            return int(MODEL_STAGE)
        else:
            return int(client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE).version)
    except Exception:  # noqa: BLE001 - version is telemetry only, never fatal
        return 0
    return 0


def get_model() -> Any:
    """Load and cache the model from the MLflow registry (thread-safe)."""
    global _model, _model_version
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = mlflow.pyfunc.load_model(_model_uri())
                _model_version = _resolve_version()
                MODEL_VERSION.set(_model_version)
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


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """Count and time every request, and track in-flight concurrency."""
    endpoint = request.url.path
    method = request.method
    INFLIGHT.inc()
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        INFLIGHT.dec()
        # Skip the scrape endpoint itself to avoid self-referential noise.
        if endpoint != "/metrics":
            REQUEST_LATENCY.labels(method, endpoint).observe(time.perf_counter() - start)
            REQUEST_COUNT.labels(method, endpoint, str(status)).inc()


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


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint (text exposition format)."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        model = get_model()
        frame = pd.DataFrame([request.features], columns=FEATURE_NAMES)
        result = model.predict(frame)
        pred = int(result[0])
        label = CLASS_NAMES[pred] if 0 <= pred < len(CLASS_NAMES) else str(pred)
        PREDICTION_COUNT.labels(label).inc()
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
