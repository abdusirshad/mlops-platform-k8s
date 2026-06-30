# Training image — runs train.py against the MLflow tracking server.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py MLproject python_env.yaml ./

# Default: train against the MLflow service. Override MLFLOW_TRACKING_URI as needed.
ENV MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MLFLOW_EXPERIMENT=iris-classifier \
    MODEL_NAME=iris-classifier

ENTRYPOINT ["python", "train.py"]
