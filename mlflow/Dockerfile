# MLflow tracking server image — SQLite backend store + local filesystem artifacts.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir mlflow==3.6.0

WORKDIR /mlflow

# Backend store (experiment/run metadata + model registry) and artifact root
# both live under /mlflow, which is mounted as a shared volume in docker-compose
# so the serving container can resolve models:/ artifact paths.
EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/store/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--serve-artifacts"]
