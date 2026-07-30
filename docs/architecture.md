# Architecture deep dive

This document is the visual + narrative deep dive for the MLOps platform. Every
diagram below renders directly on GitHub (Mermaid) and is mirrored by rendered
PNGs generated from `docs/diagrams/*.py` (mingrammer `diagrams`).

- Rendered architecture: [`diagrams/architecture.png`](diagrams/architecture.png)
- Rendered lifecycle: [`diagrams/workflow.png`](diagrams/workflow.png)

![Architecture](diagrams/architecture.png)

---

## 1. Platform architecture

```mermaid
flowchart LR
  client([Client / recruiter])

  subgraph ns["Kubernetes namespace: mlops  (or docker compose)"]
    ingress[[nginx ingress]]

    subgraph train_tier["Training tier"]
      trainer["trainer Job\ntrain.py — RandomForest"]
    end

    subgraph mlflow_tier["MLflow 3.6 — tracking + registry"]
      mlflow["mlflow server\n--serve-artifacts"]
      store[("SQLite backend\n+ artifact proxy")]
      mlflow --- store
    end

    subgraph serve_tier["Serving tier"]
      svc[[serving Service]]
      serving["FastAPI serving\n/health /ready /predict /metrics"]
      svc --> serving
    end

    subgraph mon_tier["Monitoring tier"]
      prom["Prometheus\nscrape /metrics"]
      graf["Grafana\nmodel-serving dashboard"]
      prom -->|datasource| graf
    end
  end

  client -->|HTTP| ingress --> svc
  trainer -->|log params/metrics + register| mlflow
  serving -.->|models:/iris-classifier/latest| mlflow
  prom -->|GET /metrics| serving
  client -.->|view dashboards| graf
```

The MLflow Model Registry is the single source of truth: everything served is
resolved by name + version/alias (`models:/iris-classifier/latest`), never from
a loose file.

---

## 2. MLOps lifecycle with the promotion gate

```mermaid
flowchart LR
  data[("Iris data")] --> train["train.py\nRandomForest"]
  train --> track["MLflow tracking\nparams / metrics"]
  track --> reg["MLflow registry\niris-classifier"]
  reg --> gate{"accuracy gate\n>= 0.90 ?"}
  gate -->|promote| build["build + deploy\nserving image"]
  gate -->|reject: job fails non-zero| train
  build --> serve["serve /predict"]
  serve -->|/metrics| mon["Prometheus + Grafana"]
  mon -.->|drift / retrain trigger| train
```

The gate is enforced in `training/train.py`: below `--accuracy-threshold`
(default `0.90`) the job raises `SystemExit`, exiting non-zero so a bad model is
never registered as servable — the same behaviour as a CI/CD model-validation
stage blocking a release.

![Workflow](diagrams/workflow.png)

---

## 3. Prediction request sequence

```mermaid
sequenceDiagram
  autonumber
  participant C as Client
  participant F as FastAPI serving
  participant M as MLflow registry
  participant P as Prometheus

  C->>F: POST /predict {features:[...]}
  Note over F: metrics middleware:\ninc in-flight, start timer
  alt model not yet cached (first call)
    F->>M: load models:/iris-classifier/latest
    M-->>F: pyfunc model + version
    Note over F: set serving_model_version gauge
  end
  F->>F: model.predict(features)
  F->>F: inc serving_predictions_total{iris_class}
  F-->>C: {prediction, label, model_name, model_uri}
  Note over F: observe latency, inc requests_total,\ndec in-flight
  P->>F: GET /metrics (every 15s)
  F-->>P: exposition text
```

The model is loaded lazily and cached (thread-safe) on the first request, so
container start stays fast and `/health` responds before a model exists.

---

## 4. CI/CD flow

```mermaid
flowchart LR
  push["git push / PR"] --> gha["GitHub Actions: ci.yml"]
  subgraph gha_jobs["ci.yml"]
    ruff["ruff check\ntraining serving"]
    compile["byte-compile\n+ import-check app"]
    docker["docker build (no push)\nmlflow · training · serving"]
    ruff --> compile --> docker
  end
  gha --> status(["green check ✔"])
```

CI is OSS-only (ruff + Python + Docker Buildx, `push: false`) so it runs green
on a public fork without any secrets or registry credentials.

---

## Serving metrics reference

Exposed at `GET /metrics` (Prometheus text exposition):

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `serving_requests_total` | counter | `method`, `endpoint`, `http_status` | HTTP requests handled |
| `serving_request_latency_seconds` | histogram | `method`, `endpoint` | Request latency (p50/p95/p99 via `histogram_quantile`) |
| `serving_predictions_total` | counter | `iris_class` | Predictions per predicted iris class |
| `serving_model_version` | gauge | — | Registry version currently loaded |
| `serving_inflight_requests` | gauge | — | Requests currently being processed |
