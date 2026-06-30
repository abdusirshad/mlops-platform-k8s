# mlops-platform-k8s

[![ci](https://github.com/abdusirshad/mlops-platform-k8s/actions/workflows/ci.yml/badge.svg)](https://github.com/abdusirshad/mlops-platform-k8s/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.6-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A small but **end-to-end MLOps platform** you can actually run: train a model,
track and register it in **MLflow 3**, then serve it from a containerized
**FastAPI** service — locally via `docker compose` or on **Kubernetes** (kind/k3d).

It is deliberately lightweight (scikit-learn on the Iris dataset, no GPUs, no
cloud account required) so a reviewer can clone it and have a working
prediction endpoint in a couple of minutes — while still mirroring the shape of
a real production platform: a tracking/registry tier, reproducible training, an
accuracy promotion gate, and immutable model-serving containers.

> Built by **Md Irshad — Senior Cloud & AI Platform Engineer**. It distills the
> serving + registry pattern from an AI-for-SDLC platform I built on Azure AKS:
> models flow `train → MLflow registry → containerized serving → Kubernetes`.

---

## Architecture

```
                    docker compose  /  kind  (kustomize)
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │   training/ (train.py)            serving/ (app.py, FastAPI)       │
  │   ┌────────────────────┐          ┌────────────────────────────┐  │
  │   │ sklearn RandomForest│         │ GET  /health   (liveness)   │  │
  │   │ log params/metrics  │         │ GET  /ready    (model load) │  │
  │   │ register model      │         │ POST /predict               │  │
  │   │ accuracy gate (0.90)│         └─────────────┬──────────────┘  │
  │   └─────────┬───────────┘                       │ models:/iris/.. │
  │             │ log_model + register              │ load_model      │
  │             ▼                                    ▼                 │
  │        ┌──────────────────────────────────────────────────┐       │
  │        │           MLflow Tracking Server (3.6)            │       │
  │        │  experiments · runs · metrics · Model Registry    │       │
  │        │  backend store: SQLite   artifacts: filesystem    │       │
  │        │  --serve-artifacts (artifacts proxied over HTTP)  │       │
  │        └──────────────────────────────────────────────────┘       │
  └──────────────────────────────────────────────────────────────────┘
```

- **Training** logs parameters, metrics, and the fitted model to MLflow, then
  registers it under `iris-classifier`. A configurable **accuracy gate** makes
  the job fail (non-zero exit) when the model is below threshold, so a bad model
  is never promoted.
- **MLflow** is the system of record: experiment tracking **and** the model
  registry. It runs with `--serve-artifacts`, so artifacts are uploaded and
  downloaded over HTTP through the tracking server — no shared object store is
  required for the demo.
- **Serving** loads `models:/iris-classifier/latest` from the registry on first
  request and exposes a typed `/predict` API.

---

## Repository layout

```
03-mlops-platform-k8s/
├── docker-compose.yml          # MLflow + one-shot trainer + serving
├── Makefile                    # up / train / serve / k8s-deploy / clean ...
├── mlflow/
│   └── Dockerfile              # MLflow tracking server image
├── training/
│   ├── train.py                # sklearn training, MLflow-instrumented
│   ├── MLproject               # `mlflow run .` entry point
│   ├── python_env.yaml         # MLproject environment
│   ├── requirements.txt
│   └── Dockerfile
├── serving/
│   ├── app.py                  # FastAPI: /health, /ready, /predict
│   ├── requirements.txt
│   └── Dockerfile
├── k8s/                        # kustomize manifests (deploy on kind/k3d)
│   ├── 00-namespace.yaml
│   ├── 01-configmap.yaml
│   ├── 02-mlflow.yaml          # Deployment + Service + PVC
│   ├── 03-trainer-job.yaml     # Job that trains + registers once
│   ├── 04-serving.yaml         # Deployment + Service
│   ├── 05-ingress.yaml         # optional nginx Ingress
│   └── kustomization.yaml
├── scripts/
│   └── smoke_test.sh           # up -> wait -> predict -> assert
├── .github/workflows/ci.yml    # ruff lint + py build + docker build (no push)
├── .env.example
├── .gitignore
└── LICENSE                     # MIT, © Md Irshad
```

---

## Quickstart (Docker Compose)

**Prerequisites:** Docker with the Compose plugin.

```bash
# 1. Build images and start MLflow; the trainer runs once and registers a model,
#    then the serving API comes up.
make up            # == docker compose up --build -d

# 2. Send a prediction once serving is ready (give it ~30-60s on first build).
make predict
# {"prediction":0,"label":"setosa","model_name":"iris-classifier",
#  "model_uri":"models:/iris-classifier/latest"}

# 3. Explore:
#    MLflow UI       -> http://localhost:5000   (experiments + registry)
#    Serving Swagger -> http://localhost:8000/docs

# 4. Re-train with different hyper-parameters at any time:
docker compose run --rm trainer --n-estimators 400 --max-depth 8

# 5. Tear everything down (also removes the volume + local caches):
make clean
```

One-shot end-to-end check (builds, trains, predicts, asserts the label):

```bash
make test          # == bash scripts/smoke_test.sh
```

---

## Quickstart (Kubernetes via kind)

**Prerequisites:** [`kind`](https://kind.sigs.k8s.io/), `kubectl`, Docker.

```bash
make kind-up           # create a local kind cluster named "mlops"
make k8s-deploy        # build images, load into kind, kubectl apply -k k8s/
                       # waits for: mlflow rollout, trainer Job complete, serving rollout

# Reach the serving API:
kubectl -n mlops port-forward svc/serving 8000:80
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [6.3, 3.3, 6.0, 2.5]}'
# {"prediction":2,"label":"virginica", ...}

# Clean up:
make k8s-delete
make kind-down
```

The `k8s/` manifests are plain kustomize, so `kubectl apply -k k8s/` works on any
cluster. The trainer runs as a Kubernetes `Job`; serving runs as a 2-replica
`Deployment` whose readiness probe (`/ready`) only passes once the model loads
from the registry.

---

## Running training directly (no containers)

```bash
cd training
pip install -r requirements.txt
# point at a running MLflow server (e.g. `make up` started one on :5000)
MLFLOW_TRACKING_URI=http://localhost:5000 python train.py --n-estimators 300
```

`python train.py` exits non-zero if test accuracy is below
`--accuracy-threshold` (default `0.90`), demonstrating the promotion gate.

---

## How to run / verify

| Check | Command |
|---|---|
| Lint Python | `make lint` (`ruff check training serving`) |
| Byte-compile | `python -m compileall training serving` |
| Validate compose | `docker compose config` |
| Validate manifests | `kubectl apply -k k8s/ --dry-run=client` (needs kubectl) |
| Full local E2E | `make test` |
| CI (GitHub Actions) | lint + build + import-check + `docker build` for all 3 images, no push |

`ci.yml` is OSS-only (ruff + Python + Docker Buildx) so it runs green on a public
fork without any secrets or registry credentials.

---

## Configuration

All services read the same environment variables (see `.env.example`):

| Variable | Default | Used by |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | training, serving |
| `MLFLOW_EXPERIMENT` | `iris-classifier` | training |
| `MODEL_NAME` | `iris-classifier` | training, serving |
| `MODEL_STAGE` | `latest` | serving (registry version / alias / `latest`) |
| `ACCURACY_THRESHOLD` | `0.90` | training (promotion gate) |

In Kubernetes these are supplied by the `mlops-config` ConfigMap
(`k8s/01-configmap.yaml`).

---

## How this mirrors a real AI-for-SDLC MLOps platform

This repo is a scaled-down, fully-local version of the production pattern I run
on Azure AKS:

- **One registry as the source of truth.** Everything that gets served is
  resolved from the MLflow Model Registry by name + version/alias — never from a
  loose file. Here that is SQLite + filesystem artifacts; in production it is
  Postgres + S3/Azure Blob behind the same `models:/` URIs, so the serving code
  is identical.
- **A promotion gate, not just a metric.** Training *fails the job* below the
  accuracy threshold, the same way a CI/CD model-validation stage blocks a
  release.
- **Immutable, probe-driven serving.** The inference container is the unit of
  deployment; `/ready` gates traffic until the model is actually loadable,
  which is what lets you do safe rollouts and (in prod) canaries.
- **Same artifacts, two runtimes.** The identical images run under
  `docker compose` for local dev and under Kubernetes for the cluster — the only
  difference is the orchestration manifest, which is how a platform stays
  reproducible across environments.

---

## License

[MIT](LICENSE) © 2026 Md Irshad — Senior Cloud & AI Platform Engineer.
