"""Diagrams-as-code: MLOps platform architecture.

Renders ``architecture.png`` — the training job, MLflow tracking + registry,
FastAPI serving, ingress, and the Prometheus + Grafana monitoring tier, all
inside a Kubernetes namespace ``mlops`` (mirrored by docker-compose locally).

Render:
    pip install -r docs/diagrams/requirements.txt   # needs Graphviz on PATH
    python docs/diagrams/architecture.py            # -> docs/diagrams/architecture.png
"""

from __future__ import annotations

from diagrams import Cluster, Diagram, Edge
from diagrams.k8s.compute import Deployment, Job
from diagrams.k8s.network import Ingress, Service
from diagrams.onprem.client import Client
from diagrams.onprem.monitoring import Grafana, Prometheus
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python

GRAPH_ATTR = {
    "fontsize": "20",
    "bgcolor": "white",
    "pad": "0.4",
    "splines": "spline",
}


def render() -> None:
    with Diagram(
        "MLOps Platform on Kubernetes (namespace: mlops)",
        filename="docs/diagrams/architecture",
        show=False,
        direction="LR",
        graph_attr=GRAPH_ATTR,
    ):
        user = Client("client / recruiter")

        with Cluster("Kubernetes namespace: mlops  (or docker compose)"):
            ingress = Ingress("nginx ingress")

            with Cluster("Training tier"):
                trainer = Job("trainer Job\n(train.py)")

            with Cluster("Tracking + Registry (MLflow 3.6)"):
                mlflow = Deployment("mlflow server\n--serve-artifacts")
                store = Python("SQLite backend\n+ artifact proxy")
                mlflow - Edge(style="dotted") - store

            with Cluster("Serving tier"):
                serving_svc = Service("serving svc")
                serving = Fastapi("FastAPI serving\n/health /ready\n/predict /metrics")
                serving_svc >> serving

            with Cluster("Monitoring tier"):
                prom = Prometheus("Prometheus\nscrape /metrics")
                graf = Grafana("Grafana\nmodel-serving\ndashboard")
                prom >> Edge(label="datasource") >> graf

        # Flows
        user >> Edge(label="HTTP") >> ingress >> serving_svc
        trainer >> Edge(label="log + register") >> mlflow
        serving >> Edge(label="models:/iris-classifier/latest", style="dashed") >> mlflow
        prom >> Edge(label="GET /metrics", color="firebrick") >> serving
        user >> Edge(label="dashboards", style="dotted") >> graf


if __name__ == "__main__":
    render()
