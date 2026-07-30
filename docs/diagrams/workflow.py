"""Diagrams-as-code: the MLOps lifecycle / workflow.

Renders ``workflow.png`` — data -> train -> MLflow tracking -> register ->
accuracy gate (promote / reject) -> build serving image -> deploy to k8s ->
serve /predict -> monitor (Prometheus / Grafana) -> retrain loop.

Render:
    pip install -r docs/diagrams/requirements.txt   # needs Graphviz on PATH
    python docs/diagrams/workflow.py                # -> docs/diagrams/workflow.png
"""

from __future__ import annotations

from diagrams import Cluster, Diagram, Edge
from diagrams.generic.storage import Storage
from diagrams.k8s.compute import Deployment
from diagrams.onprem.monitoring import Grafana, Prometheus
from diagrams.programming.flowchart import Decision
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
        "MLOps Lifecycle: train -> gate -> deploy -> monitor -> retrain",
        filename="docs/diagrams/workflow",
        show=False,
        direction="LR",
        graph_attr=GRAPH_ATTR,
    ):
        data = Storage("Iris data")
        train = Python("train.py\nRandomForest")

        with Cluster("MLflow 3.6"):
            tracking = Python("tracking\nparams / metrics")
            registry = Python("model registry\niris-classifier")
            tracking >> registry

        gate = Decision("accuracy gate\n>= 0.90 ?")

        image = Deployment("build + deploy\nserving image")
        serve = Fastapi("serve /predict")

        with Cluster("Monitoring"):
            prom = Prometheus("Prometheus")
            graf = Grafana("Grafana")
            prom >> graf

        data >> train >> tracking
        registry >> gate
        gate >> Edge(label="promote", color="darkgreen") >> image >> serve
        gate >> Edge(label="reject / fail job", color="firebrick", style="dashed") >> train
        serve >> Edge(label="/metrics") >> prom
        graf >> Edge(label="drift / retrain trigger", style="dotted") >> train


if __name__ == "__main__":
    render()
