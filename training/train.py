"""Train a scikit-learn classifier and log everything to MLflow.

This is the training entrypoint for the MLOps demo. It:
  1. Loads the Iris dataset (bundled with scikit-learn, no network needed).
  2. Trains a RandomForestClassifier with configurable hyper-parameters.
  3. Logs params, metrics, and the fitted model to the MLflow tracking server.
  4. Registers the model in the MLflow Model Registry under MODEL_NAME.
  5. Enforces an accuracy gate so a bad model never gets promoted.

Run it against a running MLflow server:

    python train.py
    # or
    MLFLOW_TRACKING_URI=http://localhost:5000 python train.py --n-estimators 300
"""

from __future__ import annotations

import argparse
import os

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT = "iris-classifier"
DEFAULT_MODEL_NAME = "iris-classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Iris classifier and log to MLflow.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees.")
    parser.add_argument("--max-depth", type=int, default=5, help="Max tree depth.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=float(os.environ.get("ACCURACY_THRESHOLD", "0.90")),
        help="Minimum test accuracy required to register the model.",
    )
    return parser.parse_args()


def main() -> str:
    args = parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment = os.environ.get("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT)
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    data = load_iris(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data.target,
    )

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }

    with mlflow.start_run(tags={"environment": os.environ.get("ENV", "dev")}) as run:
        mlflow.log_params(params)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro")),
        }
        mlflow.log_metrics(metrics)

        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=x_train.iloc[:2],
            registered_model_name=model_name,
        )

        print(f"Run ID:    {run.info.run_id}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"F1 (macro):{metrics['f1_macro']:.4f}")

        if metrics["accuracy"] < args.accuracy_threshold:
            raise SystemExit(
                f"Accuracy {metrics['accuracy']:.4f} is below the gate "
                f"threshold {args.accuracy_threshold:.4f}; model NOT promoted."
            )

        print(
            f"Model '{model_name}' registered. "
            f"Accuracy gate ({args.accuracy_threshold:.2f}) passed."
        )
        return run.info.run_id


if __name__ == "__main__":
    main()
