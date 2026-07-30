# MLOps platform — local + Kubernetes workflow.
# Usage: make <target>

COMPOSE        ?= docker compose
KIND_CLUSTER   ?= mlops
NAMESPACE      ?= mlops
SERVING_IMAGE  ?= mlops-platform/serving:local
TRAINER_IMAGE  ?= mlops-platform/trainer:local
MLFLOW_IMAGE   ?= mlops-platform/mlflow:local

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: up
up: ## Bring up the full stack (MLflow + trainer + serving) via docker-compose
	$(COMPOSE) up --build -d
	@echo "MLflow UI:    http://localhost:5000"
	@echo "Serving API:  http://localhost:8000/docs"

.PHONY: logs
logs: ## Tail docker-compose logs
	$(COMPOSE) logs -f

.PHONY: train
train: ## Re-run the training job against the running MLflow server
	$(COMPOSE) run --rm trainer

.PHONY: serve
serve: ## (Re)start only the serving container
	$(COMPOSE) up --build -d serving
	@echo "Serving API:  http://localhost:8000/docs"

.PHONY: predict
predict: ## Send a sample prediction request to the serving API
	curl -s -X POST http://localhost:8000/predict \
		-H 'Content-Type: application/json' \
		-d '{"features": [5.1, 3.5, 1.4, 0.2]}'
	@echo ""

.PHONY: test
test: ## Run the local smoke test (compose up -> predict -> assert)
	bash scripts/smoke_test.sh

.PHONY: lint
lint: ## Lint Python sources with ruff
	python -m ruff check training serving

.PHONY: metrics
metrics: ## Curl the serving Prometheus /metrics endpoint
	curl -s http://localhost:8000/metrics | grep -E '^serving_' | head -n 30

.PHONY: monitoring-up
monitoring-up: ## Bring up Prometheus + Grafana (compose "monitoring" profile)
	$(COMPOSE) --profile monitoring up -d
	@echo "Grafana:      http://localhost:3000  (admin / admin)"
	@echo "Prometheus:   http://localhost:9090"

.PHONY: monitoring-down
monitoring-down: ## Stop the Prometheus + Grafana monitoring tier
	$(COMPOSE) --profile monitoring stop prometheus grafana

.PHONY: diagrams
diagrams: ## Render architecture + workflow PNGs (needs Graphviz on PATH)
	pip install -r docs/diagrams/requirements.txt
	python docs/diagrams/architecture.py
	python docs/diagrams/workflow.py
	@echo "Rendered docs/diagrams/architecture.png and workflow.png"

.PHONY: build-images
build-images: ## Build all three container images
	docker build -t $(MLFLOW_IMAGE)  ./mlflow
	docker build -t $(TRAINER_IMAGE) ./training
	docker build -t $(SERVING_IMAGE) ./serving

.PHONY: kind-up
kind-up: ## Create a local kind cluster (requires kind)
	kind create cluster --name $(KIND_CLUSTER)

.PHONY: kind-load
kind-load: build-images ## Load built images into the kind cluster
	kind load docker-image $(MLFLOW_IMAGE)  --name $(KIND_CLUSTER)
	kind load docker-image $(TRAINER_IMAGE) --name $(KIND_CLUSTER)
	kind load docker-image $(SERVING_IMAGE) --name $(KIND_CLUSTER)

.PHONY: k8s-deploy
k8s-deploy: kind-load ## Build, load images into kind, and apply manifests
	kubectl apply -k k8s/
	kubectl -n $(NAMESPACE) rollout status deploy/mlflow --timeout=180s
	kubectl -n $(NAMESPACE) wait --for=condition=complete job/trainer --timeout=300s
	kubectl -n $(NAMESPACE) rollout status deploy/serving --timeout=180s
	@echo "Port-forward:  kubectl -n $(NAMESPACE) port-forward svc/serving 8000:80"

.PHONY: k8s-delete
k8s-delete: ## Delete the K8s resources
	kubectl delete -k k8s/ --ignore-not-found

.PHONY: kind-down
kind-down: ## Delete the kind cluster
	kind delete cluster --name $(KIND_CLUSTER)

.PHONY: clean
clean: ## Tear down docker-compose and remove local mlruns/state
	-$(COMPOSE) down -v
	rm -rf mlruns __pycache__ */__pycache__ .pytest_cache
	find . -name '*.pyc' -delete 2>/dev/null || true
