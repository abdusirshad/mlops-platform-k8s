#!/usr/bin/env bash
# Local end-to-end smoke test:
#   1. Build + start the stack (MLflow, trainer runs once, serving comes up).
#   2. Wait for the serving /health and /ready endpoints.
#   3. POST a sample prediction and assert the label is "setosa".
#
# Requires: docker (with compose), curl. Run from the repo root: bash scripts/smoke_test.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Bringing up the stack (this builds images and runs training once)..."
docker compose up --build -d

cleanup() {
  echo "==> Logs (serving):"
  docker compose logs --tail=30 serving || true
}
trap cleanup EXIT

echo "==> Waiting for serving /health ..."
for i in $(seq 1 60); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    echo "    serving is healthy."
    break
  fi
  sleep 3
  if [ "$i" -eq 60 ]; then echo "ERROR: serving never became healthy" >&2; exit 1; fi
done

echo "==> Waiting for serving /ready (model loaded from registry) ..."
for i in $(seq 1 40); do
  if curl -fsS http://localhost:8000/ready >/dev/null 2>&1; then
    echo "    model is ready."
    break
  fi
  sleep 3
  if [ "$i" -eq 40 ]; then echo "ERROR: model never became ready" >&2; exit 1; fi
done

echo "==> Sending a sample prediction ..."
RESP=$(curl -fsS -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}')
echo "    response: $RESP"

if echo "$RESP" | grep -q '"label":"setosa"'; then
  echo "==> SMOKE TEST PASSED"
else
  echo "==> SMOKE TEST FAILED: unexpected response" >&2
  exit 1
fi
