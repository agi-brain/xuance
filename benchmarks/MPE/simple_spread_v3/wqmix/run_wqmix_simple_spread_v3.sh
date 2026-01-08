#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${SCRIPT_DIR}/../../../"
PYTHON=python

ALGO="wqmix"
ENV="mpe"
ENV_ID="simple_spread_v3"
CONFIG_PATH="${SCRIPT_DIR}/wqmix_simple_spread_v3.yaml"

OUT_ROOT="${SCRIPT_DIR}/results"


for SEED in 1 2 3 4 5; do
  WORKDIR="${OUT_ROOT}/seed_${SEED}"
  mkdir -p "${WORKDIR}"

  echo "========== [Benchmark START] seed=${SEED} =========="

  START_TIME=$(date +%s)
  if ${PYTHON} "${PROJECT_ROOT}/train.py" \
    --algo "${ALGO}" \
    --env "${ENV}" \
    --env-id "${ENV_ID}" \
    --seed "${SEED}" \
    --config-path "${CONFIG_PATH}"; then
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  STATUS="SUCCESS"
  else
    STATUS="FAILED"
  fi

  echo "========== [Benchmark END] seed=${SEED} | status=${STATUS} | time=${ELAPSED}s =========="
  echo
done
