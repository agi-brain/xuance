#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${SCRIPT_DIR}/../../"
PYTHON=python

ALGO="noisydqn"
ENV="atari"
ENV_ID="ALE/Breakout-v5"
ENV_TAG="${ENV_ID//\//_}"   # ALE_Breakout-v5
CONFIG_PATH="${SCRIPT_DIR}/${ALGO}_atari.yaml"

OUT_ROOT="${SCRIPT_DIR}/results/${ENV_TAG}"


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
    --config-path "${CONFIG_PATH}"\
    --result-path "${OUT_ROOT}/seed_${SEED}"; then
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))
  STATUS="SUCCESS"
  else
    STATUS="FAILED"
  fi

  echo "========== [Benchmark END] seed=${SEED} | status=${STATUS} | time=${ELAPSED}s =========="
  echo
done
