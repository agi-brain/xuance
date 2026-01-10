#!/usr/bin/env bash
set -euo pipefail

# Run all benchmark scripts under SMAC/MMM2.
# This script assumes the directory layout:
# benchmarks/SMAC/MMM2/{iql,qmix,vdn}/run_*_MMM2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# List the per-algorithm scripts you want to run (in order).
SCRIPTS=(
  "${ROOT_DIR}/coma/run_coma_MMM2.sh"
  "${ROOT_DIR}/ippo/run_ippo_MMM2.sh"
  "${ROOT_DIR}/iql/run_iql_MMM2.sh"
  "${ROOT_DIR}/mappo/run_mappo_MMM2.sh"
  "${ROOT_DIR}/qmix/run_qmix_MMM2.sh"
  "${ROOT_DIR}/vdac/run_vdac_MMM2.sh"
  "${ROOT_DIR}/vdn/run_vdn_MMM2.sh"
  "${ROOT_DIR}/wqmix/run_wqmix_MMM2.sh"
)


START_ALL=$(date +%s)
echo "============================================================"
echo "[Benchmark SUITE START] SMAC / MMM2"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for s in "${SCRIPTS[@]}"; do
  if [ ! -f "$s" ]; then
    echo "[ERROR] missing script: $s" >&2
    exit 2
  fi
  if [ ! -x "$s" ]; then
    # Allow running even if executable bit is not set.
    chmod +x "$s" || true
  fi

  echo ""
  echo "------------------------------------------------------------"
  echo "[RUN] $s"
  echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "------------------------------------------------------------"

  bash "$s"

done

END_ALL=$(date +%s)
ELAPSED=$((END_ALL - START_ALL))

echo ""
echo "============================================================"
echo "[Benchmark SUITE END] SMAC / MMM2"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapse: ${ELAPSED}s"
echo "============================================================"