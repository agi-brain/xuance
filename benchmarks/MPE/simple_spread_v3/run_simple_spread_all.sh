#!/usr/bin/env bash
set -euo pipefail

# Run all benchmark scripts under MPE/simple_spread_v3.
# This script assumes the directory layout:
# benchmarks/MPE/simple_spread_v3/{iql,qmix,vdn}/run_*_simple_spread_v3.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# List the per-algorithm scripts you want to run (in order).
SCRIPTS=(
  "${ROOT_DIR}/iql/run_iql_simple_spread_v3.sh"
  "${ROOT_DIR}/vdn/run_qmix_simple_spread_v3.sh"
  "${ROOT_DIR}/qmix/run_vdn_simple_spread_v3.sh"
  "${ROOT_DIR}/vdn/run_vdn_simple_spread_v3.sh"
  "${ROOT_DIR}/vdn/run_vdn_simple_spread_v3.sh"
  "${ROOT_DIR}/vdn/run_vdn_simple_spread_v3.sh"
)


START_ALL=$(date +%s)
echo "============================================================"
echo "[Benchmark SUITE START] MPE / simple_spread_v3"
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
echo "[Benchmark SUITE END] MPE / simple_spread_v3"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapse: ${ELAPSED}s"
echo "============================================================"