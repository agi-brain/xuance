#!/usr/bin/env bash
set -euo pipefail

# Run all benchmark scripts under SMAC/8m_vs_9m.
# This script assumes the directory layout:
# benchmarks/SMAC/8m_vs_9m/{iql,qmix,vdn}/run_*_8m_vs_9m.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# List the per-algorithm scripts you want to run (in order).
SCRIPTS=(
  "${ROOT_DIR}/coma/run_coma_8m_vs_9m.sh"
  "${ROOT_DIR}/ippo/run_ippo_8m_vs_9m.sh"
  "${ROOT_DIR}/iql/run_iql_8m_vs_9m.sh"
  "${ROOT_DIR}/mappo/run_mappo_8m_vs_9m.sh"
  "${ROOT_DIR}/qmix/run_qmix_8m_vs_9m.sh"
  "${ROOT_DIR}/vdac/run_vdac_8m_vs_9m.sh"
  "${ROOT_DIR}/vdn/run_vdn_8m_vs_9m.sh"
  "${ROOT_DIR}/wqmix/run_wqmix_8m_vs_9m.sh"
)


START_ALL=$(date +%s)
echo "============================================================"
echo "[Benchmark SUITE START] SMAC / 8m_vs_9m"
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
echo "[Benchmark SUITE END] SMAC / 8m_vs_9m"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapse: ${ELAPSED}s"
echo "============================================================"