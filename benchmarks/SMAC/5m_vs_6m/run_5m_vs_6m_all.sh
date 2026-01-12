#!/usr/bin/env bash
set -euo pipefail

# Run all benchmark scripts under SMAC/5m_vs_6m.
# This script assumes the directory layout:
# benchmarks/SMAC/5m_vs_6m/{iql,qmix,vdn}/run_*_5m_vs_6m.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# List the per-algorithm scripts you want to run (in order).
SCRIPTS=(
  "${ROOT_DIR}/coma/run_coma_5m_vs_6m.sh"
  "${ROOT_DIR}/ippo/run_ippo_5m_vs_6m.sh"
  "${ROOT_DIR}/iql/run_iql_5m_vs_6m.sh"
  "${ROOT_DIR}/mappo/run_mappo_5m_vs_6m.sh"
  "${ROOT_DIR}/qmix/run_qmix_5m_vs_6m.sh"
  "${ROOT_DIR}/vdac/run_vdac_5m_vs_6m.sh"
  "${ROOT_DIR}/vdn/run_vdn_5m_vs_6m.sh"
  "${ROOT_DIR}/wqmix/run_wqmix_5m_vs_6m.sh"
)


START_ALL=$(date +%s)
echo "============================================================"
echo "[Benchmark SUITE START] SMAC / 5m_vs_6m"
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
echo "[Benchmark SUITE END] SMAC / 5m_vs_6m"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapse: ${ELAPSED}s"
echo "============================================================"