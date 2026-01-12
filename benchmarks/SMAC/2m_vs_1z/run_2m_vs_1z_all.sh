#!/usr/bin/env bash
set -euo pipefail

# Run all benchmark scripts under SMAC/2m_vs_1z.
# This script assumes the directory layout:
# benchmarks/SMAC/2m_vs_1z/{iql,qmix,vdn}/run_*_2m_vs_1z.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# List the per-algorithm scripts you want to run (in order).
SCRIPTS=(
  "${ROOT_DIR}/coma/run_coma_2m_vs_1z.sh"
  "${ROOT_DIR}/ippo/run_ippo_2m_vs_1z.sh"
  "${ROOT_DIR}/iql/run_iql_2m_vs_1z.sh"
  "${ROOT_DIR}/mappo/run_mappo_2m_vs_1z.sh"
  "${ROOT_DIR}/qmix/run_qmix_2m_vs_1z.sh"
  "${ROOT_DIR}/vdac/run_vdac_2m_vs_1z.sh"
  "${ROOT_DIR}/vdn/run_vdn_2m_vs_1z.sh"
  "${ROOT_DIR}/wqmix/run_wqmix_2m_vs_1z.sh"
)


START_ALL=$(date +%s)
echo "============================================================"
echo "[Benchmark SUITE START] SMAC / 2m_vs_1z"
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
echo "[Benchmark SUITE END] SMAC / 2m_vs_1z"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapse: ${ELAPSED}s"
echo "============================================================"