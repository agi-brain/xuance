#!/usr/bin/env bash
set -euo pipefail

# Run all benchmark scripts under MuJoCo/Humanoid-v5.
# This script assumes the directory layout:
# benchmarks/MuJoCo/Humanoid-v5/{a2c, ddpg, pg, etc.}/run_*_Humanoid-v5.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# List the per-algorithm scripts you want to run (in order).
SCRIPTS=(
  "${ROOT_DIR}/a2c/run_a2c_Humanoid-v5.sh"
  "${ROOT_DIR}/ddpg/run_ddpg_Humanoid-v5.sh"
  "${ROOT_DIR}/pg/run_pg_Humanoid-v5.sh"
  "${ROOT_DIR}/ppo/run_ppo_Humanoid-v5.sh"
  "${ROOT_DIR}/sac/run_sac_Humanoid-v5.sh"
  "${ROOT_DIR}/td3/run_td3_Humanoid-v5.sh"
)


START_ALL=$(date +%s)
echo "============================================================"
echo "[Benchmark SUITE START] MuJoCo / Humanoid-v5"
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
echo "[Benchmark SUITE END] MuJoCo / Humanoid-v5"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapse: ${ELAPSED}s"
echo "============================================================"