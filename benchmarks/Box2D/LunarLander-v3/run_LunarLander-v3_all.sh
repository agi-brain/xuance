#!/usr/bin/env bash
set -euo pipefail

# Run all benchmark scripts under Box2D/LunarLander-v3.
# This script assumes the directory layout:
# benchmarks/Box2D/LunarLander-v3/{iql,qmix,vdn}/run_*_LunarLander-v3.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# List the per-algorithm scripts you want to run (in order).
SCRIPTS=(
  "${ROOT_DIR}/a2c/run_a2c_LunarLander-v3.sh"
  "${ROOT_DIR}/c51/run_c51_LunarLander-v3.sh"
  "${ROOT_DIR}/double_dqn/run_ddqn_LunarLander-v3.sh"
  "${ROOT_DIR}/dqn/run_dqn_LunarLander-v3.sh"
  "${ROOT_DIR}/drqn/run_drqn_LunarLander-v3.sh"
  "${ROOT_DIR}/dueling_dqn/run_dueldqn_LunarLander-v3.sh"
  "${ROOT_DIR}/noisy_dqn/run_noisydqn_LunarLander-v3.sh"
  "${ROOT_DIR}/perdqn/run_perdqn_LunarLander-v3.sh"
  "${ROOT_DIR}/pg/run_pg_LunarLander-v3.sh"
  "${ROOT_DIR}/ppg/run_ppg_LunarLander-v3.sh"
  "${ROOT_DIR}/ppo/run_ppo_LunarLander-v3.sh"
  "${ROOT_DIR}/qrdqn/run_qrdqn_LunarLander-v3.sh"
  "${ROOT_DIR}/sac/run_sac_LunarLander-v3.sh"
)


START_ALL=$(date +%s)
echo "============================================================"
echo "[Benchmark SUITE START] Box2D / LunarLander-v3"
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
echo "[Benchmark SUITE END] Box2D / LunarLander-v3"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapse: ${ELAPSED}s"
echo "============================================================"