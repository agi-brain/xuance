#!/usr/bin/env bash
set -euo pipefail

# Run all benchmark scripts under ClassicControl/MountainCar-v0.
# This script assumes the directory layout:
# benchmarks/ClassicControl/MountainCar-v0/{a2c, dqn, ppo, etc.}/run_*_MountainCar-v0.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# List the per-algorithm scripts you want to run (in order).
SCRIPTS=(
  "${ROOT_DIR}/a2c/run_a2c_MountainCar-v0.sh"
  "${ROOT_DIR}/c51/run_c51_MountainCar-v0.sh"
  "${ROOT_DIR}/double_dqn/run_ddqn_MountainCar-v0.sh"
  "${ROOT_DIR}/dqn/run_dqn_MountainCar-v0.sh"
  "${ROOT_DIR}/drqn/run_drqn_MountainCar-v0.sh"
  "${ROOT_DIR}/dueling_dqn/run_dueldqn_MountainCar-v0.sh"
  "${ROOT_DIR}/noisy_dqn/run_noisydqn_MountainCar-v0.sh"
  "${ROOT_DIR}/perdqn/run_perdqn_MountainCar-v0.sh"
  "${ROOT_DIR}/pg/run_pg_MountainCar-v0.sh"
  "${ROOT_DIR}/ppg/run_ppg_MountainCar-v0.sh"
  "${ROOT_DIR}/ppo/run_ppo_MountainCar-v0.sh"
  "${ROOT_DIR}/qrdqn/run_qrdqn_MountainCar-v0.sh"
  "${ROOT_DIR}/sac/run_sac_MountainCar-v0.sh"
)


START_ALL=$(date +%s)
echo "============================================================"
echo "[Benchmark SUITE START] ClassicControl / MountainCar-v0"
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
echo "[Benchmark SUITE END] ClassicControl / MountainCar-v0"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapse: ${ELAPSED}s"
echo "============================================================"