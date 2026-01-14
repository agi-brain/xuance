# XuanCe Benchmark

XuanCe provides standardized and reproducible benchmark scripts for evaluating deep reinforcement learning (DRL) and
multi-agent reinforcement learning (MARL) algorithms. Benchmarks are designed with the following principles:

- Clarity: one script corresponds to one algorithm-task benchmark
- Reproducibility: fixed evaluation protocol and multiple random seeds
- Comparability: consistent directory layout and result format
- Extensibility: easy to add new algorithms, environments, or suites

## Table of Contents

- [Directory Structure](#directory-structure)
- [Running a Single Benchmark](#running-a-single-benchmark)
- [Running a Benchmark Suite](#running-a-benchmark-suite)
- [Evaluation Protocol](#evaluation-protocol)
- [Benchmark Results](#benchmark-results)
- [Reproducibility](#reproducibility)
- [How to Add a New Benchmark](#how-to-add-a-new-benchmark)

# Directory Structure

Benchmarks are organized by environment->scenario->algorithm:
```text
xuance-benchmarks/
├── MuJoCo/
│ └── Ant-v5/
│   ├── a2c/
│   │ ├── a2c_Ant-v5.yaml
│   │ └── run_a2c_Ant-v5.sh
│   ├── ddpg/
│   │ ├── ddpg_Ant-v5.yaml
│   │ └── run_ddpg_Ant-v5.sh
│   ├── ppo/
│   │ ├── ppo_Ant-v5.yaml
│   │ └── run_ppo_Ant-v5.sh
│   └── run_Ant-v5_all.sh
│── ...
│── benchmark.py
```

- Each algorithm-specific script (run_*.sh) defines an atomic benchmark.
- Suite scripts (e.g. run_simple_spread_all.sh) run multiple algorithms sequentially on the same task.

# Running a Single Benchmark

Each benchmark script runs the same task with multiple random seeds (default: 5).

Example: run PPO on MuJoCo Ant-v5
```bash
bash MuJoCo/Ant-v5/ppo/run_ppo_Ant-v5.sh
```

During execution, XuanCe prints algorith, environment, and evaluation information, while the benchmark script prints
clear START / END boundaries for each seed.

# Running a Benchmark Suite

To evaluate all supported algorithms on a given task, use the suite script:
```bash
bash benchmarks/MuJoCo/Ant-v5/run_Ant-v5_all.sh
```

This will sequentially run Algorithm_1, Algorithm_2, ..., Algorithm_N on the same environment with identical evaluation
settings.

# Evaluation Protocol

All benchmarks follow a unified evaluation protocol:
- Multiple independent runs with different random seeds
- Periodic evaluation during training (≈ every 1% of total steps)
- Each evaluation consists of multiple test episodes
- Reported performance is the mean episode return
- Final benchmark scores are aggregated across seeds

This design ensures fair comparison and robust performance estimation.

# Benchmark Results

Benchmark results are stored in a structured directory layout:
```text
(To be stored)
```

- Each learning_curve.csv contains the learning curve for one seed
- Aggregated results (mean ± std across seeds) can be generated using analysis scripts

    TensorBoard logs are used for visualization and debugging, while CSV files are treated as the official benchmark artifacts.

# Reproducibility

To ensure reproducibility, benchmark scripts explicitly specify:
- Algorithm name
- Environment and scenario ID
- Random seed
- Training and evaluation settings

Benchmark scripts are the source of truth for all reported results.

# How to Add a New Benchmark

This section describes how to add a new benchmark to XuanCe.
A benchmark in XuanCe is defined by one algorithm, one environment scenario, and multiple random seeds.

## Step 1: Choose the Benchmark Task

Determine the target environment and scenario.
For example:
- Environment: Atari
- Scenario: Breakout-v5

Create the corresponding directory if it does not exist:
```text
benchmarks/Atari/Breakout-v5/
```

## Step 2: Create an Algorithm-Specific Directory

Under the scenario directory, create a subdirectory for the algorithm:
```text
benchmarks/Atari/Breakout-v5/<algorithm>/
```

For example, for PPO:
```text
benchmarks/Atari/Breakout-v5/ppo/
```

## Step 3: Prepare the Algorithm Configuration (Optional)

If the algorithm requires a specific configuration file, place it in the algorithm directory:
```text
benchmarks/Atari/Breakout-v5/ppo/ppo_atari.yaml
```

This configuration file defines hyperparameters and environment-specific settings used by the benchmark.

## Step 4: Write the Benchmark Script

Create a benchmark script named:
```text
run_<algorithm>_<scenario>.sh
```

For example:
```text
run_ppo_Breakout-v5.sh
```
Each benchmark script should:
- Call the shared benchmark.py script
- Run multiple random seeds (default: 5)
- Clearly indicate the start and end of each seed
- Not duplicate algorithm or environment information already printed by XuanCe

Example structure:
```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON=python

ALGO="ppo"
ENV="Atari"
ENV_ID="Breakout-v5"

OUT_ROOT="${PROJECT_ROOT}/benchmarks/results/raw/${ENV}/${ENV_ID}/${ALGO}"

for SEED in 1 2 3 4 5; do
  WORKDIR="${OUT_ROOT}/seed_${SEED}"
  mkdir -p "${WORKDIR}"

  echo "========== [Benchmark START] seed=${SEED} =========="
  START_TIME=$(date +%s)

  if ${PYTHON} "${PROJECT_ROOT}/benchmark.py" \
      --algo "${ALGO}" \
      --env "${ENV}" \
      --env-id "${ENV_ID}" \
      --seed "${SEED}" \
      --workdir "${WORKDIR}"; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED"
  fi

  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))

  echo "========== [Benchmark END] seed=${SEED} | status=${STATUS} | time=${ELAPSED}s =========="
done
```

## Step 5: (Optional) Add the Benchmark to a Suite Script

If you want the new benchmark to be included in a benchmark suite, edit the suite script under the scenario directory:
```text
benchmarks/Atai/Breakout-v5/run_simple_spread_all.sh
```

Add the new benchmark script to the list:
```bash
SCRIPTS=(
  "${ROOT_DIR}/dqn/run_dqn_Breakout-v5.sh"
  "${ROOT_DIR}/ppo/run_ppo_Breakout-v5.sh"
  "${ROOT_DIR}/<new_algo>/run_<new_algo>_Breakout-v5.sh"
)
```

## Step 6: Run and Verify

Run the benchmark script:
```text
bash benchmarks/Atari/Breakout-v5/iql/run_ppo_Breakout-v5.sh
```

Verify that:
- All seeds run sequentially
- Each seed prints clear START / END markers
- Results are saved under the correct directory structure
- The benchmark can be reproduced by re-running the script

## Design Principles

When adding a new benchmark, please follow these principles:
- One script = one benchmark
- Benchmark scripts are the source of truth
- Do not hard-code absolute paths
- Do not duplicate logging already handled by XuanCe
- Prefer clarity and reproducibility over convenience
