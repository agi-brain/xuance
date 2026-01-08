How to Add a New Benchmark
======================================

This section describes how to add a new benchmark to XuanCe.
A benchmark in XuanCe is defined by one algorithm, one environment scenario, and multiple random seeds.

Step 1: Choose the Benchmark Task
---------------------------------------

Determine the target environment and scenario.
For example:
- Environment: Atari
- Scenario: Breakout-v5

Create the corresponding directory if it does not exist:

.. code-block::text

    benchmarks/Atari/Breakout-v5/

Step 2: Create an Algorithm-Specific Directory
---------------------------------------

Under the scenario directory, create a subdirectory for the algorithm:

.. code-block::text

     benchmarks/Atari/Breakout-v5/<algorithm>/


For example, for PPO:

.. code-block::text

    benchmarks/Atari/Breakout-v5/ppo/


Step 3: Prepare the Algorithm Configuration (Optional)
---------------------------------------

If the algorithm requires a specific configuration file, place it in the algorithm directory:

.. code-block::

    benchmarks/Atari/Breakout-v5/ppo/ppo_atari.yaml


This configuration file defines hyperparameters and environment-specific settings used by the benchmark.

Step 4: Write the Benchmark Script
---------------------------------------

Create a benchmark script named:

.. code-block::text

    run_<algorithm>_<scenario>.sh

For example:

.. code-block::text

    run_iql_simple_spread_v3.sh

Each benchmark script should:
- Call the shared train.py script
- Run multiple random seeds (default: 5)
- Clearly indicate the start and end of each seed
- Not duplicate algorithm or environment information already printed by XuanCe

Example structure:

.. code-block::bash

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

      if ${PYTHON} "${PROJECT_ROOT}/train.py" \
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

Step 5: (Optional) Add the Benchmark to a Suite Script
---------------------------------------

If you want the new benchmark to be included in a benchmark suite, edit the suite script under the scenario directory:

.. code-block::text

    benchmarks/Atai/Breakout-v5/run_simple_spread_all.sh


Add the new benchmark script to the list:

.. code-block::bash

    SCRIPTS=(
      "${ROOT_DIR}/dqn/run_dqn_Breakout-v5.sh"
      "${ROOT_DIR}/ppo/run_ppo_Breakout-v5.sh"
      "${ROOT_DIR}/<new_algo>/run_<new_algo>_Breakout-v5.sh"
    )

Step 6: Run and Verify
---------------------------------------

Run the benchmark script:

.. code-block::bash

    bash benchmarks/Atari/Breakout-v5/iql/run_ppo_Breakout-v5.sh


Verify that:
- All seeds run sequentially
- Each seed prints clear START / END markers
- Results are saved under the correct directory structure
- The benchmark can be reproduced by re-running the script

Design Principles
---------------------------------------

When adding a new benchmark, please follow these principles:

- One script = one benchmark
- Benchmark scripts are the source of truth
- Do not hard-code absolute paths
- Do not duplicate logging already handled by XuanCe
- Prefer clarity and reproducibility over convenience