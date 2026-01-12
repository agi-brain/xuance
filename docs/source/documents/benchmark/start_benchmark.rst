Benchmark
======================================

XuanCe provides standardized and reproducible benchmark scripts for evaluating deep reinforcement learning (DRL) and
multi-agent reinforcement learning (MARL) algorithms. Benchmarks are designed with the following principles:

- Clarity: one script corresponds to one algorithm-task benchmark
- Reproducibility: fixed evaluation protocol and multiple random seeds
- Comparability: consistent directory layout and result format
- Extensibility: easy to add new algorithms, environments, or suites

Directory Structure
---------------------------------------

Benchmarks are organized by environment->scenario->algorithm:

.. code-block:: text

    benchmarks/
    ├── MPE/
    │ └── simple_spread_v3/
    │ ├── iql/
    │ │ ├── iql_simple_spread_v3.yaml
    │ │ └── run_iql_simple_spread_v3.sh
    │ ├── qmix/
    │ │ ├── qmix.yaml
    │ │ └── run_qmix_simple_spread_v3.sh
    │ ├── vdn/
    │ │ ├── vdn.yaml
    │ │ └── run_vdn_simple_spread_v3.sh
    │ └── run_simple_spread_all.sh


- Each algorithm-specific script (run_*.sh) defines an atomic benchmark.
- Suite scripts (e.g. run_simple_spread_all.sh) run multiple algorithms sequentially on the same task.

Running a Single Benchmark
---------------------------------------

Each benchmark script runs the same task with multiple random seeds (default: 5).

Example: run MADDPG on MPE simple_spread_v3

.. code-block:: bash

    bash benchmarks/MPE/simple_spread_v3/iql/run_iql_simple_spread_v3.sh


During execution, XuanCe prints algorith, environment, and evaluation information, while the benchmark script prints
clear START / END boundaries for each seed.

Running a Benchmark Suite
---------------------------------------

To evaluate all supported algorithms on a given task, use the suite script:

.. code-block:: bash

    bash benchmarks/MPE/simple_spread_v3/run_simple_spread_all.sh

This will sequentially run Algorithm_1, Algorithm_2, ..., Algorithm_N on the same environment with identical evaluation
settings.

Evaluation Protocol
---------------------------------------

All benchmarks follow a unified evaluation protocol:

- Multiple independent runs with different random seeds
- Periodic evaluation during training (≈ every 1% of total steps)
- Each evaluation consists of multiple test episodes
- Reported performance is the mean episode return
- Final benchmark scores are aggregated across seeds

This design ensures fair comparison and robust performance estimation.

Benchmark Results
---------------------------------------

Benchmark results are stored in a structured directory layout:

.. code-block:: text

    (To be stored)

- Each learning_curve.csv contains the learning curve for one seed
- Aggregated results (mean ± std across seeds) can be generated using analysis scripts

    TensorBoard logs are used for visualization and debugging, while CSV files are treated as the official benchmark artifacts.

Reproducibility
---------------------------------------

To ensure reproducibility, benchmark scripts explicitly specify:

- Algorithm name
- Environment and scenario ID
- Random seed
- Training and evaluation settings

Benchmark scripts are the source of truth for all reported results.