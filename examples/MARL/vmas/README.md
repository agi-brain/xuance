# XuanCe × VMAS: Vectorized Multi-Agent Physics Simulation

[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) (Vectorized Multi-Agent Simulator) is a PyTorch-native, GPU-vectorized 2D physics simulator designed for multi-robot coordination and Multi-Agent Reinforcement Learning (MARL).

XuanCe integrates VMAS through [`NativeVecMultiAgentEnv`](../../../xuance/environment/vector_envs/native/native_vec_maenv.py) and [`VMASVecMultiAgentEnv`](../../../xuance/environment/multi_agent_env/vmas.py), stepping all parallel environments inside a single batched PyTorch VMAS instance (`cpu` or `cuda`) without multiprocessing IPC overhead.

<table align="center">
  <tr>
    <td align="center">
      <img src="../../../docs/source/_static/figures/vmas/balance.gif" width="175" alt="VMAS Balance"/><br/>
      <sub><b>balance</b> — Cooperative Beam Balancing</sub>
    </td>
    <td align="center">
      <img src="../../../docs/source/_static/figures/vmas/give_way.gif" width="175" alt="VMAS Give Way"/><br/>
      <sub><b>give_way</b> — Corridor Yielding</sub>
    </td>
    <td align="center">
      <img src="../../../docs/source/_static/figures/vmas/wheel.gif" width="175" alt="VMAS Wheel"/><br/>
      <sub><b>wheel</b> — Coordinated Wheel Spinning</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../../../docs/source/_static/figures/vmas/navigation.gif" width="175" alt="VMAS Navigation"/><br/>
      <sub><b>navigation</b> — Collision-Free Navigation</sub>
    </td>
    <td align="center">
      <img src="../../../docs/source/_static/figures/vmas/transport.gif" width="175" alt="VMAS Transport"/><br/>
      <sub><b>transport</b> — Cooperative Box Transport</sub>
    </td>
    <td align="center">
      <sub><i>Rendered via XuanCe <code>vmas_demo.py --save-gif</code><br/>(MAPPO, 32 batched envs)</i></sub>
    </td>
  </tr>
</table>

---

## Why XuanCe × VMAS?

- **Native Batched Vectorization (`NativeVecMultiAgentEnv`)**: Runs `parallels` sub-environments in one VMAS PyTorch tensor batch instead of spawning Python worker processes.
- **CPU & CUDA Physics Simulation**: Switch simulation physics between `--vmas-device cpu` and `--vmas-device cuda` without changing your training code.
- **Drop-in MARL Compatibility**:
  - Normalizes VMAS `gym.spaces` objects to `gymnasium.spaces` (`Box` / `Discrete`).
  - Separates episode `terminated` vs time-limit `truncated` (supporting VMAS `terminated_truncated=True` as well as 4-tuple fallback) and handles per-slot `reset_at(e)` partial resets.
  - Exposes concatenated global observations as `state()` (for centralized critics / value-decomposition methods like `QMIX`).
  - Preserves `DummyVecMultiAgentEnv` / `SubprocVecMultiAgentEnv` fallback via [`VMAS_Env`](../../../xuance/environment/multi_agent_env/vmas.py) and works across XuanCe's PyTorch, TensorFlow, and MindSpore backends.

---

## Installation

Install XuanCe with the optional `vmas` extra (or install `vmas` inside an existing XuanCe environment):

```bash
pip install "xuance[vmas]"
# or, in an existing XuanCe environment:
pip install vmas
```

> [!NOTE]
> On headless Linux servers (e.g., SSH/Docker without an X display), wrap commands that enable `render=True` or `--save-gif` with `xvfb-run -a`:
> ```bash
> xvfb-run -a python examples/MARL/vmas/vmas_demo.py --algo mappo --scenario balance --save-gif results/balance.gif
> ```

---

## Supported Algorithms & Built-in Configs

All six algorithms ship with scenario-agnostic configs (`env_id: null`), so any VMAS scenario name can be passed at runtime:

| Algorithm (`--algo`) | Control Mode | Action Output Activation | Vectorizer | Config File |
| :--- | :--- | :--- | :--- | :--- |
| `mappo` | Continuous | `tanh` (`[-1, 1]`) | `NativeVecMultiAgentEnv` | [`xuance/configs/mappo/vmas.yaml`](../../../xuance/configs/mappo/vmas.yaml) |
| `ippo` | Continuous | `tanh` (`[-1, 1]`) | `NativeVecMultiAgentEnv` | [`xuance/configs/ippo/vmas.yaml`](../../../xuance/configs/ippo/vmas.yaml) |
| `masac` | Continuous | `tanh` (`[-1, 1]`) | `NativeVecMultiAgentEnv` | [`xuance/configs/masac/vmas.yaml`](../../../xuance/configs/masac/vmas.yaml) |
| `maddpg` | Continuous | `tanh` (`[-1, 1]`) | `NativeVecMultiAgentEnv` | [`xuance/configs/maddpg/vmas.yaml`](../../../xuance/configs/maddpg/vmas.yaml) |
| `matd3` | Continuous | `tanh` (`[-1, 1]`) | `NativeVecMultiAgentEnv` | [`xuance/configs/matd3/vmas.yaml`](../../../xuance/configs/matd3/vmas.yaml) |
| `qmix` | Discrete | Categorical Q-values | `NativeVecMultiAgentEnv` | [`xuance/configs/qmix/vmas.yaml`](../../../xuance/configs/qmix/vmas.yaml) |

---

## Flagship Multi-Robot Scenarios

Any scenario listed in `vmas.scenarios` works directly as `--scenario <name>` (or `env_id="<name>"` in `get_runner`). Popular multi-robot benchmarks include:

| Scenario (`env_id`) | Task Description | Default Agents | Control Mode | Suggested Algo |
| :--- | :--- | :---: | :--- | :--- |
| `balance` | Robots jointly lift and balance a beam carrying a sphere to a target height | 3 | Continuous | `mappo`, `matd3` |
| `give_way` | Two robots coordinate passing through a narrow corridor from opposite sides | 2 | Continuous | `mappo`, `ippo` |
| `wheel` | Robots apply coordinated tangential forces to spin a central wheel at target velocity | 4 | Continuous | `mappo`, `maddpg` |
| `navigation` | Decentralized collision-free point-to-point navigation with lidar sensing | 4 | Continuous / Discrete | `mappo`, `masac`, `qmix` |
| `transport` | Four robots cooperatively push a heavy box past obstacles to a goal zone | 4 | Continuous | `mappo`, `masac` |
| `flocking` | Decentralized swarm cohesion, velocity alignment, and obstacle avoidance | 4 | Continuous | `mappo`, `masac` |
| `dispersion` | Agents spread out to cover dispersed food/landmarks efficiently | 4 | Continuous / Discrete | `mappo`, `qmix` |
| `football` | Cooperative multi-agent 2D robot soccer team control | 3 | Continuous | `mappo`, `masac` |

---

## Quick Start CLI (`vmas_demo.py`)

Run from the repository root (`xuance/`):

```bash
# 1. Cooperative beam balancing with MAPPO (32 batched envs) and GIF export
python examples/MARL/vmas/vmas_demo.py --algo mappo --scenario balance --running-steps 200000 --save-gif results/vmas_balance.gif

# 2. Corridor yielding with MAPPO
python examples/MARL/vmas/vmas_demo.py --algo mappo --scenario give_way --running-steps 200000 --save-gif results/vmas_give_way.gif

# 3. Continuous multi-robot navigation with MASAC
python examples/MARL/vmas/vmas_demo.py --algo masac --scenario navigation --parallels 32 --running-steps 300000

# 4. Discrete action control with QMIX
python examples/MARL/vmas/vmas_demo.py --algo qmix --scenario navigation --parallels 16 --running-steps 200000

# 5. Custom agent count + high-throughput GPU physics (--vmas-device cuda)
python examples/MARL/vmas/vmas_demo.py --algo mappo --scenario flocking --n-agents 6 --vmas-device cuda --parallels 256 --running-steps 1000000
```

### Demo CLI Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--algo` | `mappo` | MARL algorithm (`mappo`, `ippo`, `masac`, `maddpg`, `matd3`, `qmix`) |
| `--scenario` | `transport` | VMAS scenario name (any scenario in `vmas.scenarios`) |
| `--mode` | `benchmark` | Runner mode (`train`, `test`, `benchmark`) |
| `--device` | `cuda:0` (if available) else `cpu` | Policy/learner training device |
| `--vmas-device` | `cpu` | VMAS physics simulation device (`cpu` or `cuda`) |
| `--parallels` | `32` | Number of batched parallel VMAS sub-environments |
| `--buffer-size` | YAML default (`3200` on-policy / `100000` off-policy) | Optional rollout/replay buffer override |
| `--running-steps` | `200000` | Total environment steps to train |
| `--eval-interval` | `50000` | Evaluation frequency during `benchmark` |
| `--max-episode-steps` | `200` | Episode horizon (`max_steps` in VMAS) |
| `--test-episode` | `5` | Number of episodes per evaluation |
| `--seed` | `1` | Random seed (`seed` & `env_seed`) |
| `--n-agents` | `None` | Optional agent count passed via `scenario_kwargs={"n_agents": ...}` |
| `--save-gif` | `None` | Output path for a deterministic evaluation rollout GIF |

---

## Python API & Configuration Options

### 1. Standard Training via `xuance.get_runner`

```python
from argparse import Namespace
from xuance import get_runner

args = Namespace(
    device="cuda:0",
    vmas_device="cpu",
    parallels=32,
    max_episode_steps=200,
    scenario_kwargs={"n_agents": 6},  # forwarded directly to vmas.make_env(...)
    running_steps=300000,
    eval_interval=50000,
    render=False,
)
runner = get_runner(algo="mappo", env="vmas", env_id="flocking", parser_args=args)
runner.run(mode="benchmark")
```

### 2. Custom VMAS `BaseScenario` via `make_envs`

`get_runner(..., env_id=...)` expects a string `env_id` for logging and result directory paths. If you author your own `vmas.simulator.scenario.BaseScenario` subclass, instantiate the environment directly with `make_envs`:

```python
from argparse import Namespace
from xuance.common import get_arguments
from xuance.environment import make_envs
from xuance.torch.agents import MAPPO_Agents
from my_scenario import CustomRobotScenario

config = get_arguments(
    algo="mappo",
    env="vmas",
    env_id="custom_robot_scenario",
    parser_args=Namespace(parallels=32, max_episode_steps=200),
)
config.env_id = CustomRobotScenario()  # pass BaseScenario instance to vmas.make_env
envs = make_envs(config)
agents = MAPPO_Agents(config=config, envs=envs)
agents.train(config.running_steps // config.parallels)
agents.finish()
envs.close()
```

### 3. Key Configuration Parameters

| Key | Default | Notes |
| :--- | :--- | :--- |
| `vectorize` | `"NativeVecMultiAgentEnv"` | Recommended batched path; can also fall back to `"DummyVecMultiAgentEnv"` or `"SubprocVecMultiAgentEnv"` |
| `vmas_device` | `"cpu"` | `"cpu"` is typically faster for small batch sizes (`parallels <= 32`); switch to `"cuda"` for large-scale parallelism (`>= 128`) |
| `max_episode_steps` | `200` | Forwarded to `vmas.make_env(max_steps=...)` so episodes terminate cleanly at a finite horizon |
| `continuous_action` | `True` (`False` for `qmix`) | Controls continuous `Box` vs discrete `Discrete` action spaces |
| `activation_action` | `"tanh"` (continuous) | Matches VMAS's symmetric `[-1, 1]` action bounds |
| `scenario_kwargs` | `{}` | Dictionary of scenario-specific parameters unpacked into `vmas.make_env(...)` |
| `render` | `False` | Kept `False` in default configs for fast headless training/benchmarking |

---

## Troubleshooting & Notes

- **CPU vs CUDA for `vmas_device`**: For `parallels <= 32`, `vmas_device="cpu"` often achieves higher wall-clock throughput than `"cuda"` because small per-step tensor transfers dominate GPU kernel gains. Use `vmas_device="cuda"` when scaling to `128+` parallel environments.
- **`MultiDiscrete` Action Spaces**: XuanCe's built-in MARL actors currently support `Box` and single `Discrete` action spaces (`multidiscrete_actions=False` in VMAS flattens discrete choices into a single `Discrete` space). Scenarios configured with `MultiDiscrete` action spaces raise an explicit `NotImplementedError`.
- **Verification**: Contract unit tests (zero external dependency) and real VMAS integration tests live in [`tests/torch/test_vmas.py`](../../../tests/torch/test_vmas.py).

---

## References

- **VMAS**: Bettini, M., Kortvelesy, R., Blumenkamp, J., & Prorok, A. *VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning*. DARS 2022. ([GitHub](https://github.com/proroklab/VectorizedMultiAgentSimulator))
- **XuanCe**: Liu, W., et al. *XuanCe: A Comprehensive and Unified Deep Reinforcement Learning Library*. arXiv:2312.16248, 2023. ([GitHub](https://github.com/agi-brain/xuance))
