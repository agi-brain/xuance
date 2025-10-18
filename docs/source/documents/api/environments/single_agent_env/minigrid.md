# Minigrid

The MiniGrid environment is a lightweight, grid-based environment designed for research in DRL. 
It is highly customizable, supporting a variety of tasks and challenges for training agents 
with partial observability, sparse rewards, and symbolic inputs. 
Its simplicity and flexibility make it an ideal testbed for developing and benchmarking DRL algorithms.

- **Official documentation:** [**https://minigrid.farama.org/**](https://minigrid.farama.org/).
- **Github repository:** [**https://github.com/Farama-Foundation/Minigrid**](https://github.com/Farama-Foundation/Minigrid).

## Overview

There are 23 predefined Minigrid environments and 40 BabyAI environments for DRL research. 
Each environment is programmable to determine the difficulty, 
and is suite for research topics like sparse reward learning, continuous learning, etc.

### Minigrid Environments

```{raw} html
    :file: lists/minigrid_list.html
```

### BabyAI Environments

```{raw} html
    :file: lists/babyai_list.html
```

## Installation

The MiniGrid environment is not included with the installation of XuanCe. 
As an external package, it needs to be installed separately.

### From PyPI

Minigrid could be installed via ``pip``:

```{code-block} bash
pip install minigrid
```

### From GitHub

It can also be installed from GitHub:

```{code-block} bash
git clone https://github.com/Farama-Foundation/Minigrid.git
cd Minigrid
pip install -e .
```

## Usage With XuanCe

### Make Environments

Before running MiniGrid environment with XuanCe, 
you need to specify some necessary arguments, for example:

```{code-block} yaml
env_name: "MiniGrid"  # The environment name.
env_id: "MiniGrid-Empty-5x5-v0"  # The environment id.
env_seed: 1  # The random seed for the first environment.
render: True  # Whether to render the environment for visualization.
render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
RGBImgPartialObsWrapper: False  # Whether to use RGBImgPartialObsWrapper.
ImgObsWrapper: False  # Whether to use ImgObsWrapper.
vectorize: "DummyVecEnv"  # Choose the method to vectorize the environment.
parallels: 10  # The number of environments that run in parallel.
```

Then, we can make the MiniGrid environment in XuanCe with the following code:

```{code-block} python3
from xuance import make_envs
from argparse import Namespace

configs = Namespace(env_name="MiniGrid",
                    env_id="MiniGrid-Empty-5x5-v0",
                    env_seed=1,
                    render=True,
                    render_mode="human",
                    RGBImgPartialObsWrapper=False,
                    ImgObsWrapper=False,
                    vectorize="DummyVecEnv",
                    parallels=10)
envs = make_envs(configs)
envs.reset()
while True:
    actions = [envs.action_space.sample() for _ in range(configs.parallels)]  # select random actions for each environment.
    observation, reward, terminated, truncated, info = envs.step(actions)  # execute actions.
    if terminated.any() or truncated.any():
        break
envs.close()
```

### Run With An Agent

Take ``DQN`` as an example, you need to first prepare a config file like "dqn_minigrid.yaml":

```{code-block} yaml
dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorlayer"
project_name: "XuanCe_Benchmark"
logger: "tensorboard"  # Choices: tensorboard, wandb.
wandb_user_name: "your_user_name"
render: True
render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
fps: 50
test_mode: False
device: "cuda:0"  # Choose an calculating device. PyTorch: "cpu", "cuda:0"; TensorFlow: "cpu"/"CPU", "gpu"/"GPU"; MindSpore: "CPU", "GPU", "Ascend", "Davinci".
distributed_training: False  # Whether to use multi-GPU for distributed training.
master_port: '12355'  # The master port for current experiment when use distributed training.

agent: "DQN"
env_name: "MiniGrid"
env_id: "MiniGrid-Empty-5x5-v0"
env_seed: 1
RGBImgPartialObsWrapper: False
ImgObsWrapper: False
vectorize: "DummyVecEnv"
learner: "DQN_Learner"
policy: "Basic_Q_network"
representation: "Basic_MLP"
runner: "DRL"

representation_hidden_size: [64,]
q_hidden_size: [64,]
activation: 'relu'

seed: 1
parallels: 10
buffer_size: 10000
batch_size: 256
learning_rate: 0.001
gamma: 0.99

start_greedy: 0.5
end_greedy: 0.01
decay_step_greedy: 200000
sync_frequency: 50
training_frequency: 1
running_steps: 200000
start_training: 1000

use_grad_clip: False  # gradient normalization
grad_clip_norm: 0.5
use_actions_mask: False
use_obsnorm: False
use_rewnorm: False
obsnorm_range: 5
rewnorm_range: 5

test_steps: 10000
eval_interval: 20000
test_episode: 1
log_dir: "./logs/dqn/"
model_dir: "./models/dqn/"
```

In this file, apart from the parameters related to the environment mentioned above, 
the rest are all related to creating the ``DQN_Agent`` object.

After that, we can create the ``DQN_Agent`` by typing and run the following code:

```{code-block} python3
from argparse import Namespace
from xuance import make_envs
from xuance.torch.agents import DQN_Agent
from xuance.common import get_configs

configs_dict = get_configs(file_dir="dqn_minigrid.yaml")
configs = Namespace(**configs_dict)
envs = make_envs(configs)
agent = DQN_Agent(configs, envs)
agent.train(10000)
agent.save_model(model_name="final_model.pth")
agent.finish()
envs.close()
```

## Citations

**Minigrid:**

```{code-block} bash
@inproceedings{MinigridMiniworld23,
  author       = {Maxime Chevalier{-}Boisvert and Bolun Dai and Mark Towers and Rodrigo Perez{-}Vicente and Lucas Willems and Salem Lahlou and Suman Pal and Pablo Samuel Castro and Jordan Terry},
  title        = {Minigrid {\&} Miniworld: Modular {\&} Customizable Reinforcement Learning Environments for Goal-Oriented Tasks},
  booktitle    = {Advances in Neural Information Processing Systems 36, New Orleans, LA, USA},
  month        = {December},
  year         = {2023},
}
```

**BabyAI:**

```{code-block} bash
@article{chevalier2018babyai,
  title={Babyai: A platform to study the sample efficiency of grounded language learning},
  author={Chevalier-Boisvert, Maxime and Bahdanau, Dzmitry and Lahlou, Salem and Willems, Lucas and Saharia, Chitwan and Nguyen, Thien Huu and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1810.08272},
  year={2018}
}
```

## APIs

```{eval-rst}
.. automodule:: xuance.environment.single_agent_env.minigrid
    :members:
    :undoc-members:
    :show-inheritance:
```
