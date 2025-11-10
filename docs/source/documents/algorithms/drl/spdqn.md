# Split Parameterized Deep Q-Network (SP-DQN)

Split Parameterized Deep Q-Network (SP-DQN) is an extension of the traditional Deep Q-Network (DQN) designed to improve the efficiency and scalability of the Q-learning algorithm in large-scale problems. SP-DQN splits the Q-network into multiple parameterized parts, each corresponding to a different action space. By decoupling these parts, SP-DQN can effectively reduce the training complexity and memory requirements while maintaining high performance.

This table lists some general features about SP-DQN algorithm:

| Features of SP-DQN   | Values | Description                                              |
|----------------------|--------|----------------------------------------------------------|
| On-policy            | ❌      | The evaluate policy is the same as the target policy.    |
| Off-policy           | ✅      | The evaluate policy is different from the target policy. |
| Model-free           | ✅      | No need to prepare an environment dynamics model.        |
| Model-based          | ❌      | Need an environment model to train the policy.           |
| Discrete Action      | ✅      | Deal with discrete action space.                         |
| Continuous Action    | ❌      | Deal with continuous action space.                       |

## Q-Learning Recap

[**Q-Learning**](https://link.springer.com/article/10.1007/bf00992698) is a model-free RL algorithm where the agent learns a Q-value function $Q(s, a)$, which estimates the expected cumulative reward of taking action $a$ in state $s$ and following the optimal policy thereafter. The [**Bellman equation**](https://en.wikipedia.org/wiki/Bellman_equation) for Q-learning is given by:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha [ r + \gamma \max_{a'}Q(s', a') - Q(s, a) ].
$$

Where $\alpha$ is the learning rate, $r$ is the reward, $\gamma$ is the discount factor, and $s'$ is the next state.

## Split Parameterized Q-Learning

The key idea behind SP-DQN is to split the Q-network into multiple parameterized parts. Each part is responsible for estimating the Q-values corresponding to a specific subset of actions. This structure reduces the computational complexity of training, as the different parts of the network can be trained independently or in parallel.

SP-DQN applies parameter splitting to the Q-function, allowing for more efficient representation and computation in environments with large action spaces.

## Deep Q-Network with Split Parameters

SP-DQN improves on DQN by using a split network architecture. The neural network is divided into multiple parts, each corresponding to a specific action category. The general steps involved are as follows:

1. **Action Space Split**: Split the action space into disjoint subsets, and assign each subset to a different parameterized network.
2. **Independent Network Training**: Train each parameterized network independently, which reduces training complexity and memory usage.
3. **Consolidated Q-Function**: The Q-function is computed by combining the outputs of each independent network, producing the final Q-value for each action.

The loss function for training SP-DQN is similar to DQN:

$$
L = \mathbb{E}_{(s, a, s', r) \sim \mathcal{D}}[(y - Q(s, a; \theta))^2],
$$

where $y = r + \gamma \max_{a'}{Q(s', a'; \theta^{-})}$, and $\theta^{-}$ represents the parameters of the target network.

## $\epsilon$-Greedy Exploration

SP-DQN uses the same $\epsilon$-greedy exploration policy as DQN:

$$
\pi(s) = 
\begin{cases}
\arg\max_{a}Q(s, a) & \text{with probability } 1-\epsilon, \\
\text{a random action} & \text{with probability } \epsilon.
\end{cases}
$$

This policy ensures that the agent explores the environment randomly with probability $\epsilon$ and exploits the learned policy otherwise.

## Algorithm

The full algorithm for training SP-DQN is presented in Algorithm 1:


## Framework

The overall agent-environment interaction of SP-DQN, as implemented in XuanCe, is illustrated in the figure below:


## Run SP-DQN in XuanCe

Before running SP-DQN in XuanCe, you need to prepare a conda environment and install `xuance` following
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run SP-DQN directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='spdqn',
                           env='classic_control',  # Choices: classic_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run SP-DQN with different configurations, you can build a new `.yaml` file, e.g., `my_config.yaml`.
Then, run SP-DQN by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='spdqn',
                       env='classic_control',  # Choices: classic_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Custom Environment

If you would like to run XuanCe's SP-DQN in your own environment that was not included in XuanCe,
you need to define the new environment following the steps in
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepare the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
`spdqn_myenv.yaml`.

After that, you can run SP-DQN in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import SP_DQN_Agent

configs_dict = get_configs(file_dir="spdqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = SP_DQN_Agent(config=configs, envs=envs)  # Create a SP-DQN agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@article{he2017split,
  title={Split Parameterized Deep Q-Network},
  author={He, Xun and Zhang, Xiang and Liu, Jian and Lu, Yun},
  journal={arXiv preprint arXiv:1707.02785},
  year={2017}
}

