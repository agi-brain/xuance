# DQN with Noisy Layers (Noisy DQN)

**Paper Link:** [**https://arxiv.org/pdf/1706.01905**](https://arxiv.org/pdf/1706.01905).

Noisy DQN is a variant of the traditional Deep Q-Network (DQN) 
that introduces noise into the weights of the Q-network to improve exploration during the learning process. 
This is aimed at addressing one of the key challenges in reinforcement learning: balancing exploration and exploitation.

This table lists some general features about Noisy DQN algorithm:

| Features of Noisy DQN | Values | Description                                              |
|-----------------------|--------|----------------------------------------------------------|
| On-policy             | ❌      | The evaluate policy is the same as the target policy.    |
| Off-policy            | ✅      | The evaluate policy is different from the target policy. | 
| Model-free            | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based           | ❌      | Need an environment model to train the policy.           | 
| Discrete Action       | ✅      | Deal with discrete action space.                         |   
| Continuous Action     | ❌      | Deal with continuous action space.                       |

## Key Ideas of Noisy DQN

**Exploration vs. Exploitation**: In standard DQN, exploration is often controlled by an $\epsilon$-greedy policy, 
where the agent randomly selects actions with a certain probability (epsilon), 
and exploits the best-known action the rest of the time. Noisy DQN attempts to address the challenge of exploration by introducing noise directly into the network's parameters, 
rather than relying solely on random action selection.

**Noisy Networks**: Instead of using a fixed epsilon for exploration, Noisy DQN introduces noise into the parameters of the Q-network itself. 
This is done by adding parameter noise to the Q-network’s weights, which modifies the output Q-values, 
encouraging exploration of different actions and states. 

**Noisy Linear Layers**: In the Noisy DQN architecture, the traditional fully connected layers of the neural network are replaced with "noisy" layers. 
These noisy layers add noise to the weights of the layers during training, making the agent’s decision-making process inherently more exploratory. 

**The Noisy Network Formula**: For each layer in the network, the weights are parameterized as:

$$
w = \mu + \sigma \cdot \epsilon,
$$

where:
- $\mu$ is the mean or the base weight;
- $\sigma$ is the standard deviation that controls the level of noise;
- $\epsilon$ is a sample from a noise distribution (usually Gaussian). 
The noise $\epsilon$ is sampled at the beginning of each episode or iteration, ensuring the noise is dynamic during training.

The Noisy DQN has the three main benefits:

- **Improved Exploration**: By introducing noise in the Q-values, the agent is encouraged to explore a broader range of actions, rather than exploiting the current best-known action.
- **Adaptive Exploration**: The level of exploration can be adjusted automatically as part of the training, eliminating the need to manually tune exploration parameters like epsilon.
- **Efficient Training**: Noisy DQN can improve sample efficiency because it uses the exploration to visit less frequently encountered states, potentially leading to better performance in complex environments.

## Framework

Noisy DQN retains the same overall structure as 
[**DQN**](dqn.md#framework) 
(i.e., experience replay, target networks, etc.), 
but replaces the exploration mechanism with the noisy layers in the Q-network.

## Run Noisy DQN in XuanCe

Before running Noisy DQN in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run Noisy DQN directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='noisydqn',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run Noisy DQN with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the Noisy DQN by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='noisydqn',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's Noisy DQN in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
``noisydqn_myenv.yaml``.

After that, you can run Noisy DQN in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import NoisyDQN_Agent

configs_dict = get_configs(file_dir="noisydqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = NoisyDQN_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citations

```{code-block} bash
@inproceedings{
  plappert2018parameter,
  title={Parameter Space Noise for Exploration},
  author={Matthias Plappert and Rein Houthooft and Prafulla Dhariwal and Szymon Sidor and Richard Y. Chen and Xi Chen and Tamim Asfour and Pieter Abbeel and Marcin Andrychowicz},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=ByBAl2eAZ},
}
```
