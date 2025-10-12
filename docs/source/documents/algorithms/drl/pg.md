# Policy Gradient (PG)

**Paper Link:** [**Download PDF**](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

The Policy Gradient (PG) algorithm, introduced by 
[Richard Sutton](http://www.incompleteideas.net/) et al. 
in their seminal 1999 paper 
"Policy Gradient Methods for Reinforcement Learning with Function Approximation", 
is a foundational approach in reinforcement learning for optimizing policies directly. 
It is particularly effective in scenarios where value-based methods like Q-learning struggle, 
such as high-dimensional or continuous action spaces.

| Features of PG    | Values | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| On-policy         | ✅      | The evaluate policy is the same as the target policy.    |
| Off-policy        | ❌      | The evaluate policy is different from the target policy. | 
| Model-free        | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based       | ❌      | Need an environment model to train the policy.           | 
| Discrete Action   | ✅      | Deal with discrete action space.                         |   
| Continuous Action | ✅      | Deal with continuous action space.                       |

## Method

### Motivation

The goal of PG methods is to directly optimize the policy $\pi_{\theta}(a | s)$, parameterized by $\theta$, 
by maximizing the expected cumulative reward:

$$
J(\theta) = \mathbb{E}_{\pi_{\theta}}{[\sum_{t=0}^{\infty}{\gamma^t r_t}]}.
$$

Instead of approximating the value function, PG methods compute the gradient of the objective $J(\theta)$ 
with respect to the policy parameters $\theta$, and then update $\theta$ to maximize $J(\theta)$.

### Policy Gradient

The core of the PG algorithm is the policy gradient theorem, which states:

$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log{\pi_{\theta}(a|s)Q^{\pi_{\theta}}(s, a)}],
$$

where,

- $\pi_{\theta}(a|s)$: The stochastic policy that outputs the probabilities of taking action $a$ in state $s$.
- $Q^{\pi_{\theta}}(s, a)$: The action-value function under the current policy.
- $\nabla_{\theta}\log{\pi_{\theta}(a|s)}$: The gradient of the log-policy with respect to its parameters $\theta$, often called the score function.

This formulation enables the policy to be updated by following the gradient of the expected reward.

## Run PG in XuanCe

Before running PG in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run PG directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='pg',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run PG with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the PG by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='pg',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's PG in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``pg_myenv.yaml``.

After that, you can run PG in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import PG_Agent

configs_dict = get_configs(file_dir="pg_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = PG_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@article{sutton1999policy,
  title={Policy gradient methods for reinforcement learning with function approximation},
  author={Sutton, Richard S and McAllester, David and Singh, Satinder and Mansour, Yishay},
  journal={Advances in neural information processing systems},
  volume={12},
  year={1999}
}
```
