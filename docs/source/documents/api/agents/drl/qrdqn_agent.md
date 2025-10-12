# Quantile Regression Deep Q-Network (QR-DQN)

**Paper Link:** [**https://ojs.aaai.org/index.php/AAAI/article/view/11791**](https://ojs.aaai.org/index.php/AAAI/article/view/11791).

Quantile Regression Deep Q-Network (QR-DQN) is an extension of the traditional DQN 
designed to improve the handling of uncertainty and variance in reinforcement learning, 
especially in environments where the rewards can be highly variable or noisy. 
QR-DQN combines elements of quantile regression with DQN, 
allowing it to learn a distribution over Q-values rather than just a single point estimate. 
This helps improve the stability and robustness of the learning process.

This table lists some general features about QR-DQN algorithm:

| Features of QR-DQN | Values | Description                                              |
|--------------------|--------|----------------------------------------------------------|
| On-policy          | ❌      | The evaluate policy is the same as the target policy.    |
| Off-policy         | ✅      | The evaluate policy is different from the target policy. | 
| Model-free         | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based        | ❌      | Need an environment model to train the policy.           | 
| Discrete Action    | ✅      | Deal with discrete action space.                         |   
| Continuous Action  | ❌      | Deal with continuous action space.                       |

## Method

### Distributional Reinforcement Learning

Traditional Q-learning estimates the expected return (mean) for each state-action pair. 
However, in many cases, the returns can be uncertain or variable, 
and just focusing on the mean may not capture the full picture of this uncertainty.

Distributional reinforcement learning seeks to model the distribution of possible returns for each state-action pair, 
not just the expected value.

### Quantile Regression

Quantile regression is a technique that estimates specific quantiles 
(e.g., the 50th percentile, 90th percentile) of a distribution, rather than the mean. 
This allows the model to capture the entire distribution of the possible returns, 
providing richer information about the variability in future rewards.

In QR-DQN, instead of learning a single Q-value, 
the agent learns multiple quantiles of the distribution over the Q-values.

### Architecture of QR-DQN

In QR-DQN, the Q-value function is represented by a distribution over possible returns. 
Specifically, the agent approximates the quantile function of the return distribution using a set of quantile values.

The quantiles $\tau_i$ (where $\tau_i \in [0, 1]$) correspond to different points in the return distribution 
(e.g., the 10th, 50th, and 90th percentiles). 
The algorithm learns a quantile regression loss to estimate the quantiles of the Q-value distribution, 
rather than learning a single expected Q-value.

### Loss function

QR-DQN uses the quantile Huber loss, 
which is a combination of the Huber loss function (which is less sensitive to outliers) and the quantile loss. 
The quantile loss penalizes the model based on how well it predicts the desired quantiles of the Q-value distribution.

The quantile loss for a given quantile $\tau$ is defined as:

$$
L_{\tau}(Q, \hat{Q}) = \rho_{\tau}(r - Q),
$$

where $r$ is the target return (the actual reward or the next state's predicted value), 
$Q$ is the predicted quantile value for a given state-action pair,
$\hat{Q}$ is the corresponding target quantile (from Bellman backup), 
and $\rho_{\tau}(z)$ is the check function defined as:

$$
\rho_{\tau}(z) = z(\tau - \mathbb{I}[z<0]),
$$

where $\mathbb{I}[z<0]$ is the indicator function that equals 1 when $z < 0$ and 0 otherwise.

The quantile regression loss encourages the model to learn quantile values 
that minimize the discrepancy between the predicted quantiles and the true return distributions.

## Algorithm

The full algorithm for training QR-DQN is presented in Algorithm 1:

```{eval-rst}
.. image:: ./../../../../_static/figures/pseucodes/pseucode-QRDQN.png
    :width: 70%
    :align: center
```

## Run QR-DQN in XuanCe

Before running QR-DQN in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run QR-DQN directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='qrdqn',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run QR-DQN with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the QR-DQN by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='qrdqn',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's QR-DQN in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../../usage/new_envs.rst).
Then, [**prepapre the configuration file**](./../../../usage/new_envs.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``qrdqn_myenv.yaml``.

After that, you can run QR-DQN in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import QRDQN_Agent

configs_dict = get_configs(file_dir="qrdqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = QRDQN_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citations

```{code-block} bash
@inproceedings{dabney2018distributional,
  title={Distributional reinforcement learning with quantile regression},
  author={Dabney, Will and Rowland, Mark and Bellemare, Marc and Munos, R{\'e}mi},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={32},
  number={1},
  year={2018}
}
```

## APIs

### PyTorch

```{eval-rst}
.. automodule:: xuance.torch.agents.qlearning_family.qrdqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### TensorFlow2

```{eval-rst}
.. automodule:: xuance.tensorflow.agents.qlearning_family.qrdqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### MindSpore

```{eval-rst}
.. automodule:: xuance.mindspore.agents.qlearning_family.qrdqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```
