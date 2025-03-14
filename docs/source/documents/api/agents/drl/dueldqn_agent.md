# Dueling Deep Q-Network (Dueling DQN)

**Paper Link:** [**https://proceedings.mlr.press/v48/wangf16.pdf**](https://proceedings.mlr.press/v48/wangf16.pdf).

The Dueling Deep Q-Network (Dueling DQN) is an improvement to the standard DQN architecture 
that aims to enhance the efficiency and stability of Q-value estimation. 
It introduces a new neural network architecture to separately estimate the state-value function 
and the action advantage function, addressing key limitations of traditional DQNs.

This table lists some general features about Dueling DQN algorithm:

| Features of Dueling DQN | Values | Description                                              |
|-------------------------|--------|----------------------------------------------------------|
| On-policy               | ❌      | The evaluate policy is the same as the target policy.    |
| Off-policy              | ✅      | The evaluate policy is different from the target policy. | 
| Model-free              | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based             | ❌      | Need an environment model to train the policy.           | 
| Discrete Action         | ✅      | Deal with discrete action space.                         |   
| Continuous Action       | ❌      | Deal with continuous action space.                       |

## Key Ideas of Dueling DQN

Let $V(s)$ represent the overall value of state $s$. 
$A(s, a)$ is the advantage function that measures the relative benefit of taking a specific action $a$ given state $s$.
The relationship between $V(s)$, $A(s, a)$, and $Q(s, a)$ is:

$$
A(s, a) = Q(s, a) - V(s).
$$

Hence, the Q-value of a state-action pair can be expressed as:

$$
Q(s, a) = A(s, a) + V(s).
$$

This decomposition helps decouple the value of the state from the advantages of individual actions.
The architecture of Dueling DQN can be illustrated as the following figure:

```{eval-rst}
.. image:: ./../../../../_static/figures/algo_framework/duel_dqn_networks.png
    :width: 70%
    :align: center
```

## Framework

The overall agent-environment interaction of Dueling DQN, as implemented in XuanCe, is illustrated in the figure below.

```{eval-rst}
.. image:: ./../../../../_static/figures/algo_framework/dqn_framework.png
    :width: 100%
    :align: center
```

## Run Dueling DQN in XuanCe

Before running Dueling DQN in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run Dueling DQN directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='dueldqn',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run Dueling DQN with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the Dueling DQN by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='dueldqn',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../configs/configuration_examples.rst).

### Run With Customized Environment

If you would like to run XuanCe's Dueling DQN in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../../usage/new_envs.rst).
Then, [**prepapre the configuration file**](./../../../usage/new_envs.rst#step-2-create-the-config-file-and-read-the-configurations) 
``duelqn_myenv.yaml``.

After that, you can run Dueling DQN in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import DuelDQN_Agent

configs_dict = get_configs(file_dir="duel_dqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = DuelDQN_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citations

```{code-block} bash

@InProceedings{pmlr-v48-wangf16,
  title = 	 {Dueling Network Architectures for Deep Reinforcement Learning},
  author = 	 {Wang, Ziyu and Schaul, Tom and Hessel, Matteo and Hasselt, Hado and Lanctot, Marc and Freitas, Nando},
  booktitle = 	 {Proceedings of The 33rd International Conference on Machine Learning},
  pages = 	 {1995--2003},
  year = 	 {2016},
  editor = 	 {Balcan, Maria Florina and Weinberger, Kilian Q.},
  volume = 	 {48},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {New York, New York, USA},
  month = 	 {20--22 Jun},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v48/wangf16.pdf},
  url = 	 {https://proceedings.mlr.press/v48/wangf16.html},
  abstract = 	 {In recent years there have been many successes of using deep representations in reinforcement learning. Still, many of these applications use conventional architectures, such as convolutional networks, LSTMs, or auto-encoders. In this paper, we present a new neural network architecture for model-free reinforcement learning. Our dueling network represents two separate estimators: one for the state value function and one for the state-dependent action advantage function. The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm. Our results show that this architecture leads to better policy evaluation in the presence of many similar-valued actions. Moreover, the dueling architecture enables our RL agent to outperform the state-of-the-art on the Atari 2600 domain.}
}

```

## APIs

### PyTorch

```{eval-rst}
.. automodule:: xuance.torch.agents.qlearning_family.dueldqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### TensorFlow2

```{eval-rst}
.. automodule:: xuance.tensorflow.agents.qlearning_family.dueldqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### MindSpore

```{eval-rst}
.. automodule:: xuance.mindspore.agents.qlearning_family.dueldqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```
