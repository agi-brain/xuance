# Categorical 51 DQN

**Paper Link:** [**https://proceedings.mlr.press/v70/bellemare17a.html**](https://proceedings.mlr.press/v70/bellemare17a.html).

The C51 algorithm (Categorical DQN) is a variant of the 
[**DQN**](./dqn_agent.md) 
that introduces a distributional approach to reinforcement learning. 
Instead of predicting a single scalar value for the Q-function (expected future reward), 
C51 predicts a probability distribution over a discrete set of possible returns (rewards), 
enabling the agent to learn not just the expected return but also its uncertainty.

This table lists some general features about C51 algorithm:

| Features of C51   | Values | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| On-policy         | ❌      | The evaluate policy is the same as the target policy.    |
| Off-policy        | ✅      | The evaluate policy is different from the target policy. | 
| Model-free        | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based       | ❌      | Need an environment model to train the policy.           | 
| Discrete Action   | ✅      | Deal with discrete action space.                         |   
| Continuous Action | ❌      | Deal with continuous action space.                       |

## Method

### Problem With DQN

In regular DQN, the goal is to learn the expected return for each action in a given state:

$$
Q(s, a) = \mathbb{E}[G_t | S_t=s, A_t=a],
$$

where

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum^{\infty}_{k=0} {\gamma^k R_{t+k+1}}
$$

Instead of learning a single number $Q(s, a)$, 
C51 learns the entire distribution of possible rewards for each action, denoted as $Z(s, a)$.

C51 divides the possible range of rewards (e.g., from -1 to 1) into a fixed number of bins (called atoms).
For each atom, the algorithm predicts the probability that the reward will fall into that atom. 

The agent can now make decisions based on both the expected reward and its uncertainty.

### Categorical Representation

To apporoximate $Z(s, a)$, C51 represents it as a categorical distribution over $N$ discrete atoms 
within a predefined value range $[v_{min}, v_{max}]$. The atoms are defined as:

$$
z_i = v_min + i \cdot \Delta, \Delta = \frac{v_{max} - v_{min}}{N-1}, i=0, 1, \dots, N-1.
$$

Each atom $z_i$ is associated with a probability $p_i$, forming the categorical distribution:

$$
P(Z(s,a)=z_i) = p_i, \sum_{i=0}^{N-1}{p_i}=1.
$$

### Distributional Bellman Equation

The Bellman equation for the return distribution is given by:

$$
Z(s, a) := R + \gamma Z(S', A').
$$

This equation states that the distribution of returns for the current state-action pair is determined 
by the immediate reward $R$ and the discounted distribution of returns from the next state $S'$.

In practice, The next-state distribution $Z(S', A')$ is projected back onto the fixed set of atoms $z_i$ to maintain consistency.

## Algorithm

The full algorithm for training C51 is presented in Algorithm 1.

```{eval-rst}
.. image:: ./../../../../_static/figures/pseucodes/pseucode-C51.png
    :width: 65%
    :align: center
    
.. note::

    Algorithm 1 computes the projection in time linear in N.
```

## Run C51 in XuanCe

Before running C51 in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run C51 directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='c51',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run C51 with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the C51 by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='c51',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../configs/configuration_examples.rst).

### Run With Customized Environment

If you would like to run XuanCe's C51 in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../../usage/new_envs.rst).
Then, [**prepapre the configuration file**](./../../../usage/new_envs.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``c51_myenv.yaml``.

After that, you can run C51 in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import C51_Agent

configs_dict = get_configs(file_dir="c51_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = C51_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@inproceedings{bellemare2017distributional,
  title={A distributional perspective on reinforcement learning},
  author={Bellemare, Marc G and Dabney, Will and Munos, R{\'e}mi},
  booktitle={International conference on machine learning},
  pages={449--458},
  year={2017},
  organization={PMLR}
}
```

## APIs

### PyTorch

```{eval-rst}
.. automodule:: xuance.torch.agents.qlearning_family.c51_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### TensorFlow2

```{eval-rst}
.. automodule:: xuance.tensorflow.agents.qlearning_family.c51_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### MindSpore

```{eval-rst}
.. automodule:: xuance.mindspore.agents.qlearning_family.c51_agent
    :members:
    :undoc-members:
    :show-inheritance:
```
