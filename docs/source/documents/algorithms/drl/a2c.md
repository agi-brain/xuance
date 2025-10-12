# Advantage Actor Critic(A2C)

The Advantage Actor Critic(A2C) algorithm is a reinforcement learning method which is based on the policy gradient method and the value function approximation. It uses an advantage function instead of an action-state value function to improve the stability and performance of learning.

This table lists some general features about A2C algorithm:


| Features of A2C   | Values | Description                                            |
|-------------------| ------ | ------------------------------------------------------ |
| On-policy         | ✅     | The evaluate policy is the same as the target policy.  |
| Off-Policy        | ❌     | The evaluate policy is different as the target policy. |
| Model-free        | ✅     | No need to prepare an environment dynamics model.      |
| Model-based       | ❌     | Need an environment model to train the policy.         |
| Discrete Action   | ✅     | Deal with discrete action space.                       |
| Continuous Action | ✅     | Deal with continuous action space.                     |

## Actor Critic(AC) Framework

Actor Critic Method selects actions through the Actor, evaluates these actions through the Critic, and cooperates with each other to improve.

### Critic

Critic is also called value network which uses neural network $Q^\pi(s,a;w)$ to approximate the action value function $Q^\pi(s,a)$. In one-step Q-Learning, the parameters $w$ of the action value function $Q^\pi(s,a;w)$ are learned by iteratively minimizing a sequence of loss functions, where the $i$th loss function defined as

$$
L_i(w_i)=\mathbb{E}[(r+\gamma Q(s',a';w_{i})-Q(s,a;w_i))^2]
$$

where s′ is the state encountered after state s.

### Actor

Actor is also called policy network which is similar to the [**Policy Gradient(PG)**](./pg_agent.md) method. Actor directly optimizes the policy to achieve the maximum reward. Its' objective function is expressed in the following:

$$
J(\theta) = \mathbb{E}_{\pi_{\theta}}{[\sum_{t=0}^{\infty}{\gamma^t r_t}]}.
$$

To optimize the policy function $\pi_\theta$, we calculate the gradient of the objective function $J(\theta)$ with respect to the parameters $\theta$:

$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log{\pi_{\theta}(a|s)Q^{\pi_{\theta}}(s, a)}].
$$

In the process of training the actor, we use the action value $Q(s,a;w)$ obtained by the critic as an approximation of the $Q(s,a)$. By alternately training the actor and the critic, we ultimately achieve the goal of maximizing $J(\theta)$.

## Advantage Actor Critic(A2C)

In the above-mentioned Actor-Critic framework, we use $Q(s,a;w)$ to update the policy. In Advantage Actor Critic (A2C), we use the advantage function to represent the additional reward for taking a certain action relative to the average state value:

$$
A(a_t,s_t)=Q(a_t,s_t)−V(s_t)\approx r_t+\gamma V(s_{t+1}) - V(s_t),
$$

where

$$
Q_\pi(a_t,s_t)−V_\pi(s_t)=\mathbb{E}[R_t+\gamma v_\pi(S_{t+1}) - v_\pi(S_t)|S_t=s_t].
$$

The advantage function reduces the variance in policy gradient estimates. And then we can rewrite the gradient of the objective function $J(\theta)$:

$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log{\pi_{\theta}(a|s)A^\pi(s,a)}].
$$

For critic network, we rewrite the loss function as:

$$
L(w)=\mathbb{E}[(r_t+\gamma V(s_{t+1};w)-V(s_t;w))^2].
$$

## Framework

The structural framework of A2C, as implemented in XuanCe, is illustrated in the figure below.

```{eval-rst}
.. image:: ./../../../_static/figures/algo_framework/a2c_framework.png
    :width: 80%
    :align: center
```

## Run A2C in XuanCe

Before running A2C in XuanCe, you need to prepare a conda environment and install ``xuance`` following the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run A2C directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='a2c',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run A2C with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the A2C by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='a2c',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the [**tutorial of configs**](./../../configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's A2C in your own environment that was not included in XuanCe, you need to define the new environment following the steps in [**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst). Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``a2c_myenv.yaml``.

After that, you can run A2C in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import A2C_Agent

configs_dict = get_configs(file_dir="a2c_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = A2C_Agent(config=configs, envs=envs)  # Create a A2C agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```
