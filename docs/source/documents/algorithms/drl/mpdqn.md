# Multi-pass Parameterised Deep Q-Network (MP-DQN)
**Paper Link:**[**https://arxiv.org/abs/1905.04388**](...)

Multi-pass Parameterised Deep Q-Network (MP-DQN) is an extension of the DQN algorithm 
that addresses the challenge of handling both discrete and continuous action spaces simultaneously. 
It was introduced to solve parameterised action space problems, 
where actions consist of discrete choices each with associated continuous parameters.

This table lists some general features about MP-DQN algorithm:

| Features of MP-DQN | Values | Description                                              |
|--------------------|--------|----------------------------------------------------------|
| On-policy          | ❌      | The evaluate policy is the same as the target policy.    |
| Off-policy         | ✅      | The evaluate policy is different from the target policy. | 
| Model-free         | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based        | ❌      | Need an environment model to train the policy.           | 
| Discrete Action    | ✅      | Deal with discrete action space.                         |   
| Continuous Action  | ✅      | Deal with continuous action space.                       |    

## Parameterised Action Spaces
Parameterised action spaces combine discrete actions with continuous parameters. 
Each discrete action $k \in K$ has associated continuous parameters $x_k \in X_k$. 
The full action space is defined as:
$$\mathcal{A}=\bigcup_{k\in[K]}\{a_k=(k,x_k)|x_k\in\mathcal{X}_k\}.$$
This hybrid structure makes traditional RL algorithms insufficient, as they typically handle either discrete or continuous actions, 
but not both simultaneously.

## MP-DQN Architecture
MP-DQN employs a novel neural network architecture that consists of:

Shared Feature Extractor: A common backbone network that processes the state input

Discrete Action Head: Estimates Q-values for each discrete action $Q(s, k)$

Continuous Parameter Heads: For each discrete action $k$, a separate network predicts the optimal continuous parameters $\mu_k(s)$

The network outputs both the Q-values for discrete actions and the corresponding continuous parameters:
$$Q(s,k,x_k)=\mathbb{E}_{r,s^{\prime}}\left[r+\gamma\max_{k^{\prime}}Q(s^{\prime},k^{\prime},x_{k^{\prime}}^Q(s^{\prime}))|s,k,x_k\right]$$
## Multi-pass Q-value Estimation
The key innovation in MP-DQN is the multi-pass approach for Q-value estimation:

Forward Pass: For each discrete action $k$, compute the continuous parameters $\mu_k(s)$

Q-value Calculation: Estimate $Q(s, k, \mu_k(s))$ for each discrete action

Action Selection: Choose the discrete action with the highest Q-value:
$$k^*=\arg\max_{k\in K}Q(s,k,\mu_k(s))$$
The selected action is then $(k^, \mu_{k^}(s))$.

## Loss Functions
MP-DQN uses two separate loss functions:

## Q-value Loss
The Q-network is trained using the temporal difference error:
$$L_Q(\theta_Q)=\mathbb{E}_{(s,k,x_k,r,s^{\prime})\sim D}\left[\frac{1}{2}\left(y-Q(s,k,x_k;\theta_Q)\right)^2\right]$$
where the target $y$ is:
$$y=r+\gamma\operatorname*{max}_{k^{\prime}\in[K]}Q(s^{\prime},k^{\prime},x_{k^{\prime}}(s^{\prime};\theta_{x});\theta_{Q})$$
## Policy Loss
The continuous parameter networks are trained to maximize the Q-values:
$$L_x(\theta_x)=-\sum_{k=1}^KQ\left(s,k,\mathbf{x}(s;\theta_x);\theta_Q\right)$$
where $\phi$ represents the parameters of the continuous policy networks.

Experience Replay and Target Networks
Similar to DQN, MP-DQN employs:

## Experience Replay: Stores transitions $(s, k, x_k, r, s')$ in a replay buffer

Target Networks: Maintains separate target networks for both Q-value and policy networks that are updated periodically

ϵ-greedy Exploration: Uses epsilon-greedy strategy for discrete action selection with random continuous parameters during exploration


## Framework
The overall agent-environment interaction of MP-DQN, as implemented in XuanCe, is illustrated in the figure below.
```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-MPDQN.png
    :width: 80%
    :align: center
```
## Run MP-DQN in XuanCe
Before running MP-DQN in XuanCe, you need to prepare a conda environment and install xuance following
the installation steps.
### Run Build-in Demos
After completing the installation, you can open a Python console and run MP-DQN directly using the following commands:


```python3
import xuance
import xuance
runner = xuance.get_runner(method='mpdqn',
                           env='parameterised_action_space',  # Choices: parameterised_action_space
                           env_id='Platform-v0',  # Choices: Platform-v0, Goal-v0, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```
### Run With Self-defined Configs
If you want to run MP-DQN with different configurations, you can build a new .yaml file, e.g., my_mpdqn_config.yaml.
Then, run the MP-DQN by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='mpdqn',
                       env='parameterised_action_space',  # Choices: parameterised_action_space
                       env_id='Platform-v0',  # Choices: Platform-v0, Goal-v0, etc.
                       config_path="my_mpdqn_config.yaml",  # The path of my_mpdqn_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Custom Environment
If you would like to run XuanCe's MP-DQN in your own parameterised action environment,
you need to define the new environment following the steps in
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``mpdqn_myenv.yaml``.

After that, you can run MP-DQN in your own environment with the following code:

```python3

import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import MP_DQN_Agent

configs_dict = get_configs(file_dir="mpdqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyParameterisedEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = MP_DQN_Agent(config=configs, envs=envs)  # Create a MP-DQN agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```
## Strengths of MP-DQN
Handles both discrete and continuous action components simultaneously

Suitable for real-world applications with hybrid action spaces

Maintains sample efficiency through experience replay

Provides stable training through target networks

Extends the applicability of DQN to parameterised action problems


## Citation
```{code-block}

@misc{bester2019multipassqnetworksdeepreinforcement,
      title={Multi-Pass Q-Networks for Deep Reinforcement Learning with Parameterised Action Spaces}, 
      author={Craig J. Bester and Steven D. James and George D. Konidaris},
      year={2019},
      eprint={1905.04388},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1905.04388}, 
}

```