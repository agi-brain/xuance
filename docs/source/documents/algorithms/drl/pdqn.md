# Parametrized Deep Q-Network(P-DQN)

**Paper Link:**[**https://arxiv.org/pdf/1810.06394**](https://arxiv.org/pdf/1810.06394)

Parameterized Deep Q-Network(P-DQN) is a framework for the hybrid action space without approximation or relaxation,combining the spirits of both DQN (dealing with discrete action space) and DDPG (dealing with continuous action space) by seamlessly integrating them.

This table lists some general features about P-DQN algorithm:


| Features of P-DQN | Values | Description                                            |
| ----------------- | ------ | ------------------------------------------------------ |
| On-policy         | ❌     | The evaluate policy is the same as the target policy.  |
| Off-Policy        | ✅     | The evaluate policy is different as the target policy. |
| Model-free        | ✅     | No need to prepare an environment dynamics model.      |
| Model-based       | ❌     | Need an environment model to train the policy.         |
| Discrete Action   | ✅     | Deal with discrete action space.                       |
| Continuous Action | ✅     | Deal with continuous action space.                     |
| Hybrid Action     | ✅     | Deal with hybrid action space.                         |

## Discrete-Continuous Hybrid Action Space

The hybrid action is defiend by the following hierarchical structure. Firstly we choose a high level action $\mathcal{k}$ from a discrete set $[K]$; upon choosing $\mathcal{k}$, we further choose a low level parameter $\mathcal{x_k}\in\mathcal{\mathcal{X}_k}$ which is associated with the $k$-th high level action.Here $\mathcal{X}_k$ is a continuous set for all $k\in[K]$.

$$
\mathcal{A}=\{ (k,x_k)|x_k \in \mathcal{X}_k \quad for\; all\;k\in[K] \}
$$

## Key Idea of P-DQN

In hybrid action space, we denote the action value function by $Q(s,a)=Q(s,k,x_k)$ where $s\in S$, $k\in[K]$, and $x_k\in\mathcal{X}_k$. Let $k_t$ be the discrete action selected at time $t$ and let $x_t$ be the associated continuous parameter. Then the Bellman equation becomes:

$$
Q(s_t,k_t,x_{k_t})=\underset{r_t,s_{t+1}}{\mathbb{E}}[r_t+\gamma\underset{k\in[K]}{max}\underset{x_k\in\mathcal{X}_k}{sup}Q(s_{t+1},k,x_k)|s_t=s,a_t=(k_t,x_{k_t})].
$$

When the function $Q$ is fixed, for any $s\in S$ and $k\in[K]$, we can view $argsup_{x_k\in\mathcal{X}_k}Q(s,k,x_k)$ as a function $x_k^Q$: $S→ \mathcal{X}_k$. Then Bellman equation can be rewrite as:

$$
Q(s_t,k_t,x_{k_t})=\underset{r_t,s_{t+1}}{\mathbb{E}}[r_t+\gamma\underset{k\in[K]}{max}Q(s_{t+1},k,x_k^Q)|s_t=s].
$$

Similar to the deep Q-Network, we use a deep neural network $Q(s,k,x_k,\omega)$ to approximate $Q(s,k,x_k)$ where $\omega$ denotes the network weights. For such a $Q(s,k,x_k,\omega)$ we approximate $x_k^Q$ with a deterministic policy network $x_k(·;\theta):S→ \mathcal{X}_k$ where θ denotes the network weights of the policy network. When $\omega$ is fixed, we want to find $\theta$ such that:

$$
Q(s,k,x_k(s;\theta);\omega)\approx \underset{x_k\in\mathcal{X_k}}{sup}Q(s,k,x_k;\omega) \quad for \; each \; k\in[K].
$$

Then similar to DQN, we could estimate $\omega$ by minimizing the mean-squared Bellman error via gradient descent.

$$
y_t={\sum_{i=0}^{n-1} }{\gamma^ir_{t+i}+\gamma^n \underset{k\in[K]}{max} Q(s_{t+n},k,x_k(s_{t+n};\theta);\omega)}.
$$

We use the least squares loss function for $\omega$ like DQN, since we aim to find $\theta$ that maximize $Q(s,k,x_k(s;\theta);\omega)$ with $\omega$ fixed, we use the loss function for $\omega$ as following:

$$
\ell^Q(\omega)=\frac{1}{2}[Q(s_t,k,x_k;\omega)-y_t]^2 \quad and \quad \ell^\Theta (\theta)=-\sum_{k=1}^{K}Q(s_t,k,x_k(s_t;\theta);\omega_t).
$$

Then update the weights using stochastic gradient methods, we would minimize the loss function $\ell^\Theta (\theta)$ when $\omega_t$ is fixed.

## Algorithm

The full algorithm for training P-DQN is presented in Algorithm 1:

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-PDQN.png
    :width: 100%
    :align: center
```

## Run P-DQN in XuanCe

Before running P-DQN in XuanCe, you need to prepare a conda environment and install ``xuance`` following
the [**installation steps**](./../../usage/installation.rst#install-via-pypi).

### Run With Custom Demos

If you would like to run XuanCe's P-DQN in your own environment that was not included in XuanCe, you need to define the new environment following the steps in [**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst). Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)``pdqn_myenv.yaml``.

After that, you can run P-DQN in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTER_ENV
from xuance.environment import make_envs
from xuance.torch.agents import PDQN_Agent

config_dict = get_configs(file_dir="pdqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = PDQN_Agent(config=configs, envs=envs)  # Create a PDQN agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block}
@article{xiong2018parametrized,
  title={Parametrized deep q-networks learning: Reinforcement learning with discrete-continuous hybrid action space},
  author={Xiong, Jiechao and Wang, Qing and Yang, Zhuoran and Sun, Peng and Han, Lei and Zheng, Yang and Fu, Haobo and Zhang, Tong and Liu, Ji and Liu, Han},
  journal={arXiv preprint arXiv:1810.06394},
  year={2018}
}
```
