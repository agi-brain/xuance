# Multi-agent Soft Actor-Critic (MASAC)

**Paper Link:** [**https://arxiv.org/abs/2104.06655**](https://arxiv.org/abs/2104.06655)

Multi-agent Soft Actor-Critic (MA-SAC) is a multi-agent deep reinforcement learning algorithm
that integrates the fundamental framework of the SAC algorithm with value function decomposition.
It follows the paradigm of centralized training and decentralized execution (CTDE), enabling efficient off-policy learning.
Moreover, it partially alleviates the credit assignment problem in both discrete and continuous action spaces.

This table lists some general features about MASAC algorithm:   

| Features of MASAC                                       | Values | Description                                                                                                   |
|---------------------------------------------------------|--------|---------------------------------------------------------------------------------------------------------------|
| Fully Decentralized                                     | ❌      | There is no communication between agents.                                                                     |
| Fully Centralized                                       | ❌      | Agents send all information to the central controller, and the controller will make decisions for all agents. | 
| Centralized Training With Decentralized Execution(CTDE) | ✅      | The central controller is used in training and abandoned in execution.                                        | 
| On-policy                                               | ❌      | The evaluate policy is the same as the target policy.                                                         | 
| Off-policy                                              | ✅      | The evaluate policy is different from the target policy.                                                      |   
| Model-free                                              | ✅      | No need to prepare an environment dynamics model.                                                             |
| Model-based                                             | ❌      | Need an environment model to train the policy.                                                                | 
| Discrete Action                                         | ✅      | Deal with discrete action space.                                                                              | 
| Continuous Action                                       | ✅      | Deal with continuous action space.                                                                            |

## Framework

The following figure shows the algorithm structure of MASAC.

```{eval-rst}
.. image:: ./../../../_static/figures/algo_framework/MASAC_framework.png
    :width: 80%
    :align: center
```

In this framework, Left: mixing network structure. Red figures are the hyper-networks that produce the weights
and biases for the mixing network layers. Middle: the overall Qmix architecture. Right: agent’s local Q
network, which is in green, the $i$ means the corresponding one-hot vector to distinguish different agents.

## Key Ideas of MASAC

### Value Function Decomposition

Each agent learns a local Q-value function, which are then aggregated by a learnable mixing network to produce the global joint Q-value.

$$
Q^{\mathrm{tot}}(\tau,\mathbf{a})=q^{mix}(s,\left[q^i\left(\tau^i,a^i\right)\right])
$$

$q^{mix}$ represents a non-linear monotonic factorization structure.

### multi-agent Soft Actor-Critic

This method adopts the practical approximation to soft policy iteration, the critic loss function of the MASAC method in multi-agent setting is:

$$
\mathcal{L}(\phi)=\mathbb{E}_{\mathcal{D}} \left[\left(r_t + \gamma \min_{j\in\{1,2\}} \hat{Q}_{\phi_j^{\prime}}^{\mathrm{targ}} - Q_\phi^{\mathrm{tot}}(s_t,\tau_t,a_t)\right)^2\right]
$$

Where $\phi$ denotes the parameters of the current Q-network, and $\phi_j^{\prime}$ denotes the parameters of the target Q-network.The target value is defined as:

$$
r_t+\gamma\min_{j\in\{1,2\}}\hat{Q}_{\phi_j^{\prime}}^{\mathrm{targ}}
$$

Where $\hat{Q}_{\phi_j^{\prime}}^{\mathrm{targ}}$ can be written as:

$$
\hat{Q}_{\phi_j^{\prime}}^{\mathrm{targ}}=\mathbb{E}_{\pi_\theta}\left[Q_{\phi_j^{\prime}}^{\mathrm{tot}}(s_{t+1},\tau_{t+1},a_{t+1})-\alpha\mathrm{log}\pi(a_{t+1}|\tau_{t+1})\right]
$$

Where $\alpha\log\pi\left(a_{t+1}\mid\tau_{t+1}\right)$ is regarded as an entropy regularization term.

Iterative proces based on soft policy, the objective for policy update is below:

$$
\mathcal{L}(\theta)=\mathbb{E}_{\mathcal{D}} \left[\alpha\log\pi(a_t|\tau_t)-Q_{\phi^{\prime}}^{\mathrm{tot}}(s_t,\tau_t,a_t)\right]
$$

$\alpha$ is a hyper-parameter that
controls the trade-off between maximizing the entropy of policy and the expected discounted return.It can be designed to learn dynamically similar to SAC:

$$
\mathcal{L}(\alpha)=\mathbb{E}_{a_t\sim\pi_t} \left[-\alpha\log\pi_t(a_t|\tau_t)-\alpha\overline{\mathcal{H}} \right]
$$

## Algorithm

The full algorithm for training MASAC is presented in Algorithm 1:

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-MASAC.png
    :width: 80%
    :align: center
```

## Run MASAC in XuanCe

Before running MASAC in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run MASAC directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='masac',
                           env='mpe',
                           env_id='simple_spread_v3',
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run MASAC with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the MASAC by the following code block:

```python3
import xuance 
runner = xuance.get_runner(method='masac',
                       env='mpe',
                       env_id='simple_spread_v3',
                       config_path="my_config.yaml",
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).


### Run With Custom Environment

If you would like to run XuanCe's MASAC in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``masac_myenv.yaml``.

After that, you can run MASAC in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import MASAC_Agents

configs_dict = get_configs(file_dir="masac_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = MASAC_Agents(config=configs, envs=envs)  # Create a MASAC agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@misc{pu2021decomposedsoftactorcriticmethod,
      title={Decomposed Soft Actor-Critic Method for Cooperative Multi-Agent Reinforcement Learning}, 
      author={Yuan Pu and Shaochen Wang and Rui Yang and Xin Yao and Bin Li},
      year={2021},
      eprint={2104.06655},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2104.06655}, 
}
```