# Multi-agent Proximal Policy Optimization (MAPPO)

**Paper Link:** [**https://proceedings.neurips.cc/paper_files/paper/2022**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html).

## Background and Research Motivation

In the field of Multi - Agent Reinforcement Learning (MARL), the problem of collaborative decision - making in cooperative tasks has been a research hotspot in recent years. Traditional multi - agent training methods often rely on the Centralized Training with Decentralized Execution (CTDE) paradigm, aiming to use global information to improve training stability while ensuring that each agent makes independent decisions based only on local observations during the execution stage.

Although off - policy algorithms such as QMIX and MADDPG have achieved remarkable results in multiple benchmark tasks, on - policy methods have shown great potential in large - scale distributed systems due to their good convergence and stability. Among them, Proximal Policy Optimization (PPO), as one of the most successful policy gradient algorithms in single - agent scenarios, is widely favored for its simplicity, high efficiency, and strong hyperparameter robustness.

However, in multi - agent environments, the application of PPO has long been questioned, mainly due to two assumptions: one is that its sample efficiency is lower than that of off - policy methods; the other is that the implementation experience in single - agent environments is difficult to transfer to multi - agent scenarios. Recent research shows that after appropriate adjustments, the multi - agent algorithm based on PPO, namely Multi - Agent PPO (MAPPO), performs excellently in various cooperative tasks and even surpasses the mainstream off - policy baselines. This discovery has prompted the academic community to re - examine the status of PPO in MARL, and MAPPO has thus become an important class of strong baseline algorithms.

This table lists some general features about MAPPO algorithm:

| Features of MAPPO   | Values | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| Fully Decentralized                  | ❌      | There is no communication between agents.                                  |
| Fully Centralized                    | ❌      | Agents send all information to the central controller, and the controller will make decisions for all agents. |
| Centralized Training with Decentralized Execution (CTDE) | ✅ | The central controller is used in training and abandoned in execution.     |
| On-policy         | ✅      | The evaluate policy is the same as the target policy.    |
| Off-policy        | ❌      | The evaluate policy is different from the target policy. | 
| Model-free        | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based       | ❌      | Need an environment model to train the policy.           | 
| Discrete Action   | ✅      | Deal with discrete action space.                         |   
| Continuous Action | ❌      | Deal with continuous action space.                       | 


## Key implementation details

MAPPO is essentially a natural extension of the standard Proximal Policy Optimization (PPO) algorithm under the Counterfactual Multi - Agent Policy Gradients (CTDE) framework.

### Separation of Policy and Value Function

MAPPO adopts two independent neural networks:

**Policy Network** $ \pi_\theta(u^a|o^a) $: Parameterizes the action distribution of an agent, taking the local observation $ o^a $ as input and outputting action probabilities. For homogeneous agents (e.g., the same unit type in SMAC), a parameter sharing mechanism is adopted, and different agents are distinguished by inputting "agent ID".

**Value Network** $ V_\phi(s) $: Only used for variance reduction in the training phase, taking the global state $ s $ (e.g., the concatenation of all agents' observations in MPE, the global battlefield information provided by the environment in SMAC) as input and outputting the state value $ V(s) $.

### Parameter Sharing
In environments with homogeneous agents (such as most SMAC maps), MAPPO typically employs a parameter - sharing strategy: all agents share the same set of policy and value network parameters. This not only significantly reduces the number of model parameters but also enhances data utilization, helping to alleviate the sparse reward problem in multi - agent learning.
### Value Estimation Optimization

#### Generalized Advantage Estimation (GAE)

Estimating the advantage function using GAE:

$$
A_t^{GAE} = \sum_{l=0}^{T - t} (\gamma\lambda)^l \delta_{t + l}, \text{ where } \delta_k = r_k + \gamma V(s_{k+1}) - V(s_k) ,
$$

and further improve the training stability through advantage normalization.

#### Value Normalization

Aiming at the unstable value learning caused by large reward fluctuations in multi-agent scenarios (e.g., the reward difference between victory and defeat in SMAC can reach over 200), MAPPO normalizes the value target using running mean and standard deviation: $V_{\text{norm}}(s_t) = \frac{V(s_t) - \mu_{\text{running}}}{\sigma_{\text{running}} + \epsilon}$  

where $\mu_{\text{running}}$ and $\sigma_{\text{running}}$ are the running mean and standard deviation of the value target updated in real time during training, and $\epsilon = 1e-8$ avoids division by zero. The paper’s empirical results show that this optimization can increase the win rate of complex SMAC maps (e.g., 3s5z) by 15%-20%.

### Clipping Mechanism

MAPPO employs the double - clipping mechanism of PPO, which acts on the policy ratio and value loss respectively:

Policy Clipping:

$$
\mathcal{L}^{CLIP} = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_{\theta}(a_t|o_t)}{\pi_{\theta_{old}}(a_t|o_t)}$​.

Value Clipping: Prevent the value function from being updated excessively and improve the training stability.

## Algorithm

The full algorithm for training MAPPO is presented in Algorithm 1:

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-MAPPO.png
    :width: 80%
```  
## Run MAPPO in XuanCe

Before running MAPPO in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run MAPPO directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='mappo',
                    env='mpe',  
                    env_id='simple_spread_v3',  
                    is_test=False)
runner.run() 
```
### Run With Self-defined Configs

If you want to run MAPPO with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the MAPPO by the following code block:

```python3
import xuance
runner = xuance.get_runner(method='mappo',
                       env='mpe', 
                       env_id='simple_spread_v3',  
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()
```
### Run With Custom Environment

If you would like to run XuanCe's MAPPO in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``mappo_myenv.yaml``.

After that, you can run MAPPO in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import MAPPO_Agents

configs_dict = get_configs(file_dir="mappo_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs) 
Agent = MAPPO_Agents(config=configs, envs=envs) 
Agent.train(configs.running_steps // configs.parallels)  
Agent.save_model("final_train_model.pth") 
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@article{yu2022surprising,
  title={The surprising effectiveness of ppo in cooperative multi-agent games},
  author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={24611--24624},
  year={2022}
}
```



