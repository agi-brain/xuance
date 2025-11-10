# Independent Proximal Policy Optimization (IPPO)

**Paper Link:** [**https://proceedings.neurips.cc/paper_files/paper/2022**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html).

Independent Proximal Policy Optimization (IPPO) is a distributed policy - gradient method for Cooperative Multi - Agent Reinforcement Learning (MARL). Although Proximal Policy Optimization (PPO) has become one of the mainstream algorithms in single - agent reinforcement learning due to its stability and efficiency, its application in multi - agent environments has long been underestimated. Traditional views hold that compared with off - policy methods (such as MADDPG, QMIX, etc.), PPO - like algorithms have lower sample efficiency and are difficult to handle high - dimensional and non - stationary multi - agent training tasks.

However, recent research shows that with reasonable configuration of implementation details, multi - agent algorithms based on PPO can achieve performance that even surpasses mainstream off - policy methods in various benchmark tasks (Yu et al., NeurIPS 2022). Among them, as a typical architecture, IPPO embodies the idea of "Decentralized Execution with Independent Learning". Its original design intention is to achieve effective coordination of multiple agents only through local observations and shared reward signals without relying on a centralized critic or explicit communication mechanisms.

This method is especially suitable for practical scenarios with partial observability, heterogeneity, or limited communication. It has good scalability and engineering deployment potential, and thus has become an important part of the modern MARL baseline system.

This table lists some general features about IPPO algorithm:

| Features of IPPO   | Values | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| Fully Decentralized                  | ✅      | There is no communication between agents.                                  |
| Fully Centralized                    | ❌      | Agents send all information to the central controller, and the controller will make decisions for all agents. |
| Centralized Training with Decentralized Execution (CTDE) | ❌ | The central controller is used in training and abandoned in execution.     |
| On-policy         | ✅      | The evaluate policy is the same as the target policy.    |
| Off-policy        | ❌      | The evaluate policy is different from the target policy. | 
| Model-free        | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based       | ❌      | Need an environment model to train the policy.           | 
| Discrete Action   | ✅      | Deal with discrete action space.                         |   
| Continuous Action | ❌      | Deal with continuous action space.                       |

## Background and Research Motivation

### Performance Bottlenecks of Traditional Decentralized Methods
Early decentralized algorithms (such as IAC and Independent Q-Learning (IQL)) adopt a "independent learning per agent" mode. Although they avoid the dependence on global information, they have two major flaws: First, the **environmental non-stationarity is exacerbated** — the policy update of each agent changes the "environment" of other agents, leading to the accumulation of value estimation biases; Second, **credit assignment is ambiguous** — global rewards cannot be directly associated with the actions of a single agent, making it prone to problems like "free-riding" or "over-exploration". 

### "Application Gap" of PPO in Multi-Agent Scenarios

Single-agent PPO achieves a balance between stability and sample utilization through designs such as **importance sampling clipping** (limiting the magnitude of policy updates) and **Generalized Advantage Estimation (GAE)** (balancing the bias and variance of the advantage function), and supports parallel sampling. However, there have been two long-standing cognitive biases in the MARL field:

**Sample Efficiency Misconception**: It is believed that on-policy methods need to collect new samples in real time, and their efficiency is far lower than that of off-policy methods (such as QMIX and MADDPG);

**Fear of Non-Stationarity**: It is believed that the dynamic changes in multi-agent environments will amplify PPO’s gradient noise, leading to training collapse.
These cognitions have kept the multi-agent extension of PPO (such as IPPO) on the research fringe for a long time, and the few attempts (such as early IPPO variants) have not systematically verified its effectiveness in complex tasks.

## Key Ideas

Each agent runs a standard Proximal Policy Optimization (PPO) algorithm independently. The policy - network parameters are not shared among them (unless the agents are homogeneous), and they do not rely on a centralized value function either.

### Decentralized Decision - making Mechanism

Each agent $i \in \{1, \ldots, n\}$ maintains an independent policy network $\pi_{\theta_i}(a_i|o_i)$ and makes decision actions $a_i$​ solely based on its local observation $o_i$​, satisfying the demand for autonomous operation of each entity in the actual system.


### Policy Update under the IID (Independent and Identically Distributed) Assumption

Although the overall environment is non - stationary (i.e., changes in the policies of other agents lead to dynamic changes in the environment), IPPO still assumes that within each training batch, the experiences of each agent can be regarded as independent and identically distributed samples, and policy updates are carried out accordingly.

### Cooperative Incentives Driven by Shared Rewards

All agents share the same global reward function Rt​, thus forming a common optimization goal. This setting ensures that the direction of individual policy improvement is consistent with the improvement of collective performance, avoiding the coordination problems brought about by competitive games.

Although IPPO does not use a centralized value function for variance reduction, it still inherits the key advantage of PPO - the clipped surrogate objective, which effectively controls the policy update step size and prevents training collapse due to policy mutations.

## Limitations

### Multi-Agent Scale Sensitivity

In a fully decentralized architecture, agents can only make decisions based on local observations, lacking the ability to infer teammates' strategic intentions. When the number of agents exceeds 3, the problem of "collaboration fragmentation" intensifies — the action intentions of individual agents fail to align, leading to a sharp reduction in the coordination of joint policies.

### Suboptimal Solutions Caused by Information Asymmetry

In partially observable scenarios requiring global collaboration, IPPO is prone to suboptimal strategies due to the lack of global state information.Local observations cannot fully reflect the true state of the environment (e.g., scenarios involving "coordinated combat between air and ground units" in SMAC). Agents struggle to judge the overall situation, leading to issues such as mismatched unit cooperation and irrational resource allocation.

### Sample Efficiency Still Inferior to Centralized Off-Policy Methods

Although IPPO performs well among on-policy algorithms, it has a significant gap in sample efficiency compared to mainstream off-policy methods (e.g., QMIX, VDAC).The on-policy paradigm requires real-time collection of new samples and cannot reuse historical experiences; in contrast, off-policy methods can store and utilize samples long-term through replay buffers, reducing the cost of environmental interaction.

### Persistent Interference from Environmental Non-Stationarity

Each agent's policy update alters the "training environment" of other agents, leading to the accumulation of value function estimation biases. Although IPPO mitigates this issue through measures like "reducing training epochs" and "value normalization," training is still prone to oscillations in ultra-large-scale agent scenarios (e.g., clusters of 100+ agents).

## Run IPPO in XuanCe

Before running IPPO in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run IPPO directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='ippo',
                    env='mpe',  
                    env_id='simple_spread_v3',  
                    is_test=False)
runner.run() 
```
### Run With Self-defined Configs

If you want to run IPPO with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the IPPO by the following code block:

```python3
import xuance
runner = xuance.get_runner(method='ippo',
                       env='mpe', 
                       env_id='simple_spread_v3',  
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()
```
### Run With Custom Environment

If you would like to run XuanCe's IPPO in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``ippo_myenv.yaml``.

After that, you can run IPPO in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import IPPO_Agents

configs_dict = get_configs(file_dir="ippo_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs) 
Agent = IPPO_Agents(config=configs, envs=envs) 
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

