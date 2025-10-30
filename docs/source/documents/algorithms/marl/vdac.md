# Value-Decomposition Actor-Critic (VDAC)

**Paper Link:** [**https://ojs.aaai.org/index.php/AAAI/article/view/17353**](https://ojs.aaai.org/index.php/AAAI/article/view/17353).

## Backround and Motivation 

In Multi - Agent Reinforcement Learning (MARL), effectively coordinating multiple decision - making agents to achieve a global optimal strategy is one of the core challenges. Typical methods can be divided into two paradigms: Value - Based and Policy - Based. In recent years, the framework of Centralized Training with Decentralized Execution (CTDE) has made remarkable progress.

Among them, QMIX, as a typical value - decomposition method, decomposes the joint action - value function Qtot​ into a non - linear combination of local action - values Qa​ of each agent through a monotonic mixing network and performs well in benchmark tests such as the StarCraft Micro - Management (SMAC) task. However, QMIX relies on the off - policy learning paradigm and is difficult to be efficiently integrated into an efficient on - policy framework (such as Advantage Actor - Critic, A2C), which limits its sample efficiency and training efficiency.

On the other hand, although multi - agent Actor - Critic methods such as COMA have good training efficiency advantages, their performance is still significantly lower than that of QMIX, with an obvious performance gap. This contradiction has prompted researchers to explore a new architecture with both high training efficiency and strong policy performance.

To this end, Su et al. (2021) proposed the Value - Decomposition Actor - Critic (VDAC) framework, aiming to bridge the gap between multi - agent Q - learning and Actor - Critic methods and build a unified paradigm that can improve algorithm performance while ensuring training efficiency.


This table lists some general features about VDAC algorithm:

| Features of VDAC   | Values | Description                                              |
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

## Key Ideas

The core innovation of VDAC lies in extending the "value - decomposition" concept from the traditional action - value decomposition to state - value decomposition and embedding it into the Actor - Critic architecture to form a new credit assignment mechanism.
Its theoretical foundation stems from the Difference Rewards mechanism: the learning signal of an individual agent should be positively correlated with its marginal contribution to the global return. VDAC formalizes this concept as the following monotonicity constraint condition:
$$
\frac{\partial V_{tot}(s)}{\partial V^{a}(\tau^{a})} \geq 0, \quad \forall a \in \{1, \ldots, n\}
$$
where:

$V_{tot}(s):$ The global state - value function, which depends on the real state s of the environment.

$V^{a}(\tau^{a}):$ The local state - value function of the a - th agent, which depends on its observation history $\tau^{a}$.

This condition ensures that when the local value of any agent is increased and the strategies of other agents remain unchanged, the overall system value will not decrease, thus achieving effective credit assignment and strategy coordination.

## Method

### VDAC-sum

VDAC - sum is a basic implementation of the VDAC framework. Its core idea is to represent the global state - value $V_{tot}(s)$ as a linear sum of the local state - values $V^a(o^a)$ of each agent. This method follows the following decomposition form:

$$
V_{tot}(s) = \sum_{a = 1}^{n} V^{a}(o^{a})
$$

where s represents the true state of the environment, $o^a$ is the local observation of agent a, and $V^a(o^a)$ is estimated by a distributed critic. This critic shares all parameters except the output layer with the actor network of the corresponding agent. This parameter - sharing mechanism not only reduces the model complexity but also promotes the generalization ability of the policy.
Key Design and Theoretical Basis

Since the weight coefficient is always 1 (a positive number), the above linear combination naturally satisfies the key monotonicity condition required by VDAC:

$$
\frac{\partial V_{tot}(s)}{\partial V^a} \geq 0, \quad \forall a \in \{1, \ldots, n\}
$$

This property ensures that any agent's improvement of its local state - value will not harm the long - term return expectation of the overall system, thus effectively alleviating the credit assignment problem.

Use the least - squares method to optimize the local critic parameters θv​. The loss function is defined as:

$$
L_{t}(\theta_{v}) = (y_{i} - \sum_{a} V_{\theta_{v}}^{a}(o_{t}^{a}))^{2}
$$

Where $y_t = \sum_{i=0}^{k - t - 1} \gamma^i r_i + \gamma^{k - t} V_{tot}(s_k)$ is the target value after n - step bootstrapping.
The policy gradient is calculated based on the simple temporal - difference advantage (TD advantage):

$$
g = \mathbb{E}_\pi \left[ \sum_a \nabla_\theta \log \pi(u^a|\tau^a) \cdot A(s, u) \right], \quad A(s, u) = r + \gamma V'(s') - V(s)
$$

Although the VDAC - sum structure is concise and ensures convergence, its representational ability is limited by the linear assumption and can only approximate a restricted class of centralized state - value functions. Additionally, this variant does not fully utilize the global state information for training, so its performance ceiling is lower than that of models with additional state inputs.

### VDAC-mix

VDAC - mix is an extension of VDAC - sum, aiming to overcome its limited representational ability. This method introduces a non - negative weighted feed - forward neural network (called the mixing network) to fuse the local state values $V^a(o^a)$ of each agent in a non - linear way, thereby generating the global state value $V_{tot}(s):$

$$
V_{tot}(s) = f_{mix}(V^1(o^1), \cdots, V^n(o^n))
$$

Among them, $f_{mix}$​ is a neural network structure whose parameters are dynamically generated by hypernetworks, and its design goal is to approximate any monotonically increasing function.

Distributed critics are still trained by minimizing the prediction error:

$$
L_t(\theta_v) = \left(y_t - f_{\text{mix}}(V_{\theta_v}^1, \cdots, V_{\theta_v}^n)\right)^2
$$

The TD advantage policy gradient formula used in VDAC - sum is also used to ensure that the policy update direction is consistent with the global value improvement.

VDAC - mix significantly enhances the model's representational ability and can capture more complex collaborative patterns. Experiments show that in high - difficulty tasks (such as 3s5z, bane vs bane), the median win rate of VDAC - mix is significantly better than other baseline methods, especially showing higher stability and final performance in comparison with QMIX.

## Framework

The following figure shows the algorithm structure of VDAC-sum.

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/VDACsum_framework.png
    :width: 100%
    :align: center

    Figure 1. VDAC-sum
```
The following figure shows the algorithm structure of VDAC-mix.

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework//VDACvmix_framework.png
    :width: 100%
    :align: center

    Figure 2. VDAC-mix
```

## Run VDAC in XuanCe

Before running VDAC in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run VDAC directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='vdac',
                    env='mpe',  
                    env_id='simple_spread_v3',  
                    is_test=False)
runner.run() 
```
### Run With Self-defined Configs

If you want to run VDAC with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the VDAC by the following code block:

```python3
import xuance
runner = xuance.get_runner(method='vdac',
                       env='mpe', 
                       env_id='simple_spread_v3',  
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()
```
### Run With Custom Environment

If you would like to run XuanCe's VDAC in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``vdac_myenv.yaml``.

After that, you can run VDAC in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import VDAC_Agents

configs_dict = get_configs(file_dir="vdac_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs) 
Agent = VDAC_Agents(config=configs, envs=envs) 
Agent.train(configs.running_steps // configs.parallels)  
Agent.save_model("final_train_model.pth") 
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@inproceedings{su2021value,
  title={Value-decomposition multi-agent actor-critics},
  author={Su, Jianyu and Adams, Stephen and Beling, Peter},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={35},
  number={13},
  pages={11352--11360},
  year={2021}
}
```
