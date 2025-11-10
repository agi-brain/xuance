# Independent Q-Learning (IQL)

**Paper Link:** [**https://hal.science/file/index/docid/720669/filename/Matignon2012independent.pdf**](https://hal.science/file/index/docid/720669/filename/Matignon2012independent.pdf).

## 1. Introduction
Independent Learners (ILs) in fully cooperative Multi-Agent Systems (MAS) are agents that learn without communication, only through interaction with the environment. The core representative is **Decentralized Q-Learning** (the basic implementation of IQL). Its goal is to enable agents to converge to the global Pareto-optimal Nash equilibrium by independently maintaining local action-value functions. This article, based on the classic paper by Matignon et al. (2012), concisely explains the algorithm, features, and experimental validation of IQL.


## 2. Feature Table of IQL Algorithm
| Features of IQL               | Values | Description                                                                 |
|-------------------------------|-----|-----------------------------------------------------------------------------|
| Fully Decentralized           | ✅   | There is no communication between agents; they learn and decide completely independently. |
| Fully Centralized             | ❌   | There is no centralized controller for unified decision-making. |
| Centralized Training with Decentralized Execution (CTDE) | ❌   | There is no centralized controller in both training and execution phases; it is fully decentralized throughout. |
| On-policy                     | ❌   | The evaluation policy is different from the target policy (IQL based on Q-learning is usually off-policy). |
| Off-policy                    | ✅   | The evaluation policy is different from the target policy, and off-policy techniques such as experience replay can be utilized. |
| Model-free                    | ✅   | No environmental dynamics model is needed; learning is done directly through interaction with the environment. |
| Model-based                   | ❌   | No environmental model is needed to assist policy training. |
| Discrete Action               | ✅   | Suitable for discrete action spaces. |
| Continuous Action             | ❌   | Does not natively support continuous action spaces and needs to be extended with methods such as discretization. |


## 3. Core Coordination Problems of Independent Learners
### 3.1 Pareto-selection Problem
There exist at least two incompatible Pareto-optimal equilibria ($\exists i, \pi_i \neq \hat{\pi}_i$ and $U_{i,<\hat{\pi}_i, \pi_{-i}>}(s) < U_{i,\pi}(s)$), and agents tend to choose non-optimal joint actions.  

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/IQL_Climbing_Penalty.png
    :width: 100%    
    :align: center
    
    Figure 1 : Displays Climbing/Penalty payoffs, aiding Pareto equilibrium understanding.
```

### 3.2 Non-stationarity Problem
The strategies of other agents change dynamically, leading to non-stationary transition probabilities $T(s,a_i,s')$ from the perspective of a single agent, violating the stationary assumption of single-agent RL.

### 3.3 Stochasticity Problem
Environmental noise (such as random rewards) makes it difficult for agents to distinguish whether "reward fluctuations come from the environment or the strategies of other agents".
```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/IQL_Stochastic_Climbing.png
    :width: 100%    
    :align: center
    
    Figure 2 : Displays stochastic Climbing payoffs, aiding environmental stochasticity understanding.
```

### 3.4 Alter-exploration Problem
The exploration actions of a single agent interfere with the learning of other agents. The global exploration rate is $\psi=1-(1-\epsilon)^n$ ($n$ is the number of agents, $\epsilon$ is the individual exploration rate), which can easily trigger a vicious cycle of strategy destruction.

### 3.5 Shadowed Equilibrium Problem
An equilibrium $\bar{\pi}$ is "covered" by a strategy $\hat{\pi}$. There exists an agent $i$ whose payoff from unilaterally deviating from $\bar{\pi}$ is lower than the minimum payoff from deviating from $\hat{\pi}$, leading to the selection of the suboptimal $\hat{\pi}$ (e.g., $<a,a>$ is shadowed by $<c,c>$ in the Climbing Game).


## 4. Core Algorithm: Decentralized Q-Learning
### 4.1 Q-value Update Formula
After agent $i$ takes action $a_i$, it updates $Q_i$ based on the reward $r=R(s,<a_i,a_{-i}>)$ and the next state $s'$: 

$$
Q_{i}(s,a_i) \leftarrow (1-\alpha)Q_i(s,a_i) + \alpha\left( r + \gamma \max_{u \in A_i} Q_i(s',u) \right)
$$

where $\alpha \in [0,1]$ is the learning rate and $\gamma$ is the discount factor.

### 4.2 Exploration Strategies
- **ε-greedy**: Randomly select an action with probability $\epsilon$, and select the action with the maximum current $Q_i$ value with probability $1-\epsilon$;
- **Softmax**: Select actions based on the Boltzmann distribution, with the temperature $\tau$ controlling the degree of exploration:

$$
\pi_i(s,a) = \frac{e^{\frac{Q_i(s,a)}{\tau}}}{\sum_{u \in A_i} e^{\frac{Q_i(s,u)}{\tau}}}
$$  

The paper recommends the **GLIE strategy** (the exploration rate decays to 0 as the number of learning iterations increases), ensuring greedy behavior in the limit.


## 5. Introduction to Related Improved Algorithms
| Algorithm Name               | Core Improvements                                                                 | Solved Core Problems                 |
|------------------------------|-----------------------------------------------------------------------------------|--------------------------------------|
| Distributed Q-Learning       | Optimistic update (only update Q-values to historical maxima) + equilibrium selection (fix strategies with social conventions) | Shadowed equilibrium, alter-exploration |
| Hysteretic Q-Learning        | Dual learning rates ($\alpha>\beta$: use $\alpha$ when Q increases, $\beta$ when Q decreases) | Stochasticity, shadowed equilibrium |
| Recursive FMQ                | Interpolate Q-values based on "maximum reward frequency of actions" to balance optimism and accuracy | Stochasticity, alter-exploration (only applicable to matrix games) |
| WoLF PHC                     | Dual policy learning rates ($\delta_L>\delta_W$: learn fast when losing, learn slow when winning) | Non-stationarity, alter-exploration |


## 6. Experimental Validation and Visualization
### 6.1 Matrix Game Experiments
To verify the convergence of the algorithm in single-state games, the results are shown in the following table:

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/IQL_ConvRate_Games.png
    :width: 100%    
    :align: center
    
    Table 1 : Compares IQL algorithms’ convergence rates in games, showing coordination performance.
```

### 6.2 GLIE Parameter Sensitivity Experiments
Decentralized Q-Learning relies heavily on GLIE strategy parameters (such as $\tau_{ini}$ and $\delta$ of Softmax), and the results are shown in the following table and figure:  

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/IQL_GLIE_Param.png
    :width: 100%    
    :align: center
    
    Table 2 : Shows Decentralized Q-Learning’s convergence under GLIE, reflecting parameter impact.
```

### 6.3 Multi-Agent Extension Experiments
The change in algorithm robustness when the number of agents increases is shown in the following table:
```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/IQL_ConvRate_MultiAgent.png
    :width: 100%    
    :align: center
    
    Table 3 : Shows algorithms’ convergence with different agents, indicating IQL’s robustness.
```

## 7. Analysis of Algorithm Advantages and Disadvantages
### 7.1 Advantages of Decentralized Q-Learning (Basic IQL)
1. Simple implementation: No communication required, only local Q-values need to be maintained;
2. Low overhead: Storage and computation grow linearly with the number of agents;
3. Strong versatility: Applicable to various fully cooperative scenarios (such as multi-robot tasks).

### 7.2 Disadvantages of Decentralized Q-Learning
1. Dependence on GLIE parameters: Difficult to tune, directly affecting convergence;
2. Sensitivity to shadowed equilibria: Extremely low convergence rate in the Climbing Game;
3. Poor resistance to stochasticity: Environmental noise can easily lead to Q-value estimation bias.

## 8. Introduction to Relevant Algorithm Pseudocodes
This paper provides 3 core pseudocodes for IQL and its improvements, serving as direct implementation references.

### 8.1 Distributed Q-Learning Pseudocode 
Implements IQL’s optimistic update: initialization (random policy, $Q_{i,\text{max}}$ = min reward), ε-greedy action selection, optimistic $Q_{i,\text{max}}$ update, equilibrium selection. Applicable to deterministic cooperative Markov games.  

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-IQL-Distributed.png
    :width: 80%
    :align: center
```
Solves IQL’s shadowed equilibrium/alter-exploration; 100% convergence in deterministic games (paper-verified).

### 8.2 WoLF PHC Pseudocode 
Targets IQL’s non-stationarity: initialization, policy-based action selection, TD Q-value update, dual-rate tuning ($\delta_W$ for winning, $\delta_L$ for losing). 

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-IQL-WoLF.png
    :width: 80%
    :align: center
```

Enhances adaptability to others’ strategies; reduces fluctuations from non-stationarity/alter-exploration.

### 8.3 Recursive FMQ Pseudocode
For IQL’s stochasticity in single-state matrix games: initialization, $Q_i$/$F_i$ update, linear interpolation via $F_i$, equilibrium selection. Only for matrix games.  
```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-IQL-Recursive.png
    :width: 80%
    :align: center
```
Distinguishes environmental stochasticity from exploration; 100% convergence in partially stochastic matrix games .

## Run IQL in XuanCe

Before running IQL in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run IQL directly using the following commands:

```python3
import xuance
# Create runner for IQL algorithm
runner = xuance.get_runner(method='iql',
                           env='sc2',  # Choices: sc2, mpe, robotic_warehouse, football, magent2.
                           env_id='3m',  # Choices: 3m, 2m_vs_1z, 8m, 1c3s5z, 2s3z, 25m, 5m_vs_6m, 8m_vs_9m, MMM2, etc.
                           is_test=False)  # False for training, True for testing
runner.run()  # Start running (or runner.benchmark() for benchmarking)
```

### Run With Self-defined Configs

If you want to run IQL with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the IQL by the following code block:

```python3
import xuance as xp
# Create runner for IQL algorithm
runner = xp.get_runner(method='iql',
                       env='sc2',  # Choices: sc2, mpe, robotic_warehouse, football, magent2.
                       env_id='3m',  # Choices: 3m, 2m_vs_1z, 8m, 1c3s5z, 2s3z, 25m, 5m_vs_6m, 8m_vs_9m, MMM2, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)  # False for training, True for testing
runner.run()  # Start running (or runner.benchmark() for benchmarking)
```
To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's IQL in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``iql_myenv.yaml``.

After that, you can run IQL in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_MULTI_AGENT_ENV 
from xuance.environment import make_envs
from xuance.torch.agents.multi_agent_rl.iql_agents import IQL_Agents 

configs_dict = get_configs(file_dir="iql_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = IQL_Agents(config=configs, envs=envs)  # Create a IQL agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@article{matignon2012independent,
  title={Independent Reinforcement Learners in Cooperative Markov Games: A Survey Regarding Coordination Problems},
  author={Matignon, Laetitia and Laurent, Guillaume J and Le Fort-Piat, Nadine},
  journal={The Knowledge Engineering Review},
  volume={27},
  number={1},
  pages={1--31},
  year={2012},
  publisher={Cambridge University Press},
  doi={10.1017/S0269888912000057},
  url={https://hal.science/file/index/docid/720669/filename/Matignon2012independent.pdf}
}
```
