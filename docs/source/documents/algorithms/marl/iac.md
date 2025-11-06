# Independent Actor-Critic (IAC)

**Paper Link:** [**https://ojs.aaai.org/index.php/AAAI/article/view/11794**](https://ojs.aaai.org/index.php/AAAI/article/view/11794).

In the field of Multi - Agent Reinforcement Learning (MARL), Independent Actor - Critic (IAC) is a fundamental and intuitive distributed learning paradigm. It aims to solve decision - making problems in collaborative systems composed of multiple agents. The core idea of the IAC algorithm is to directly generalize the Actor - Critic (AC) framework in a single - agent environment to a multi - agent scenario. Each agent runs a complete AC architecture independently and learns through local experience.

This table lists some general features about IAC algorithm:

| Features of IAC   | Values | Description                                              |
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

## Backround

The key idea of the IAC algorithm stems from the direct generalization of the single - agent Actor - Critic framework, and its design intention is to provide a lightweight learning solution without cross - agent coordination for multi - agent systems. In fully cooperative multi - agent tasks, the IAC algorithm regards each agent as an independent learning individual. By configuring a dedicated Actor (executor) and Critic (evaluator) component for each agent, it realizes a distributed decision - making and value - evaluation process. This design strictly follows the "fully decentralized" paradigm, requiring neither global information sharing during the training phase nor cross - agent communication during the execution phase, greatly reducing the implementation complexity of the algorithm.

## Core Components and Operational Mechanisms

### Independent Actor Network 

 The Actor network of each agent outputs an action distribution based solely on its own action-observation history $\tau^a \in T \equiv (Z \times U)^*$, i.e., the policy $\pi^a(u^a|\tau^a) : T \times U \to [0, 1]$. During the execution phase, each Actor operates completely independently, making decisions relying only on locally observable information, which conforms to the perception constraints of agents in practical scenarios.

### Dedicated Critic Network

The Critic network of each agent is responsible for evaluating the value of its own policy. It typically takes the local observation $z \in Z$ and the executed action $u^a \in U$ as inputs, and outputs the state value $V(s)$ or the advantage function $A(s, a)$, providing an update basis for the corresponding Actor network.

### Parameter Update Mechanism

IAC adopts the classic Actor-Critic update rule. The Critic network optimizes the accuracy of value estimation through Temporal Difference error (TD-error), and the Actor network performs policy gradient ascent based on the value signal provided by the Critic to maximize the expected cumulative reward. In parameter-sharing scenarios, multiple agents can reuse the same set of network parameters while maintaining their independent learning processes.

### Formal expression

Combined with the stochastic game $G = \langle S, U, P, r, Z, O, n, \gamma \rangle$ defined in the COMA paper, the operational scenario of the IAC algorithm can be formally described as follows: $n$ agents $a \in A \equiv \{1, \dots, n\}$ are in the environmental state $s \in S$, obtain local observations based on the observation function $O(s, a) : S \times A \to Z$, form a joint action $\mathbf{u} \in \mathbf{U} \equiv U^n$ by choosing actions through independent policies, and obtain a global reward according to the shared reward function $r(s, \mathbf{u}) : S \times \mathbf{U} \to \mathbb{R}$. Due to the lack of a global information fusion mechanism, the Critic network of each agent can only approximately evaluate the correlation between the global reward and its own action based on local signals.

## Limitations

### Gradient Confusion Caused by the Credit Assignment Problem
In fully cooperative multi-agent tasks, reward signals are usually globally shared, and the action value of a single agent cannot be directly distinguished by the global reward. The IAC algorithm adopts a "each-for-themselves" value assessment mode, where all agents' Critic networks are updated based on the same global reward. This leads to the gradient signals received by the Actor networks being unable to accurately reflect the actual contribution of their own actions to the team's performance, resulting in a "equal-sharing" training dilemma.

The COMA paper further points out that when the number of agents increases, this gradient confusion problem will be significantly exacerbated. Since the gradient update of each agent is interfered by the actions of all other agents, the parameter-sharing IAC model will generate serious noisy gradients, making it difficult for the training process to converge to the optimal joint strategy.

### Suboptimal Solutions Caused by the Lack of Interaction Modeling
The core assumption of the IAC algorithm is that "agent behavior can be regarded as part of the environmental dynamics", that is, each agent equates the actions of other agents to environmental noise and does not need to actively model interaction relationships. This assumption has obvious flaws in dynamic and complex collaborative scenarios: agents cannot predict the action intentions of their teammates, leading to a significant decrease in the coordination of joint actions.

For example, in multi-agent tasks that require division of labor and cooperation, the IAC algorithm may cause multiple agents to repeatedly perform the same effective action or miss key collaborative steps. The COMA paper confirms through comparative experiments that this characteristic of lacking interaction modeling makes the IAC algorithm only converge to suboptimal solutions in most cooperative tasks, and its performance drops sharply as the number of agents increases.

### Value Estimation Bias Caused by Observational Limitations
The Critic network of the IAC algorithm only relies on the local observations of the agent for value assessment. In partially observable scenarios, local observations cannot fully reflect the true state $s \in S$ of the environment. This partial observability leads to systematic biases in the value estimation of the Critic network, which in turn misleads the policy update direction of the Actor network.

Compared with the centralized Critic adopted by the COMA algorithm (which takes the global state or joint action-observation history as input), the decentralized Critic of IAC lacks a global information correction mechanism and is difficult to accurately estimate the joint action value $Q(s, \mathbf{u})$. This causes the error in the calculation of the advantage function to accumulate continuously, ultimately affecting the training stability.

## Run IAC in XuanCe

Before running IAC in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run IAC directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='iac',
                    env='mpe',  
                    env_id='simple_spread_v3',  
                    is_test=False)
runner.run() 
```
### Run With Self-defined Configs

If you want to run IAC with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the IAC by the following code block:

```python3
import xuance
runner = xuance.get_runner(method='iac',
                       env='mpe', 
                       env_id='simple_spread_v3',  
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()
```
### Run With Custom Environment

If you would like to run XuanCe's IAC in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``iac_myenv.yaml``.

After that, you can run IAC in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import IAC_Agents

configs_dict = get_configs(file_dir="iac_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs) 
Agent = IAC_Agents(config=configs, envs=envs) 
Agent.train(configs.running_steps // configs.parallels)  
Agent.save_model("final_train_model.pth") 
Agent.finish()  # Finish the training.
```


## Citation

```{code-block} bash
@inproceedings{foerster2018counterfactual,
  title={Counterfactual multi-agent policy gradients},
  author={Foerster, Jakob and Farquhar, Gregory and Afouras, Triantafyllos and Nardelli, Nantas and Whiteson, Shimon},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={32},
  number={1},
  year={2018}
}
```
