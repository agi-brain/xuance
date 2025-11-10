# Individual Controlled Continuous Communication Model (IC3Net)

**Paper Link:** [**https://arxiv.org/pdf/1812.09755**](https://arxiv.org/pdf/1812.09755).

Learning when to communicate and doing that effectively is essential in multi-agent tasks.
In the previous introduction to the [**CommNet**](./commnet.md) algorithm, 
we mention that it uses continuous communication but has been restricted to fully cooperative tasks.

IC3Net algorithm, an advanced version of [**CommNet**](./commnet.md), was developed at the New Year University,
and is a well-known algorithm in the field of multi-agent reinforcement learning.
It was published at ICLR 2019.

This table lists some general features about IC3Net algorithm:

| Features of IC3Net                                       | Values | Description                                                                                                  |
|----------------------------------------------------------|--------|--------------------------------------------------------------------------------------------------------------|
| Fully Decentralized                                      | ❌      | There is no communication between agents.                                                                    |
| Fully Centralized                                        | ❌      | Agents send all information to the central controller and the controller will make decisions for all agents. |
| Centralized Training With Decentralized Execution (CTDE) | ✅      | The central controller is used in training and abandonded in execution.                                      |
| On-policy                                                | ✅      | The evaluate policy is the same as the target policy.                                                        |
| Off-policy                                               | ❌      | The evaluate policy is different from the target policy.                                                     | 
| Model-free                                               | ✅      | No need to prepare an environment dynamics model.                                                            | 
| Model-based                                              | ❌      | Need an environment model to train the policy.                                                               | 
| Discrete Action                                          | ✅      | Deal with discrete action space.                                                                             |   
| Continuous Action                                        | ✅      | Deal with continuous action space.                                                                           |    

## Research Background and Motivation

Multi-Agent Reinforcement Learning (MARL) are increasingly widely used in real-world scenarios, 
from autonomous driving fleet scheduling to intelligent battles in complex games like StarCraft. 
Efficient communication between agents is crucial for achieving collaborative goals. 
However, early multi-agent communication algorithms had three core limitations that severely restricted their adaptability to real scenarios:

### Insufficient Scenario Adaptability

The classic CommNet algorithm, while achieving end-to-end training for continuous communication, 
is only suitable for **fully cooperative scenarios** — agents must share all hidden states and pursue a unified global reward. 
Yet, most real-world scenarios are **mixed cooperative (semi-cooperative)** or **competitive**: 
for example, in a basketball game, teammates need to cooperate but also compete for individual points;
In such scenarios, [CommNet](./commnet.md)’s "undifferentiated full communication" leads to strategy leakage or performance degradation.

### Credit Assignment Dilemma

Traditional algorithms adopt **global average rewards** (all agents receive the same reward), making it impossible to distinguish individual contributions. 
For instance, in a traffic intersection task, if a car causes a collision due to a wrong decision, all cars receive a penalty. 
Agents struggle to identify their own mistakes, resulting in slow training convergence and poor scalability 
— a problem that becomes more prominent as the number of agents increases.

### Lack of Control Over Communication Timing

In early algorithms, agents communicate continuously regardless of scenario needs: "invalid communication" in competitive scenarios may leak strategies 
(e.g., prey exposing its location to predators), while "redundant communication" in cooperative scenarios increases computational overhead. 
In reality, humans "communicate at the right time" based on scenarios (e.g., only passing information at key nodes during team combat), 
but early algorithms lacked this adaptive capability.

## Core Mechanisms of IC3Net

### Basic Model Structure: From Independent Control to Controlled Communication

IC3Net’s basic framework follows a temporal "encoding-communication-decoding" structure, 
with the core being the addition of a "communication gate" and "individual reward optimization" to CommNet. 
First, define key symbols:
- $J$: Number of active agents in the current environment (supports dynamic changes);
- $o_j^t$: Local observation of agent $j$ at time step $t$ (e.g., vehicle position, prey’s field of view);
- $h_j^t, s_j^t$: LSTM hidden state and cell state of agent $j$ at time step $t$ (stores historical information);
- $g_j^t$: Communication gating action of agent $j$ at time step $t$ (binary variable: 1 = communicate, 0 = no communication);
- $c_j^t$: Communication vector received by agent $j$ at time step $t$;
- $r_j^t$: Individual reward of agent $j$ at time step $t$ (distinguished from global reward).

The core process of the model is as follows:

#### Step 1: Observation Encoding and State Update
First, let us describe an independent controller model where each agent is controlled by an individual [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory).
Agents first convert local observations $o_j^t$ into feature vectors via an encoder $e(\cdot)$, then input them into an LSTM to update the hidden state.

For the $j$-th agent:

$$
h_j^{t+1}, s_j^{t+1} = \text{LSTM}\left(e(o_j^t), h_j^t, s_j^t\right)
$$

where hidden state, $h_t$ and cell state, $s_t$ were introduced in LSTM.

The encoder $e(\cdot)$ is a fully connected neural network, and all agents share LSTM parameters to ensure the model’s permutation invariance to agent ordering.

IC3Net extends this independent controller model by allowing agents to communicate their internal state, gated by a discrete action.

The formula above is revised in this form:

$$
h_j^{t+1}, s_j^{t+1} = \text{LSTM}\left(e(o_j^t) + c_j^t, h_j^t, s_j^t\right)
$$

where $c_j^t$ will be introduced in the following text.

#### Step 2: Communication Gating Decision
Agents output the communication gating action $g_j^t$ via a gating network $f^g(\cdot)$ (linear layer + Softmax) to decide whether to transmit information to other agents:

$$
g_j^t \sim \text{Softmax}\left(f^g(h_j^t)\right)
$$

$g_j^t$ is a binary action (obtained through sampling), essentially acting as a "communication switch" for the agent 
— the policy network autonomously learns when to switch it on/off based on scenario benefits.

#### Step 3: Controlled Communication Vector Generation
Unlike CommNet’s "global average broadcast," IC3Net’s communication vector only includes the mean hidden state of "agents that choose to communicate," 
then maps it to a unified dimension via a linear transformation matrix $C$:

$$
c_j^{t+1} = C \cdot \frac{1}{\sum_{j' \neq j} g_{j'}^t} \sum_{j' \neq j} g_{j'}^t \cdot h_{j'}^{t+1}
$$

If no other agents communicate ($\sum_{j' \neq j} g_{j'}^t = 0$), then $c_j^{t+1} = 0$ (no communication signal). 
This design not only avoids redundant information transmission but also adapts to dynamic changes in the number of agents.

#### Step 4: Action Decoding and Individual Reward Optimization
An agent’s action is generated from the hidden state $h_j^t$ via a policy network $\pi(\cdot)$:

$$
a_j^t = \pi(h_j^t)
$$

The training objective is to maximize each agent’s cumulative individual reward $\sum_{t=1}^T r_j^t$, rather than the global reward 
— this is the core solution to the credit assignment problem.

## Algorithm Design: Training Framework and Baseline Comparison

### Training Algorithm: REINFORCE + Shared Parameter Optimization

IC3Net uses the [REINFORCE](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf) algorithm to train the policy network (including action policy $\pi$ and gating policy $f^g$), with the core optimization objective:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=1}^T \nabla_\theta \log \pi(a_j^t | h_j^t; \theta) \cdot \sum_{k=t}^T r_j^k\right]
$$

where $\theta$ represents shared parameters (shared across LSTM, policy network, and gating network) to ensure the model’s permutation invariance to agent ordering.

To reduce training variance, IC3Net introduces a baseline function $b(o_j^t)$ (state value function), modifying the optimization objective to:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=1}^T \nabla_\theta \log \pi(a_j^t | h_j^t; \theta) \cdot \left(\sum_{k=t}^T r_j^k - b(o_j^t)\right)\right]
$$

The baseline function is fitted by an independent value network to further improve training stability.

### Detailed Parameter Optimization Processes for IC3Net

> Feature : Independent Subnetworks + Shared Parameters

#### 1. Parameter Initialization

*Observation Preprocessing*: Convert the local observation $o_j^t$ of each agent $j$ into a feature vector independently using an encoder $e(\cdot)$, 
without concatenating global observations;

*Network Parameter Initialization*: Initialize shared parameters (reused by all subnetworks), including:
  - LSTM parameters: Hidden state dimension of 128, initialize weights for input gate, forget gate, and output gate (all agents share the LSTM structure and parameters);
  - Gating network parameters $f^g(\cdot)$;
  - Communication vector transformation weight \(C\);
  - Policy network $\pi(\cdot)$ and baseline network $b(o_j^t)$;

#### 2. Training Loop Phase 

*Forward Propagation*: Individual Observation → Controlled Communication → Action Output
(The details have been shown in the part of Basic Model Structure).

*Individual Reward and Gradient Calculation*

- **Step 1: Individual Reward Calculation**: Assign independent rewards $r_j^t$ based on the scenario type, which leads to precise credit assignment. 

- **Step 2: Gradient Calculation (REINFORCE + Baseline)**: Calculate gradients for both the "action policy" and "gating policy," based on individual rewards.

  ① Action policy gradient: $- \nabla_\theta \log \pi(a_j^t | h_j^t) \cdot (r_j^t - b(o_j^t))$;  
  ② Gating policy gradient: $- \nabla_\theta \log P(g_j^t | h_j^t) \cdot (r_j^t - b(o_j^t))$, where $P(g_j^t | h_j^t)$ is the probability of the gating action;
  - Baseline network gradient: $\nabla_\theta (r_j^t - b(o_j^t))^2$, minimizing the gap between the baseline and the actual reward to reduce variance.

*Parameter Update*

- **Step 1: Gradient Aggregation**: 
Accumulate the "action gradients + gating gradients + baseline gradients" of $J$ agents $\sum_{j=1}^J \nabla_\theta^{\text{total}}$ to avoid the impact of gradient fluctuations from a single agent.

- **Step 2: Shared Parameter Update**: Update all shared parameters.

## Run IC3Net in XuanCe

Before running IC3Net in XuanCe, you need to prepare a conda environment and install ``xuance``following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run CommNet directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='ic3net',
                           env='mpe',  # Choices: mpe, sc2
                           env_id='simple_spread_v3',  # Choices: simple_spread_v3, etc
                           is_test=False)
runner.run()
```

### Run With Self-defined Configs

If you want to run IC3Net with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the IC3Net by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='ic3net',
                       env='mpe',  # Choices: mpe, sc2
                       env_id='simple_spread_v3',  # Choices: simple_spread_v3, etc
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's IC3Net in your own environment that was not included in XuanCe,
you need to define the new environment following the steps in
[**New Environment Tutorial**](./../../usage/custom_env/custom_marl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_marl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``ic3net_myenv.yaml``.

After that, you can run IC3Net in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import IC3Net_Agents

configs_dict = get_configs(file_dir="ic3net_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = IC3Net_Agents(config=configs, envs=envs)  # Create IC3Net agents from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@article{singh2018learning,
  title={Learning when to communicate at scale in multiagent cooperative and competitive tasks},
  author={Singh, Amanpreet and Jain, Tushar and Sukhbaatar, Sainbayar},
  journal={arXiv preprint arXiv:1812.09755},
  year={2018}
}
```