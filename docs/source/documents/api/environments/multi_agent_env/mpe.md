# Multi-Agent Particle Environment

The Multi-Agent Particle Environment (MPE) is a lightweight and customizable simulation environment designed
for multi-agent reinforcement learning (MARL).
It provides a suite of cooperative, competitive, and mixed-scenario tasks
that can be used to test and develop algorithms for learning in multi-agent systems.
The environment emphasizes simplicity and modularity,
making it accessible for researchers to extend and adapt to their specific needs.

[**Official documentation**](https://pettingzoo.farama.org/environments/mpe/#) | 
[**GitHub repository**](https://github.com/openai/multiagent-particle-envs.git) | 
[**Paper**](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)

## Overview

### Tasks gallery

The MPE environment includes nine distinct tasks: "Simple", "Simple Adversary", "Simple Crypto", "Simple Push",
"Simple Reference", "Simple Speaker Listener", "Simple Spread", "Simple Tag", and "Simple World Comm". 
A selection of these tasks is shown below.

```{raw} html
    :file: lists/mpe_list.html
```

### Features

#### Simple Spread

The [Simple Spread](https://pettingzoo.farama.org/environments/mpe/simple_spread) environment has N agents and N landmarks.
The goal of N agents is to navigate to the N landmarks cooperatively while avoiding collisions.
The table below lists some general information about Simple Spread environment.

| env_id       | ``simple_spread_v3``                      |
|--------------|-------------------------------------------|
| Agents       | agents= ['agent_0', 'agent_1', 'agent_2'] |
| Agent number | 3                                         |
| Actions      | Discrete/Continuous                       |
| Action space | Discrete(5)/Box(0.0, 1.0, (5))            |
| Observation  | Box(-inf, inf, (18,))                     |
| State        | Box(-inf, inf, (54,))                     |

Agent observations: [``self_vel``, ``self_pos``, ``landmark_rel_positions``, ``other_agent_rel_positions``, ``communication``],

The actions: [``no_action``, ``move_left``, ``move_right``, ``move_down``, ``move_up``].

**Reward settings:**

Each agent is rewarded based on how close the closest one of them is to the landmarks,
and is penalized when it collides with other agents.

The reward setting of agent $i$ can be described by the formula below:

$$
r_i = -(\min_j{\text{dist}(L_1, A_j)} + \min_j{\text{dist}(L_2, A_j)} + \min_j{\text{dist}(L_3, A_j)}) + p_i,
$$

where, $\text{dist}(L_1, A_j)$ denotes the Euclidean distance between landmark $L_1$ and agent $A_j$,
$p_i$ denotes the penalty of collisions of agent $A_i$.

#### Simple Adversary

The [Simple Adversary](https://pettingzoo.farama.org/environments/mpe/simple_adversary/) environment 
has 1 adversary (red), N agents (green), and N landmarks. By default, N=2.
One of the landmark is the "target" (green), but the agents cannot distinct the target landmark by observing the color.

The goal of agents is navigating to the target landmark via cooperation while avoiding to the adversary.
The goal of adversary is navigating to the target landmark while it doesn't know which one is the target.

The table below lists some general information about Simple Adversary environment.

| env_id                    | ``simple_adversary_v3``                       |
|---------------------------|-----------------------------------------------|
| Agents                    | agents= ['adversary_0', 'agent_0', 'agent_1'] |
| Agent number              | 1 + 2                                         |
| Actions                   | Discrete/Continuous                           |
| Action space              | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation<br/>Adversary | Box(-inf, inf, (8,))                          |
| Observation<br/>Agents    | Box(-inf, inf, (10,))                         |
| State                     | Box(-inf, inf, (28,))                         |

Agents observations: [``self_pos``, ``self_vel``, ``goal_rel_position``, ``landmark_rel_position``, ``other_agent_rel_positions``],

Adversary observation: [``landmark_rel_position``, ``other_agents_rel_positions``],

Agents actions: [``no_action``, ``move_left``, ``move_right``, ``move_down``, ``move_up``].

Adversary actions: [``no_action``, ``move_left``, ``move_right``, ``move_down``, ``move_up``].

**Reward settings:**

Good agents (green) are rewarded based on how close the closest one of them is to the target landmark, 
but negatively rewarded based on how close the adversary is to the target landmark. 

The adversary (red) is rewarded based on distance to the target, but it doesn't know which landmark is the target landmark.

The reward can be described by the formulae below:

$$
\begin{align}
& \text{(Adversary)} & r_{\text{adv}} = -\text{dist}(Adv, L) + p_{\text{adv}}, \\
& \text{(Agent)} & r_i = -\min_j{\text{dist}(Agt^{(j)}, L)} + p_i,
\end{align}
$$

where $\text{dist}(Adv, L)$ denotes the distance between the adversary ($Adv$) and the landmark ($L$),
$\text{dist}(Agt^{(j)}, L)$ denotes the distance between the j-th agent ($Agt^{(j)$) and the landmark ($L$),
$p_{\text{adv}}$ is the adversary's penalty of collision, and $p_i$ is the i-th agent's penalty of collision.

#### Simple Push

The [Simple Push](https://pettingzoo.farama.org/environments/mpe/simple_push/) environment 
has 1 adversary (red), 1 good agent (green), and 1 landmarks.

The goal of good agents is navigating to the target landmark.
The goal of adversary is navigating to the target landmark while push the good agent away from the landmark.

The table below lists some general information about Simple Push environment.

| env_id                    | ``simple_push_v3``                 |
|---------------------------|------------------------------------|
| Agents                    | agents= ['adversary_0', 'agent_0'] |
| Agent number              | 1 + 1                              |
| Actions                   | Discrete/Continuous                |
| Action space              | Discrete(5)/Box(0.0, 1.0, (5))     |
| Observation<br/>Adversary | Box(-inf, inf, (8,))               |
| Observation<br/>Agents    | Box(-inf, inf, (19,))              |
| State                     | Box(-inf, inf, (27,))              |

Good agent observations: [``self_vel``, ``goal_rel_position``, ``goal_landmark_id``, ``all_landmark_rel_positions``, ``landmark_ids``, ``other_agent_rel_positions``],

Adversary observation: [``self_vel``, ``all_landmark_rel_positions``, ``other_agent_rel_positions``],

Agent actions: [``no_action``, ``move_left``, ``move_right``, ``move_down``, ``move_up``].

Adversary actions: [``no_action``, ``move_left``, ``move_right``, ``move_down``, ``move_up``].

**Reward settings:**

The good agent is rewarded based on the distance to the landmark. 
The adversary is rewarded if it is close to the landmark, 
and if the agent is far from the landmark (the difference of the distances). 

The reward can be described by the formulae below:

$$
\begin{align}
& \text{(Adversary)} & r_{\text{adv}} = -\text{dist}(Adv, L) + \text{dist}(Adv, Agt) + p_{\text{adv}}, \\
& \text{(Agent)} & r_{\text{agt}} = -\text{dist}(Agt, L) + p_{\text{agt}},
\end{align}
$$

where $Adv^{(i)}$ denotes the i-th adversary and $Agt^{(j)}$ denotes j-th agent.
$p_i$ is the i-th agent's penalty of collision.

## Installation

MPE is included as part of the PettingZoo library. To install it, run:

```{code-block} bash
pip install pettingzoo[mpe]
```

```{eval-rst}
.. note::

    In XuanCe, the installation of MPE environment is included in the installation of xuance.
    Hence, by default, the library can run MPE tasks directly without the need to install pettingzoo[mpe] solely.
```

To verify your installation, run:

```{code-block} python3
from pettingzoo.mpe import simple_v3

env = simple_v3.parallel_env()
env.reset()
print("MPE environment loaded successfully!")
```

List of available MPE environments included in PettingZoo:

- **simple_v3**: Cooperative navigation.
- **simple_spread_v3**: Cooperative navigation.
- **simple_push_v3**: Cooperative-push task.
- **simple_adversary_v3**: Adversarial navigation.
- **simple_tag_v3**: Predator-prey environment.
- **simple_reference_v3**: Communication task.
- **simple_crypto_v3**: Communication task.
- **simple_speaker_listener**: Communication task.
- **simple_world_comm**: Communication task.


To learn more about MPE, refer to the PettingZoo documentation:
[https://pettingzoo.farama.org/environments/mpe](https://pettingzoo.farama.org/environments/mpe/)

## Citation

Paper published in Neural Information Processing Systems (NeurIPs) 2017 (or NIPS 2017):

```{code-block} bash
@article{lowe2017multi,
      title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
      author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
      journal={Neural Information Processing Systems (NIPS)},
      year={2017}
}
```

Original particle world environment:

```{code-block} bash
@article{mordatch2017emergence,
      title={Emergence of Grounded Compositional Language in Multi-Agent Populations},
      author={Mordatch, Igor and Abbeel, Pieter},
      journal={arXiv preprint arXiv:1703.04908},
      year={2017}
}
```

## APIs

```{eval-rst}
.. automodule:: xuance.environment.multi_agent_env.mpe
    :members:
    :undoc-members:
    :show-inheritance:
```
