# Targeted Multi-Agent Communication Model (TarMAC)
**Paper Link:** [**https://proceedings.mlr.press/v97/das19a/das19a.pdf**](https://proceedings.mlr.press/v97/das19a/das19a.pdf).

## 1. Architecture Positioning and Core Objectives  
TarMAC (Targeted Multi-Agent Communication) is a multi-agent reinforcement learning (MARL) architecture proposed to address inefficiencies in traditional multi-agent communication. Its core goals are:  
- Achieve **unsupervised targeted communication**: Agents independently learn "who to communicate with" and "what to communicate" solely through task rewards, without additional communication supervision.  
- Support **multi-round interaction**: Adapt to complex tasks (e.g., multi-junction traffic scheduling, 3D navigation) by repeating communication within a single time step to convey complete information.  
- Adopt the **CTDE (Centralized Training with Decentralized Execution) framework**: Use centralized information for stable training and decentralized decision-making for practical deployment.  

## 2. Core Features  

| Features of TarMAC                      | Values | Description                                                                 |
|-----------------------------------------|--------|-----------------------------------------------------------------------------|
| Fully Decentralized                     | ❌      | Relies on targeted message passing for collaboration; full decentralization without interaction fails. |
| Fully Centralized                       | ❌      | No central controller; agents decide decentrally via local observations and messages. |
| Centralized Training with Decentralized Execution (CTDE) | ✅      | Uses centralized info (e.g., agents’ hidden states) for training; executes decentrally with targeted messages. |
| On-policy                               | ✅      | Synchronous batched Actor-Critic (no experience replay); evaluation policy matches target policy. |
| Off-policy                              | ❌      | No experience replay; evaluation policy does not differ from target policy. |
| Model-free                              | ✅      | Learns from agent-environment interaction; no environment dynamics model needed. |
| Model-based                             | ❌      | No environment model required for policy training.                          |
| Discrete Action                         | ✅      | Designed for discrete actions (e.g., "up/down" in SHAPES); supports scalability. |
| Continuous Action                       | ❌      | Primarily for discrete actions; continuous action support is not core.     |

## 3. Overall Architecture   

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/TarMAC_Schematic.png
    :width: 100%    
    :align: center
    
    Figure 1: Schematic of TarMAC’s multi-agent architecture (Left: Agent policy flow; Right: Targeted communication mechanism)
```

The architecture consists of two interconnected, end-to-end differentiable components 

### 3.1 Agent Policy Flow (Left Side)  
At each time step $t$:  
- **Input**: Each agent receives two types of data: ① Local observation $\omega_i^t$ (e.g., 5×5 local grid in SHAPES, 224×224 first-person image in House3D); ② Aggregated message $c_i^t$ from the previous time step.  
- **Core Calculation**: A 1-layer GRU (Gated Recurrent Unit) updates the hidden state $h_i^t$ (encodes the complete history of "observations-actions-messages").  
- **Output**: Two independent outputs are generated from $h_i^t$: ① Discrete environmental action $a_i^t$ (e.g., "accelerate/brake" in traffic junctions); ② Targeted communication message $m_i^t$ (contains signature and value components).  

### 3.2 Targeted Communication Mechanism (Right Side)  
- **Message Sending**: The sender encodes attributes of intended recipients via the "signature $k_i^t$" of $m_i^t$.  
- **Message Receiving**: The receiver generates a query vector $q_j^{t+1}$ from its hidden state $h_j^{t+1}$, calculates attention weights by matching $q_j^{t+1}$ with senders’ signatures, and aggregates relevant messages into $c_j^{t+1}$ (for the next time step).  

## 4. Core Module 1: Targeted Communication  
This is TarMAC’s key innovation, implementing unsupervised targeting through three steps :  

### 4.1 Step 1: Message Construction  
Each agent’s message $m_i^t$ combines a "signature" (for recipient matching) and a "value" (for actual content):  

$$
m_i^t = \left[ \underbrace{k_i^t}_{\text{Signature (Intended Recipient Attributes)}} \parallel \underbrace{v_i^t}_{\text{Value (Actual Communication Content)}} \right].
$$  

- $k_i^t$: 16-dimensional vector (uniform in all experiments) encoding target attributes (e.g., "red target-related" in SHAPES, "west-to-east driving" in traffic junctions).  
- $v_i^t$: 32-dimensional vector (uniform in all experiments) carrying practical information (e.g., agent coordinates in SHAPES, target direction in House3D).  

### 4.2 Step 2: Attention Weight Calculation 
The receiver $j$ generates a $d_k$-dimensional query vector $q_j^{t+1}$ from its hidden state $h_j^{t+1}$ (where $d_k=16$, the signature dimension uniformly set in the file), then calculates the matching degree with the signatures of all senders $k_1^t, k_2^t, ..., k_N^t$, and obtains the attention weight $\alpha_j$ after softmax normalization. The formula is:  

$$
\alpha_j = \text{softmax}\left[ \frac{q_j^{t+1^T}k_1^t}{\sqrt{d_k}}, ..., \frac{q_j^{t+1^T}k_N^t}{\sqrt{d_k}} \right].
$$  

- Numerator: The dot product $q_j^{t+1^T}k_i^t$ reflects the matching degree between the receiver's information demand (encoded in $q_j^{t+1}$) and the sender's message attribute (encoded in $k_i^t$); a larger value indicates a higher relevance, such as a query vector for "searching for red targets" matching a signature vector with "red attribute".  
- Denominator: $\sqrt{d_k}$ (the square root of the signature dimension $d_k$) is used to avoid numerical saturation caused by the dot product of high-dimensional vectors, ensuring the softmax function can flexibly distribute weights, which is consistent with the parameter setting and design logic in the file.

### 4.3 Step 3: Message Aggregation  
The receiver aggregates senders’ value vectors using attention weights to get the next time step’s input message $c_j^{t+1}$:  

$$
c_j^{t+1} = \sum_{i=1}^N \alpha_{ji} v_i^t.
$$  

- Verification: The paper shows this step filters irrelevant messages—e.g., an agent targeting blue assigns weights >0.8 to "blue-attribute" messages and <0.05 to irrelevant ones.  

## 5. Core Module 2: Multi-Round Communication  
Single-round communication fails to handle complex tasks, so TarMAC adds multi-round interaction (within one time step):  

### 5.1 Core Formula  
The hidden state is updated to accumulate multi-round information:  

$$
{h^{\prime}}_j^t=\tanh\left(W_{h\to h^{\prime}}[
\begin{array}
{c}c_j^{t+1}\parallel h_j^t
\end{array}]\right).
$$

- Input: Concatenation of "current aggregated message $c_j^{t+1}$" and "initial hidden state $h_j^t$" (preserves historical and new information).  
- Transformation: $W_{h \to h'}$ (learnable linear matrix) maps to a 128-dimensional GRU hidden state; $\tanh$ constrains values to $[-1, 1]$ to avoid gradient explosion.  

### 5.2 Iteration Rule  
- Optimal Round Count: Experiments confirm 2 rounds work best—1st round conveys general information (e.g., "target is north"), 2nd round refines details (e.g., "detour around the door to the north"). More than 2 rounds provide no performance gain but increase training time.  

## 6. Key Experimental Verification 
Experimental parameters are uniform: RMSProp optimizer (learning rate $7 \times 10^{-4}$), discount factor $\gamma = 0.99$, mean of 5 independent runs.  

### 6.1 Experiment 1: SHAPES 

```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/TarMAC_Timing.png
    :width: 100%    
    :align: center
    
    Figure 2: Timing diagram of agent communication in SHAPES (t=1 to t=21)
```
- Task: 4 agents search for multi-color targets in a 50×50 grid.  
- Core Result: TarMAC’s success rate (85.8±2.5%) outperforms "no communication" (69.1±4.6%) and "communication without attention" (82.4±2.1%).  
- Visual Insight: At $t=2$, agents targeting red focus on those that observed red; at $t=21$, all agents use self-attention after reaching targets.  

### 6.2 Experiment 2: Traffic Junction 
```{eval-rst}
.. figure:: ./../../../_static/figures/algo_framework/TarMAC_Traffic.png
    :width: 100%    
    :align: center
    
    Figure 3: Traffic junction model interpretation (a: Braking probability; b: Attention position; c: Dynamic team adaptation
```

- Task: Cars avoid collisions at 4 two-way junctions.  
- Core Result: 2-round communication (97.1±1.6%) outperforms CommNet (78.9±3.4%) and 1-round communication (84.6±3.2%).  
- Visual Insight: Cars brake near junctions, focus attention on high-risk areas ("right after the first junction"), and adapt to dynamic team sizes (number of attended cars correlates with total cars).

## Run TarMAC in XuanCe

Before running TarMAC in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

```python3
import xuance
# Create runner for TarMAC algorithm
runner = xuance.get_runner(method='tarmac',
                           env='sc2',  # Choices: sc2, mpe
                           env_id='3m',  # Choices: 3m, 2m_vs_1z, 8m, 1c3s5z, 2s3z, 25m, 5m_vs_6m, 8m_vs_9m, MMM2, etc.
                           is_test=False)  # False for training, True for testing
runner.run()  # Start running (or runner.benchmark() for benchmarking)
```

### Run With Self-defined Configs

If you want to run TarMAC with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the TarMAC by the following code block:

```python3
import xuance as xp
# Create runner for TarMAC algorithm
runner = xp.get_runner(method='TarMAC',
                       env='sc2',  # Choices: sc2, mpe
                       env_id='3m',  # Choices: 3m, 2m_vs_1z, 8m, 1c3s5z, 2s3z, 25m, 5m_vs_6m, 8m_vs_9m, MMM2, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)  # False for training, True for testing
runner.run()  # Start running (or runner.benchmark() for benchmarking)
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's TarMAC in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``tarmac_myenv.yaml``.

After that, you can run TarMAC in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_MULTI_AGENT_ENV 
from xuance.environment import make_envs
from xuance.torch.agents.multi_agent_rl.tarmac_agents import TarMAC_Agents 

configs_dict = get_configs(file_dir="TarMAC_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = TarMAC_Agents(config=configs, envs=envs)  # Create a TarMAC agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@InProceedings{das2019tarmac,
  title     = {TarMAC: Targeted Multi-Agent Communication},
  author    = {Das, Abhishek and Gervet, Th{\'e}ophile and Romoff, Joshua and Batra, Dhruv and Parikh, Devi and Rabbat, Mike and Pineau, Joelle},
  booktitle = {International Conference on Machine Learning},
  pages     = {1538--1546},
  year      = {2019},
  publisher = {PMLR},
  pdf       = {http://proceedings.mlr.press/v97/das19a/das19a.pdf},
  url       = {https://proceedings.mlr.press/v97/das19a.html}
}
```