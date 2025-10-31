# Independent Soft Actor-Critic (ISAC)

**Paper Link:** [**https://arxiv.org/abs/2104.06655**](https://arxiv.org/abs/2104.06655)

Independent Soft Actor-Critic (I-SAC) is a multi-agent deep reinforcement learning algorithm
which consider SAC algorithm as the baseline and fully decentralized as the training method.
Decentralized decision-making avoids exponential growth in action space and thus improves learning efficiency.

This table lists some general features about ISAC algorithm:   

| Features of ISAC                                        | Values | Description                                                                                                   |
|---------------------------------------------------------|--------|---------------------------------------------------------------------------------------------------------------|
| Fully Decentralized                                     | ✅      | There is no communication between agents.                                                                     |
| Fully Centralized                                       | ❌      | Agents send all information to the central controller, and the controller will make decisions for all agents. | 
| Centralized Training With Decentralized Execution(CTDE) | ❌      | The central controller is used in training and abandoned in execution.                                        | 
| On-policy                                               | ❌      | The evaluate policy is the same as the target policy.                                                         | 
| Off-policy                                              | ✅      | The evaluate policy is different from the target policy.                                                      |   
| Model-free                                              | ✅      | No need to prepare an environment dynamics model.                                                             |
| Model-based                                             | ❌      | Need an environment model to train the policy.                                                                | 
| Discrete Action                                         | ✅      | Deal with discrete action space.                                                                              | 
| Continuous Action                                       | ✅      | Deal with continuous action space.                                                                            |

## Key Ideas of ISAC

### Critic update

Critic's target is to minimize TD error which is similar to SAC:

$$
\mathcal{L}_{Q^i}(\phi)=\mathbb{E}_{(o_t,a_t)\sim D}[\frac{1}{2}(Q^i(a_t,o_t)-y_t)^2],\forall i\in\{1,2\}
$$

Where $y_t=r_t+(1-d_t)\gamma\mathbb{E}_{a_{t+1}\sim\pi}[\min_{j\in\{1,2\}}(\hat{Q}^j(a_{t+1},o_{t+1}))-\alpha\log\pi(a_{t+1}|o_{t+1})]$,
$Q^i(a_t,o_t)$ represents the value of Q-network.

### Actor update

The policy is optimized by minimizing a loss function that combines the expected return and the entropy of the policy, encouraging both high reward and sufficient exploration.
The actor loss function is formulated as:

$$
\mathcal{J}_\pi(\theta_i)=\mathbb{E}_{o_t\sim D,a_t\sim\pi}[\alpha\log\pi_{\theta_i}(a_t|o_t)-\min_{j\in\{1,2\}}(Q^j(a_t,o_t))]
$$

Which is similar to SAC.

## Algorithm

The full algorithm for training ISAC is presented in Algorithm 1:

**Algorithm 1: ISAC Algorithm**  
**Input:**
Shared replay buffer $D$ (capacity $N$);
Policy network function $\pi$ (parameters $\theta$);
Critic network function $Q^1$, $Q^2$ (parameters $\phi_1$, $\phi_2$);
Target critics network function $\hat{Q}^1$, $\hat{Q}^2$ initialized with $\hat{\phi}_i = \phi_i$.  
**Output:**
Trained policy $\pi$.

**For** episode = 0,...,M **Do**  
&nbsp;&nbsp;&nbsp;&nbsp;t=0;Initialize state $o_t$  
&nbsp;&nbsp;&nbsp;&nbsp;**While** t < TimeLimit **Do**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**For** agent = 0,...,K **Do**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample action $a_k^t \sim \pi(o_k^t)$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**End For**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Combine actions $a_k^t$ into joint action $a_t$    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Execute $a_t$, observe reward $r_t$, next observation $o_{t+1}$ and done flag $d_{t+1}$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Store $(o_t, a_t, r_t, o_{t+1}, d_{t+1})$ in buffer $D$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**For** e = 0,...,E **Do**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample batch from $D$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute target $y_t=r_t+(1-d_t)\gamma\mathbb{E}_{a_{t+1}\sim\pi}[\min_{j\in\{1,2\}}(\hat{Q}^j(a_{t+1},o_{t+1}))-\alpha\log\pi(a_{t+1}|o_{t+1})]$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update $Q^i$'s weight $\phi_i\leftarrow\phi_i-\omega\nabla \mathcal{L}_{q_i}(\phi),\quad\forall i\in\{1,2\}$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update $\pi$'s weight $\theta\leftarrow\theta-\lambda\nabla \mathcal{J}_{\pi}(\theta)$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update $\hat{Q}^i$'s weight $\hat{\phi}_i \leftarrow \tau \phi_i + (1-\tau)\hat{\phi}_i, \quad\forall i\in\{1,2\}$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**End For**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**If** $d_{t+1}$ **then**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;break  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**End If**  
&nbsp;&nbsp;&nbsp;&nbsp;**End While**  
**End For**

## Run ISAC in XuanCe

Before running ISAC in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run ISAC directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='isac',
                           env='mpe',
                           env_id='simple_spread_v3',
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run ISAC with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the ISAC by the following code block:

```python3
import xuance
runner = xuance.get_runner(method='isac',
                       env='mpe',
                       env_id='simple_spread_v3',
                       config_path="my_config.yaml",
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).


### Run With Custom Environment

If you would like to run XuanCe's ISAC in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``isac_myenv.yaml``.

After that, you can run ISAC in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import ISAC_Agents

configs_dict = get_configs(file_dir="isac_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = ISAC_Agents(config=configs, envs=envs)  # Create a ISAC agent from XuanCe.
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