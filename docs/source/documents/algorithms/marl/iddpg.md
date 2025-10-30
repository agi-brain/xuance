# Independent Deep Deterministic Policy Gradient (IDDPG)

**Paper Link:** [**https://arxiv.org/abs/1706.02275**](https://arxiv.org/abs/1706.02275)

Independent Deep Deterministic Policy Gradient (I-DDPG) is a multi-agent deep reinforcement learning algorithm 
which consider DDPG algorithm as the baseline and fully decentralized as the training method.
It means that multiple single agents are applied to multi-agent environment,
each agent learns independently, and from the perspective of each agent, the decisions of other agents are part of the environment. 

This table lists some general features about IDDPG algorithm:   

| Features of IDDPG                                       | Values | Description                                                                                                   |
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

## Key Ideas of IDDPG

Each agent trains its own actor and critic network independently,
**does not share parameters**, and **does not directly perceive the strategies of other agents**.

### Critic update

Critic's target is to minimize TD error which is similar to DDPG:

$$
\mathcal{L}(\phi_i)=\mathbb{E}_{(o_i,a_i,r_i,\sigma_i^{\prime})\sim\mathcal{D}}\left[\left(Q_i(o_i,a_i;\phi_i)-y_i\right)^2\right]
$$

Where $y_i=r_i+\gamma Q_i^\prime(o_i^\prime,\pi_i^\prime(o_i^\prime;\theta_i^\prime))$, $\phi_i$ represents Q network parameters, $\theta_i^\prime$ represents policy target network parameters.

### Actor update

Actor's policy gradient direction is similar to DDPG:

$$
\nabla_{\theta_i}J(\theta_i)=\mathbb{E}_{o_i\sim\mathcal{D}}\left[\nabla_{\theta_i}\pi_i(o_i;\theta_i)\nabla_{a_i}Q_i(o_i,a_i;\phi_i)|_{a_i=\pi_i(o_i;\theta_i)}\right]
$$

Each agent updates the actor independently through its critic gradient feedback.

## Run IDDPG in XuanCe

Before running IDDPG in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run IDDPG directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='iddpg',
                           env='mpe',
                           env_id='simple_spread_v3',
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run IDDPG with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the IDDPG by the following code block:

```python3
import xuance
runner = xuance.get_runner(method='iddpg',
                       env='mpe',
                       env_id='simple_spread_v3',
                       config_path="my_config.yaml",
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).


### Run With Custom Environment

If you would like to run XuanCe's IDDPG in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``iddpg_myenv.yaml``.

After that, you can run IDDPG in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import IDDPG_Agents

configs_dict = get_configs(file_dir="iddpg_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = IDDPG_Agents(config=configs, envs=envs)  # Create a IDDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
```