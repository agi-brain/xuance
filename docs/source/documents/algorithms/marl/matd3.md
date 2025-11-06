# Multi-agent Twin Delayed Deep Deterministic Policy Gradient (MATD3)

**Paper Link:** [**https://arxiv.org/abs/1910.01465**](https://arxiv.org/abs/1910.01465)

Multi-agent Twin Delayed Deep Deterministic Policy Gradient (MATD3), extends TD3 to the multi-agent domain
in a similar manner to the extension of DDPG to MADDPG.
Its core goal is to reduce the overestimation bias of centralized value function in multi-agent reinforcement learning.

This table lists some general features about MATD3 algorithm:   

| Features of MATD3                                       | Values | Description                                                                                                   |
|---------------------------------------------------------|--------|---------------------------------------------------------------------------------------------------------------|
| Fully Decentralized                                     | ❌      | There is no communication between agents.                                                                     |
| Fully Centralized                                       | ❌      | Agents send all information to the central controller, and the controller will make decisions for all agents. | 
| Centralized Training With Decentralized Execution(CTDE) | ✅      | The central controller is used in training and abandoned in execution.                                        | 
| On-policy                                               | ❌      | The evaluate policy is the same as the target policy.                                                         | 
| Off-policy                                              | ✅      | The evaluate policy is different from the target policy.                                                      |   
| Model-free                                              | ✅      | No need to prepare an environment dynamics model.                                                             |
| Model-based                                             | ❌      | Need an environment model to train the policy.                                                                | 
| Discrete Action                                         | ❌      | Deal with discrete action space.                                                                              | 
| Continuous Action                                       | ✅      | Deal with continuous action space.                                                                            |

## Key Ideas of MATD3

### Target Policy Smoothing

Adding clipped Gaussian noise $\epsilon=\mathrm{clip}(\mathcal{N}(0,\sigma),-c,c)$ to the actions of all agents in the critic update:
$a_{j}^{\prime}=\mu_{\theta_{j}^{\prime}}(o_{j}^{\prime})+\epsilon$.Finally,the target function of the critic is expressed as:

$$
y_i=r_i+\gamma\min_{j=1,2}Q_{i,\theta_j^{\prime}}^\pi(\mathbf{x}^{\prime},\mu_1^{\prime}(o_1^{\prime})+\epsilon,...,\mu_N^{\prime}(o_N^{\prime})+\epsilon)
$$

Where $\mu_j^{\prime}$ being short for $\mu_{\theta_{j}^{\prime}}$.

### Critic update

The critic loss function can be expressed as :

$$
L(\theta_i)=\mathbb{E}_{x,a,r,x^{\prime}}\left[\left(Q_{i,\theta_j}^{\pi} \left(x,a_1,\ldots,a_N\right)-y_i\right)^2\right]
$$

Where $i$ expresses each agent.

### Actor update

The deterministic policy of agent $i$ can be optimized by gradient descent:

$$
\nabla_{\theta_i}J\left(\mu_i\right)=\mathbb{E}_{x,a\sim D}\left[\nabla_{\theta_i}\mu_i\left(a_{i}\left|o_{i}\right)\nabla_{a_i}Q_{i,\theta_j}^\mu\left(x,a_1,\ldots,a_N\left|\right._{a_i=\mu_i(o_i)}\right)\right]\right.
$$

Where $\mu_i$ being short for $\mu_{\theta_{i}}$.

## Algorithm

The full algorithm for training MATD3 is presented in Algorithm 1:

```{eval-rst}
.. image:: ./../../../_static/figures/pseucodes/pseucode-MATD3.png
    :width: 80%
    :align: center
```

## Run MATD3 in XuanCe

Before running MATD3 in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run MATD3 directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='matd3',
                           env='mpe',
                           env_id='simple_spread_v3',
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run MATD3 with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the MATD3 by the following code block:

```python3
import xuance
runner = xuance.get_runner(method='matd3',
                       env='mpe',
                       env_id='simple_spread_v3',
                       config_path="my_config.yaml",
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).


### Run With Custom Environment

If you would like to run XuanCe's MATD3 in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``matd3_myenv.yaml``.

After that, you can run MATD3 in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import MATD3_Agents

configs_dict = get_configs(file_dir="matd3_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = MATD3_Agents(config=configs, envs=envs)  # Create a MATD3 agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```


## Citation

```{code-block} bash
@misc{ackermann2019reducingoverestimationbiasmultiagent,
      title={Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics}, 
      author={Johannes Ackermann and Volker Gabler and Takayuki Osa and Masashi Sugiyama},
      year={2019},
      eprint={1910.01465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1910.01465}, 
}

```