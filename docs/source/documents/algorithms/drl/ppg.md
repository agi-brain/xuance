# Phasic Policy Gradient (PPG)

**Paper Link:** [**https://proceedings.mlr.press/v139/cobbe21a**](https://proceedings.mlr.press/v139/cobbe21a)

The Phasic Policy Gradient (PPG) algorithm is an advanced reinforcement learning method designed to improve the efficiency of policy optimization. 
It builds upon the 
[**PPO**](ppoclip_agent.md) 
framework by introducing a two-phase training approach, 
which decouples the policy optimization from auxiliary value function learning.

| Features of PG    | Values | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| On-policy         | ✅      | The evaluate policy is the same as the target policy.    |
| Off-policy        | ❌      | The evaluate policy is different from the target policy. | 
| Model-free        | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based       | ❌      | Need an environment model to train the policy.           | 
| Discrete Action   | ✅      | Deal with discrete action space.                         |   
| Continuous Action | ✅      | Deal with continuous action space.                       |

## Method

In traditional PPO, the value function is used as a baseline for the policy gradient and is trained jointly with the policy. 
However, this entanglement can limit the effectiveness of the learning process, 
as the policy optimization might interfere with the value function's learning and vice versa.

PPG addresses this limitation by introducing phased learning, 
separating policy optimization and value function learning into distinct stages.

PPG works in two phases:

**Policy Phase**:
- Optimize the policy using PPO, focusing on maximizing the reward.
- The value function acts as a baseline for policy optimization but isn't trained during this phase.

During the policy phase, the policy $\pi_{\theta}$ is optimized using the standard PPO objective:

$$
L_{PPO}(\theta) = \mathbb{E}[\min{r_t(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t}],
$$

where:
- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio; 
- $A_t$ is the advantage estimate;
- $\epsilon$ is a clipping parameter.

**Auxiliary Phase**:
- Train the value function using auxiliary loss terms.
- This phase ensures that the value function accurately predicts returns, improving its utility as a baseline.

In the auxiliary phase, the value function $V_{\phi}$ is updated to minimize the auxiliary loss:

$$
L_{aux}(\phi) = \mathbb{E}[(V_{\phi}(s_t) - R_t)^2 + \beta \cdot L_{consistency}],
$$

where:
- $R_t$ is the target return;
- $L_{consistency}$ enforces consistency between the policy's actions and value predictions.
- $\beta$ is weight for the consistency term.

These phases alternate, 
enabling the value function and policy to learn more effectively without directly interfering with each other.

## Run PPG in XuanCe

Before running PPG in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run PPG directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='ppg',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run PPG with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the PPG by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='ppg',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's PPG in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``ppg_myenv.yaml``.

After that, you can run PPG in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import PPG_Agent

configs_dict = get_configs(file_dir="ppg_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = PPG_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@inproceedings{cobbe2021phasic,
  title={Phasic policy gradient},
  author={Cobbe, Karl W and Hilton, Jacob and Klimov, Oleg and Schulman, John},
  booktitle={International Conference on Machine Learning},
  pages={2020--2027},
  year={2021},
  organization={PMLR}
}
```
