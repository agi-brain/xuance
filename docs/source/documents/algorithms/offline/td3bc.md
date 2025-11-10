# Twin Delayed Deep Deterministic Policy Gradient with Behavior Cloning (TD3BC)

**Paper Link:** [**https://proceedings.neurips.cc/paper_files/paper/2021/file/a8166da05c5a094f7dc03724b41886e5-Paper.pdf**](https://proceedings.neurips.cc/paper_files/paper/2021/file/a8166da05c5a094f7dc03724b41886e5-Paper.pdf)

Twin Delayed Deep Deterministic Policy Gradient with Behavior Cloning (TD3BC) is an enhancement of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm, designed to address the issues of sample inefficiency and training instability in multi-agent reinforcement learning (MARL) scenarios. TD3BC introduces three key improvements by integrating behavior cloning (BC) with TD3’s framework, policy update objective, state processing and hyper-parameter setting.

This table lists some general features about TD3BC algorithm:


| Features of MFQ   | Values | Description                                              |
| ------------------- | -------- | ---------------------------------------------------------- |
| On-policy         | ❌     | The evaluate policy is the same as the target policy.    |
| Off-policy        | ✅     | The evaluate policy is different from the target policy. |
| Model-free        | ✅     | No need to prepare an environment dynamics model.        |
| Model-based       | ❌     | Need an environment model to train the policy.           |
| Discrete Action   | ❌     | Deal with discrete action space.                         |
| Continuous Action | ✅     | Deal with continuous action space.                       |

## TD3 Framework

The [**TD3**](https://arxiv.org/abs/1802.09477) algorithm is an improvement over the [**DDPG**](https://arxiv.org/abs/1509.02971) algorithm:

- **Double Critic Networks**:To address the issue of Q-value overestimation, the minimum value from the two networks is taken as the target Q-value each time:

$$
y = r + \gamma \min_{i=1,2} Q_{\theta_i'}(s', \pi_{\phi_1}(s'))
$$

- **Delayed Policy Updates**: Recognizing the importance of target networks in reducing error accumulation and the negative effects of high variance estimates on policy updates, the TD3 algorithm delays policy updates.
- **Target Policy Smoothing Regularization**:To address the problem that deterministic policies may overfit to narrow peaks in the value estimate, by adding Gaussian noise to the target policy, the actions are smoothed:

$$
a' = \pi_{\phi'}(s') + \epsilon
\\ \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)
$$

When calculating the target value , the minimum of the two critics’ estimates is taken, that is $y = r + \gamma \min_{i=1,2} Q_{\theta_i'}(s', \pi_{\phi'}(s') + \epsilon)$ where $\epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$.

## Key Idea Of TD3BC

- **Policy Update Objective**:TD3’s policy $\pi$ is updated with the deterministic policy gradient([**DPG**](https://proceedings.mlr.press/v32/silver14.html)): $\pi = \argmax_{\pi} \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ Q(s, \pi(s)) \right]$, TD3+BC incorporates a behavior cloning term into the original policy as a regularization term. With a single hyperparameter $\lambda$ to control the strength of the regularizer.:

$$
\pi = \argmax_{\pi} \mathbb{E}_{s \sim \mathcal{D}} \left[ Q(s, \pi(s)) \right] \rightarrow \pi = \argmax_{\pi} \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \lambda \, Q(s, \pi(s)) - \left( \pi(s) - a \right)^2 \right]
$$

- **State Processing**:TD3+BC performs normalization on states, standardizing them to have a mean of 0 and a standard deviation of 1, $s_{i}=\frac{s_{i}-\mu_{i}}{\sigma_{i}+\epsilon}$, where  $\epsilon$ is a small normalization constant. This helps improve the generalization ability and training stability of the network across different environments.
- **Hyper-Parameter Setting**:$\lambda$ is essentially a hyperparameter, but the balance between RL (maximizing the Q-value) and imitation learning (minimizing the behavior cloning term) highly depends on the scale of the Q-value. Therefore, a normalization term can be added to $\lambda$, which is essentially a normalization term based on the average absolute value of the Q-value:

$$
\lambda = \frac{\alpha}{\frac{1}{N} \sum_{(s_i, a_i)} \left| Q(s_i, a_i) \right|}
$$

The above are all the changes that TD3+BC has made to TD3, and can be implemented by modifying only a handful of lines in most codebases.

## Run TD3BC in XuanCe

Before running TD3BC in XuanCe, you need to prepare a conda environment and install ``xuance`` following
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run TD3BC directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='td3bc',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's TD3BC in your own environment that was not included in XuanCe,
you need to define the new environment following the steps in
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations)
``td3bc_myenv.yaml``.

After that, you can run TD3BC in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import TD3_BC_Agent

configs_dict = get_configs(file_dir="mfq_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = TD3_BC_Agent(config=configs, envs=envs)  # Create a TD3BC agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citations

```{code-block} bash
@inproceedings{
  fujimoto2021a,
  title={A Minimalist Approach to Offline Reinforcement Learning},
  author={Fujimoto, Scott and Gu, Shixiang (Shane)},
  booktitle={Advances in Neural Information Processing Systems 34 (NeurIPS 2021)},
  year={2021},
  url={https://proceedings.neurips.cc/paper_files/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html},
}
```
