# DQN with Prioritized Experience Replay (PerDQN)

**Paper Link:** [**https://arxiv.org/pdf/1511.05952**](https://arxiv.org/pdf/1511.05952)

DQN with Prioritized Experience Replay (PER DQN) is a variant of the traditional DQN 
that incorporates Prioritized Experience Replay to improve the agent's learning efficiency 
by prioritizing certain experiences during training.

This table lists some general features about PER DQN algorithm:

| Features of PER DQN | Values | Description                                              |
|---------------------|--------|----------------------------------------------------------|
| On-policy           | ❌      | The evaluate policy is the same as the target policy.    |
| Off-policy          | ✅      | The evaluate policy is different from the target policy. | 
| Model-free          | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based         | ❌      | Need an environment model to train the policy.           | 
| Discrete Action     | ✅      | Deal with discrete action space.                         |   
| Continuous Action   | ❌      | Deal with continuous action space.                       |

## Method

In standard [**DQN**](./dqn_agent.md#deep-q-netowrk), experiences are stored in a replay buffer, 
and the agent samples uniformly from this buffer to train its Q-network. 
However, this uniform sampling can be inefficient, especially when certain experiences are more important for learning than others. 
PER DQN addresses this by prioritizing experiences that are expected to provide more useful information for improving the agent's policy.

### Prioritized Experience Replay

- In Prioritized Experience Replay (PER), instead of sampling uniformly from the buffer, experiences are prioritized based on their temporal-difference (TD) error.
- The TD error is the difference between the expected Q-value (from the Bellman equation) and the current Q-value predicted by the agent's Q-network.
- High TD error means that the experience has high learning potential because it indicates that the agent’s current Q-function is not accurately predicting the future reward for that experience.

### How PER DQN Works

- In PER DQN, the replay buffer is augmented with priority sampling. The priority of an experience is proportional to its TD error.
- When the agent samples experiences for training, those with higher TD errors are more likely to be selected.
- This focuses the agent’s learning on experiences that are more surprising or difficult, accelerating the learning process by revisiting important experiences more frequently.

### Importance Sampling

To avoid biasing the training process due to preferential sampling of experiences, importance sampling is used.
Each experience is assigned a weight that compensates for the non-uniform sampling. 
This ensures that the agent learns correctly even when the experiences are not uniformly sampled.

### Mathematical Details

The priority $p_i$ of experience $i$ is calculated using the TD error $\delta_i$, typically in the form:

$$
p_i = |\delta_i| + \epsilon,
$$

where $\delta_i$ is the absolute value of the TD error, 
$\epsilon$ is a small constant added to ensure that experiences with zero TD error are still included in the buffer.

The probability of sampling experience $i$ is given by:

$$
P(i) = \frac{p^{\alpha}_i}{\sum_{k}{p^{\alpha}_{k}}},
$$

where $\alpha$ controls how much prioritization is used (i.e., how much the TD error affects the sampling probability).

## Algorithm

The full algorithm for training PER DQN is presented in Algorithm 1:

```{eval-rst}
.. image:: ./../../../../_static/figures/pseucodes/pseucode-PERDQN.png
    :width: 88%
    :align: center
```

## Run PER DQN in XuanCe

Before running PER DQN in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run PER DQN directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='perdqn',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run PER DQN with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the PER DQN by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='perdqn',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../configs/configuration_examples.rst).

### Run With Customized Environment

If you would like to run XuanCe's PER DQN in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../../usage/new_envs.rst).
Then, [**prepapre the configuration file**](./../../../usage/new_envs.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``perdqn_myenv.yaml``.

After that, you can run PER DQN in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import PerDQN_Agent

configs_dict = get_configs(file_dir="perdqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = PerDQN_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citations

```{code-block} bash
@inproceedings{DBLP:journals/corr/SchaulQAS15,
  author       = {Tom Schaul and
                  John Quan and
                  Ioannis Antonoglou and
                  David Silver},
  editor       = {Yoshua Bengio and
                  Yann LeCun},
  title        = {Prioritized Experience Replay},
  booktitle    = {4th International Conference on Learning Representations, {ICLR} 2016,
                  San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings},
  year         = {2016},
  url          = {http://arxiv.org/abs/1511.05952},
  timestamp    = {Thu, 25 Jul 2019 14:25:38 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/SchaulQAS15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## APIs

### PyTorch

```{eval-rst}
.. automodule:: xuance.torch.agents.qlearning_family.perdqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### TensorFlow2

```{eval-rst}
.. automodule:: xuance.tensorflow.agents.qlearning_family.perdqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### MindSpore

```{eval-rst}
.. automodule:: xuance.mindspore.agents.qlearning_family.perdqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```
