# Double Deep Q-Network (Double DQN)

Double Deep Q-Network (Double DQN) is an enhancement of the original Deep Q-Network (DQN) algorithm, 
designed to address the issue of overestimation of Q-values, which is a common problem in Q-learning-based methods. 
Overestimation occurs when the Q-learning algorithm selects actions based on 
a maximum Q-value that might be overestimated due to noise or approximation errors. 
This can lead to suboptimal policies and unstable training.

Paper Link: 
[**AAAI**](https://ojs.aaai.org/index.php/AAAI/article/view/10295), 
[**ArXiv**](https://arxiv.org/pdf/1509.06461),
[**Double Q-learnig**](https://proceedings.neurips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)

## The Risk of Overestimating

In standard DQN, overestimation occurs due to the use of a single Q-network for both selecting and evaluating actions. 

As introduced before, [**DQN**](dqn_agent.md) updates the Q-value for a state-action pair $Q(s, a)$ 
by using the maximum of Q-value of the next state $\max_{a'}Q(s', a')$ as part of the target. 

If the Q-network overestimates one or more state-action values, the overestimation propagates and accumulates over time.
This will result in overly optimistic Q-values which can lead to unstable training and cause policy to favor suboptimal actions.

Overestimating biases the agent toward actions that appear better than they are, 
potentially leading to poor decision-making and performance degradation in complex environments.

Besides, if overestimation becomes severe, it can destablize training, causing the Q-values to diverge.

## Main Idea

The main idea of Double DQN is to decouple the action selection and action evaluation processes, 
reducing the risk of overestimating Q-values. 
This is achieved by maintaining two separate Q-value estimation steps:
- **Action Selection**: Use the current Q-network to select the best action for a given state.

$$
a^* = \arg\max_{a}Q(s', s; \theta).
$$

- **Action Evaluation**: Use the target Q-network to evaluate the value of the selected action.

$$
y = r + \gamma Q(s', a*;\theta^{-}).
$$

Then, update the Q-network by minizing the loss:

$$
L(\theta) = \mathbb{E}_{(s, a, s', r) \sim \mathcal{D}}[(y-Q(s, a; \theta))^2].
$$

Finally, don't forget to update the target networks: $\theta^{-} \leftarrow \theta$.

## Run Double DQN in XuanCe

Before running Double DQN in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](https://xuance.readthedocs.io/en/latest/documents/usage/installation.html).

The overall agent-environment interaction of Double DQN, as implemented in XuanCe, is illustrated in the figure below.

```{eval-rst}
.. image:: ./../../../../_static/figures/algo_framework/dqn_framework.png
    :width: 100%
    :align: center
```

### Run Build-in Demos

After completing the installation, you can open a Python console and run DQN directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='ddqn',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run Double DQN with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the Double DQN by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='ddqn',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](https://xuance.readthedocs.io/en/latest/documents/api/configs/configuration_examples.html).

### Run With Customized Environment

If you would like to run XuanCe's Double DQN in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](https://xuance.readthedocs.io/en/latest/documents/usage/new_envs.html#step-1-create-a-new-environment).
Then, [**prepapre the configuration file**](https://xuance.readthedocs.io/en/latest/documents/usage/new_envs.html#step-2-create-the-config-file-and-read-the-configurations) 
``dqn_myenv.yaml``.

After that, you can run Double DQN in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import DDQN_Agent

configs_dict = get_configs(file_dir="dqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = DDQN_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citations

```{code-block} bash
@article{hasselt2010double,
  title={Double Q-learning},
  author={Hasselt, Hado},
  journal={Advances in neural information processing systems},
  volume={23},
  year={2010}
}
```

```{code-block} bash
@inproceedings{van2016deep,
  title={Deep reinforcement learning with double q-learning},
  author={Van Hasselt, Hado and Guez, Arthur and Silver, David},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={30},
  number={1},
  year={2016}
}
```

## APIs

### PyTorch

```{eval-rst}
.. automodule:: xuance.torch.agents.qlearning_family.ddqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### TensorFlow2

```{eval-rst}
.. automodule:: xuance.tensorflow.agents.qlearning_family.ddqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```

### MindSpore

```{eval-rst}
.. automodule:: xuance.mindspore.agents.qlearning_family.ddqn_agent
    :members:
    :undoc-members:
    :show-inheritance:
```