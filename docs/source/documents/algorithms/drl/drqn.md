# Deep Recurrent Q-Network (DRQN)

**Paper Link:** [**https://cdn.aaai.org/ocs/11673/11673-51288-1-PB.pdf**](https://cdn.aaai.org/ocs/11673/11673-51288-1-PB.pdf)

The Deep Recurrent Q-Network (DRQN) is an extension of the 
[**DQN**](dqn.md) designed to handle partially observable environments. 
Unlike DQN, which relies on fully observable 
[Markov Decision Processes (MDPs)](https://en.wikipedia.org/wiki/Markov_decision_process), 
DRQN incorporates recurrent neural networks (RNNs) to manage 
[Partially Observable Markov Decision Processes (POMDPs)](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process), 
where the agent does not have access to the full state of the environment.

This table lists some general features about DRQN algorithm:

| Features of DRQN  | Values | Description                                              |
|-------------------|--------|----------------------------------------------------------|
| On-policy         | ❌      | The evaluate policy is the same as the target policy.    |
| Off-policy        | ✅      | The evaluate policy is different from the target policy. | 
| Model-free        | ✅      | No need to prepare an environment dynamics model.        | 
| Model-based       | ❌      | Need an environment model to train the policy.           | 
| Discrete Action   | ✅      | Deal with discrete action space.                         |   
| Continuous Action | ❌      | Deal with continuous action space.                       |

## Architecture

DRQN replaces the fully connected layers of DQN with an RNN (commonly an LSTM or GRU layer).

| Gated Recurrent Units (GRUs)                                                         | Long Short-Term Memory (LSTM)                                                          |
|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| <img src="../../../../_static/figures/algo_framework/GRU.png" alt="GRU" width="300"> | <img src="../../../../_static/figures/algo_framework/LSTM.png" alt="LSTM" width="300"> |

This allows the network to maintain a memory of past observations, enabling it to infer the hidden state of the environment.

Instead of relying solely on the current observation, 
DRQN uses a sequence of observations (a history) to predict Q-values.
The RNN processes the observation sequence and outputs Q-values for each action, as in standard DQN.

Differences between DQN and DRQN:

| Feature           | DQN                       | DRQN                           |
|-------------------|---------------------------|--------------------------------|
| Network Type      | Feedforward CNN/FC layers | Recurrent (LSTM/GRU)           |
| Observation Input | Single observation        | Sequence of observations       |
| Use Case          | Fully observable MDPs     | Partially observable POMDPs    |
| Memory Mechanism  | No memory                 | Captures temporal dependencies |

## Run DRQN in XuanCe

Before running DRQN in XuanCe, you need to prepare a conda environment and install ``xuance`` following 
the [**installation steps**](./../../usage/installation.rst#install-xuance).

### Run Build-in Demos

After completing the installation, you can open a Python console and run DRQN directly using the following commands:

```python3
import xuance
runner = xuance.get_runner(method='drqn',
                           env='classic_control',  # Choices: claasi_control, box2d, atari.
                           env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                           is_test=False)
runner.run()  # Or runner.benchmark()
```

### Run With Self-defined Configs

If you want to run DRQN with different configurations, you can build a new ``.yaml`` file, e.g., ``my_config.yaml``.
Then, run the DRQN by the following code block:

```python3
import xuance as xp
runner = xp.get_runner(method='drqn',
                       env='classic_control',  # Choices: claasi_control, box2d, atari.
                       env_id='CartPole-v1',  # Choices: CartPole-v1, LunarLander-v2, ALE/Breakout-v5, etc.
                       config_path="my_config.yaml",  # The path of my_config.yaml file should be correct.
                       is_test=False)
runner.run()  # Or runner.benchmark()
```

To learn more about the configurations, please visit the 
[**tutorial of configs**](./../../api/configs/configuration_examples.rst).

### Run With Custom Environment

If you would like to run XuanCe's DRQN in your own environment that was not included in XuanCe, 
you need to define the new environment following the steps in 
[**New Environment Tutorial**](./../../usage/custom_env/custom_drl_env.rst).
Then, [**prepapre the configuration file**](./../../usage/custom_env/custom_drl_env.rst#step-2-create-the-config-file-and-read-the-configurations) 
 ``drqn_myenv.yaml``.

After that, you can run DRQN in your own environment with the following code:

```python3
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import DRQN_Agent

configs_dict = get_configs(file_dir="drqn_myenv.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = MyNewEnv

envs = make_envs(configs)  # Make parallel environments.
Agent = DRQN_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
```

## Citation

```{code-block} bash
@inproceedings{hausknecht2015deep,
  title={Deep recurrent q-learning for partially observable mdps},
  author={Hausknecht, Matthew and Stone, Peter},
  booktitle={2015 aaai fall symposium series},
  year={2015}
}
```
