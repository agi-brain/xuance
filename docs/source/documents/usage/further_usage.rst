Further Usage
================================

The previous page demonstrated how to directly run an algorithm by calling the runner.
In order to help users better understand the internal implementation process of "XuanCe",
and facilitate further algorithm development and implementation of their own reinforcement learning tasks,
this section will take the PPO algorithm training on the MuJoCo environment task as an example,
and provide a detailed introduction on how to call the API from the bottom level to implement reinforcement learning model training.

.. raw:: html

   <a href="https://colab.research.google.com/github/agi-brain/xuance/blob/master/docs/source/notebook-colab/further_usage.ipynb"
      target="_blank"
      rel="noopener noreferrer"
      style="float: left; margin-left: 0px;">
       <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
   </a>
   <br>

Step 1: Create config file
--------------------------------

A config file should contains the necessary arguments of a PPO agent, and should be a YAML file.
Here we show a config file named "ppo_configs/ppo_mujoco_config.yaml" for MuJoCo environment in gym.

.. code-block:: yaml

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorlayer"
    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # Choices: tensorboard, wandb.
    wandb_user_name: "your_user_name"  # The username of wandb when the logger is wandb.
    render: False # Whether to render the environment when testing.
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    fps: 50  # The frames per second for the rendering videos in log file.
    test_mode: False  # Whether to run in test mode.
    device: "cuda:0"  # Choose an calculating device.
    distributed_training: False  # Whether to use multi-GPU for distributed training.
    master_port: '12355'  # The master port for current experiment when use distributed training.

    agent: "PPO_Clip"  # The agent name.
    env_name: "MuJoCo"  # The environment device.
    env_id: "Ant-v4"  # The environment id.
    env_seed: 1
    vectorize: "DummyVecEnv"  # The vecrized method to create n parallel environments. Choices: DummyVecEnv, or SubprocVecEnv.
    learner: "PPOCLIP_Learner"
    policy: "Gaussian_AC"  # choice: Gaussian_AC for continuous actions, Categorical_AC for discrete actions.
    representation: "Basic_MLP"  # The representation name.

    representation_hidden_size: [256,]  # The size of hidden layers for representation network.
    actor_hidden_size: [256,]  # The size of hidden layers for actor network.
    critic_hidden_size: [256,]  # The size of hidden layers for critic network.
    activation: "leaky_relu"  # The activation function for each hidden layer.
    activation_action: 'tanh'  # The activation function for the last layer of actor network.

    seed: 79811  # The random seed.
    parallels: 16  # The number of environments to run in parallel.
    running_steps: 1000000  # The total running steps for all environments.
    horizon_size: 256  # the horizon size for an environment, buffer_size = horizon_size * parallels.
    n_epochs: 16  # The number of training epochs.
    n_minibatch: 8  # The number of minibatch for each training epoch. batch_size = buffer_size // n_minibatch.
    learning_rate: 0.0004  # The learning rate.

    vf_coef: 0.25  # Coefficient factor for critic loss.
    ent_coef: 0.0  # Coefficient factor for entropy loss.
    target_kl: 0.25  # For PPO_KL learner.
    kl_coef: 1.0  # For PPO_KL learner.
    clip_range: 0.2  # The clip range for ratio in PPO_Clip learner.
    gamma: 0.99  # Discount factor.
    use_gae: True  # Use GAE trick.
    gae_lambda: 0.95  # The GAE lambda.
    use_advnorm: True  # Whether to use advantage normalization.

    use_grad_clip: True  # Whether to clip the gradient during training.
    clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
    grad_clip_norm: 0.5  # The max norm of the gradient.
    use_actions_mask: False  # Whether to use action mask values.
    use_obsnorm: True  # Whether to use observation normalization.
    use_rewnorm: True  # Whether to use reward normalization.
    obsnorm_range: 5  # The range of observation if use observation normalization.
    rewnorm_range: 5  # The range of reward if use reward normalization.

    test_steps: 10000  # The total steps for testing.
    eval_interval: 5000  # The evaluate interval when use benchmark method.
    test_episode: 5  # The test episodes.
    log_dir: "logs/ppo/"  # The main directory of log files.
    model_dir: "models/ppo/"  # The main directory of model files.


Step 2: Get the attributes of the example
----------------------------------------------

This section mainly includes parameter reading, environment creation, model creation, and model training.
First, create a `ppo_mujoco.py` file. The code writing process can be divided into the following steps:

**Step 2.0 Import necessary tools**

.. code-block:: python

    import argparse
    import numpy as np
    from copy import deepcopy
    from xuance.common import get_configs, recursive_dict_update
    from xuance.environment import make_envs
    from xuance.torch.utils.operations import set_seed
    from xuance.torch.agents import PPOCLIP_Agent

**Step 2.1 Get the hyper-parameters of command in console**

Define the following function ``parse_args()``,
which uses the Python package `argparse` to read the command line instructions and obtain the instruction parameters.

.. code-block:: python

    import argparse

    def parse_args():
        parser = argparse.ArgumentParser("Example of XuanCe: PPO for MuJoCo.")
        parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4")

        return parser.parse_args()

**Step 2.2 Get all attributes of the example**

First, the ``parse_args()`` function from Step 2.1 is called to read the command line instruction parameters,
and then the configuration parameters from Step 1 are obtained.

.. code-block:: python

    if __name__ == "__main__":
        parser = parse_args()
        configs_dict = get_configs(file_dir="ppo_configs/ppo_mujoco_config.yaml")
        configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
        configs = argparse.Namespace(**configs_dict)

In this step, the ``get_configs()`` method from "XuanCe" is called.
This method can read the configuration files from the specified directory, and return a dictionary variable.

Then, the ``recursive_dict_update`` method of "XuanCe" is called.
This method can update the configurations of the .yaml file from the ``parser`` variable.

Finally, convert the dictionary variable as ``Namespace`` type.

Step 3: Create environment, PPO Agent, and run the task
--------------------------------------------------------

.. code-block:: python

    import argparse
    import numpy as np
    from copy import deepcopy
    from xuance.common import get_configs, recursive_dict_update
    from xuance.environment import make_envs
    from xuance.torch.utils.operations import set_seed
    from xuance.torch.agents import PPOCLIP_Agent


    def parse_args():
        parser = argparse.ArgumentParser("Example of XuanCe: PPO for MuJoCo.")
        parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4")

        return parser.parse_args()


    if __name__ == "__main__":
        parser = parse_args()
        configs_dict = get_configs(file_dir="ppo_configs/ppo_mujoco_config.yaml")
        configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
        configs = argparse.Namespace(**configs_dict)

        set_seed(configs.seed)
        envs = make_envs(configs)
        Agent = PPOCLIP_Agent(config=configs, envs=envs)

        train_information = {"Deep learning toolbox": configs.dl_toolbox,
                             "Calculating device": configs.device,
                             "Algorithm": configs.agent,
                             "Environment": configs.env_name,
                             "Scenario": configs.env_id}
        for k, v in train_information.items():
            print(f"{k}: {v}")

        Agent.train(configs.running_steps // configs.parallels)
        Agent.save_model("final_train_model.pth")
        print("Finish training!")

        Agent.finish()


After finishing the above three steps, you can run the `python_mujoco.py` file in console and train the model:

.. code-block:: bash

    python ppo_mujoco.py --env-id Ant-v4

The source code of this example can be visited at the following link:

`https://github.com/agi-brain/xuance/blob/master/examples/ppo/ppo_mujoco.py <https://github.com/agi-brain/xuance/blob/master/examples/ppo/ppo_mujoco.py>`_

Distributed training with multi-GPUs
--------------------------------------

XuanCe supports multi-GPU training to maximize GPU resource utilization, enabling more efficient DRL model training.

To train DRL models using multiple GPUs, you need to set ``distributed_training`` to True,
the following parameters are relevant:
- distributed_training (bool): Specifies whether to enable multi-GPU distributed training. Set to True to activate distributed training; otherwise, it remains disabled.
- master_port (int): Defines the master port for the current experiment when distributed training is enabled.`: The master port for current experiment when use distributed training.
