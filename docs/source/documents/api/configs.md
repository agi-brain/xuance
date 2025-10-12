# Configs

```{eval-rst}
.. toctree::
    :hidden:

    Basic Configurations <configs/basic_configurations>
    Configuration Examples <configs/configuration_examples>
    Custom Configurations <configs/custom_configurations>
```

- [Basic Configurations](configs/basic_configurations)
- [Configuration Examples](configs/configuration_examples)
- [Custom Configurations](configs/custom_configurations)

XuanCe provides a structured way to manage configurations for various DRL/MARL scenarios,
making it easy to experiment with different setups.

## Arguments setting tutorial

| Argument                   | Description                                                                                      | Choices/Type                                                                                    |
|----------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| agent                      | The name of the agent.                                                                           | DQN, PPO, etc                                                                                   |
| policy                     | The name of the policy.                                                                          | Basic_Q_network, <br/>Categorical_AC, etc.                                                      |
| learner                    | The name of the learner.                                                                         | DQN_Learner, <br/>PPO_Learner, etc.                                                             |
| representation             | The name of the representation.                                                                  | Basic_Identical, <br/>Basic_MLP, <br/>Basic_CNN, <br/>Basic_RNN, etc.                           |
| env_name                   | The name of the environment.                                                                     | Classic Control, <br/>Box2D, etc.                                                               |
| env_id                     | The environment id.                                                                              | 'CartPole-v1', <br/>'Ant-v4', etc.                                                              |
| env_seed                   | The environment seed.                                                                            | int                                                                                             |
| vectorize                  | The vectorization method for environments.                                                       | DummyVecEnv, <br/>DummyVecMultiAgentEnv, <br/>SubprocVecEnv, <br/>SubprocVecMultiAgentEnv, etc. |
| parallels                  | The number of environments that run in parallel.                                                 | int                                                                                             |
| representation_hidden_size | The hidden units for representation module.                                                      | List of int, <br/>e.g., [64, 64]                                                                |
| activation                 | The activation method for each hidden layer.                                                     | 'relu', <br/>'sigmoid', <br/>'leaky_relu', etc.                                                 |
| seed                       | Random seed for initializing the networks.                                                       | int                                                                                             |
| buffer_size                | Size of the replay buffer.                                                                       | int                                                                                             |
| batch_size                 | Batch size for one-step training.                                                                | int                                                                                             |
| learning_rate              | The learning rate to update the networks.                                                        | float32                                                                                         |
| gamma                      | The discount factor.                                                                             | float32                                                                                         |
| start_greedy               | The initialized greedy for selecting actions.                                                    | float32                                                                                         |
| end_greedy                 | The final greedy for selecting actions.                                                          | float32                                                                                         |
| decay_step_greedy          | The steps for the process of greedy decay.                                                       | int                                                                                             |
| sync_frequency             | The synchronization frequency for target networks.                                               | int                                                                                             |
| training_frequency         | The training period.                                                                             | int                                                                                             |
| running_steps              | The total running steps for the experiment.                                                      | int                                                                                             |
| start_training             | When to start training the networks.                                                             | int                                                                                             |
| use_grad_clip              | Whether to use the gradient clip when do gradient descent.                                       | bool                                                                                            |
| grad_clip_norm             | The gradient normalization when use_grad_clip is True.                                           | float32                                                                                         |
| use_action_mask            | Whether to use the action masks when the environment contains some actions that are unavailable. | bool (default is False)                                                                         |
| use_obsnorm                | Whether to use observation normalization trick.                                                  | bool                                                                                            |
| obsnorm_range              | The range of normalized observatinos.                                                            | float                                                                                           |
| use_rewnorm                | Whether to use the reward normalization trick.                                                   | bool                                                                                            |
| rewnorm_range              | The range of normalized rewards.                                                                 | float                                                                                           |
| test_steps                 | The steps for testing the model.                                                                 | int                                                                                             |
| eval_interval              | The interval steps for evaluating the model during training.                                     | int                                                                                             |
| test_episode               | The episodes for evaluating the model during training.                                           | int                                                                                             |
| log_dir                    | The directory for saving the logger file.                                                        | str                                                                                             |
| model_dir                  | The directory for saving the model.                                                              | str                                                                                             |


