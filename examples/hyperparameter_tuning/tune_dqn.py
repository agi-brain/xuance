from xuance.common import HyperParameterTuner, set_hyperparameters


tuner = HyperParameterTuner(method='dqn',
                             config_path='./examples/dqn/dqn_configs/dqn_cartpole.yaml',
                             running_steps=1000,
                             test_episodes=2)

selected_hyperparameters = tuner.select_hyperparameter(['learning_rate'])

overrides = {
    'learning_rate': [0.001, 0.0001, 0.00001]
}
selected_hyperparameters = set_hyperparameters(selected_hyperparameters, overrides)

tuner.tune(selected_hyperparameters, n_trails=5, pruner=None)
