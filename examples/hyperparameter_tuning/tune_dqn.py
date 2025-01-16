from xuance.common import HyperParameterTuner, set_hyperparameters
from optuna.visualization import plot_optimization_history

tuner = HyperParameterTuner(method='dqn',
                            config_path='../dqn/dqn_configs/dqn_cartpole.yaml',
                            running_steps=1000,
                            test_episodes=2)

selected_hyperparameters = tuner.select_hyperparameter(['learning_rate'])

overrides = {
    'learning_rate': [0.001, 0.0001, 0.00001],  # Categorical: Search in the set of {0.001, 0.0001, 0.00001}.
    'gamma': (0.9, 0.99)  # Search from 0.9 to 0.99.
}
selected_hyperparameters = set_hyperparameters(selected_hyperparameters, overrides)

study = tuner.tune(selected_hyperparameters, n_trials=30, pruner=None)
fig = plot_optimization_history(study)
fig.show()
