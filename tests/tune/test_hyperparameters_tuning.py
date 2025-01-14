# Test the tuning tools.

from xuance.common import HyperParameterTuner
import unittest

n_steps = 10000
test_episodes = 2
n_trials = 2

test_methods_drl_discrete = ['dqn', 'ddqn', 'dueldqn', 'noisydqn', 'c51', 'qrdqn', 'drqn',
                             'pg', 'a2c', 'ppo', 'ppg']

test_methods_drl_continuous = ['pg', 'a2c', 'ppo', 'ppg', 'sac', 'ddpg', 'td3']

test_methods_marl_value_decomposition = ['iql', 'vdn', 'qmix', 'wqmix', 'qtran', 'dcg']

test_methods_marl_policy_gradient = ['iddpg', 'maddpg', 'isac', 'masac']

test_methods_marl_on_policy = ['ippo', 'mappo', 'matd3', 'iac', 'coma', 'vdac']

hyperpamameters = ['gamma']


class TestValueBaseAlgo(unittest.TestCase):
    def test_drl_discrete_action_space(self):
        for method in test_methods_drl_discrete:
            tuner = HyperParameterTuner(method=method,
                                        config_path=f'../../examples/{method}/{method}_configs/{method}_cartpole.yaml',
                                        running_steps=n_steps,
                                        test_episodes=test_episodes)
            selected_hyperparameters = tuner.select_hyperparameter(hyperpamameters)
            tuner.tune(selected_hyperparameters, n_trials=n_trials, pruner=None)

    def test_drl_continuous_action_space(self):
        for method in test_methods_drl_continuous:
            tuner = HyperParameterTuner(method=method,
                                        config_path=f'../../examples/{method}/{method}_configs/{method}_pendulum.yaml',
                                        running_steps=n_steps,
                                        test_episodes=test_episodes)
            selected_hyperparameters = tuner.select_hyperparameter(hyperpamameters)
            tuner.tune(selected_hyperparameters, n_trials=n_trials, pruner=None)

    # def test_marl_value_decomposition(self):
    #     for method in test_methods_marl_value_decomposition:
    #         tuner = HyperParameterTuner(method=method,
    #                                     config_path=f'../method/{method}_configs/{method}_cartpole.yaml',
    #                                     running_steps=n_steps,
    #                                     test_episodes=test_episodes)
    #         selected_hyperparameters = tuner.select_hyperparameter(hyperpamameters)
    #         tuner.tune(selected_hyperparameters, n_trials=n_trials, pruner=None)
    #
    # def test_marl_policy_gradient(self):
    #     for method in test_methods_marl_policy_gradient:
    #         tuner = HyperParameterTuner(method=method,
    #                                     config_path=f'../method/{method}_configs/{method}_cartpole.yaml',
    #                                     running_steps=n_steps,
    #                                     test_episodes=test_episodes)
    #         selected_hyperparameters = tuner.select_hyperparameter(hyperpamameters)
    #         tuner.tune(selected_hyperparameters, n_trials=n_trials, pruner=None)
    #
    # def test_marl_on_policy(self):
    #     for method in test_methods_marl_on_policy:
    #         tuner = HyperParameterTuner(method=method,
    #                                     config_path=f'../method/{method}_configs/{method}_cartpole.yaml',
    #                                     running_steps=n_steps,
    #                                     test_episodes=test_episodes)
    #         selected_hyperparameters = tuner.select_hyperparameter(hyperpamameters)
    #         tuner.tune(selected_hyperparameters, n_trials=n_trials, pruner=None)


if __name__ == "__main__":
    unittest.main()
