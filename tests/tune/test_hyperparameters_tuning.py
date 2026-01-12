# Test the tuning tools.

from xuance.common import HyperParameterTuner
import unittest

running_steps = 10000
test_episodes = 2
n_trials = 2

test_algos_drl_discrete = ['dqn', 'ddqn', 'dueldqn', 'noisydqn', 'c51', 'qrdqn', 'drqn',
                             'pg', 'a2c', 'ppo', 'ppg']

test_algos_drl_continuous = ['pg', 'a2c', 'ppo', 'ppg', 'sac', 'ddpg', 'td3']

test_algos_marl_value_decomposition = ['iql', 'vdn', 'qmix', 'wqmix', 'qtran', 'dcg']

test_algos_marl_policy_gradient = ['iddpg', 'maddpg', 'isac', 'masac']

test_algos_marl_on_policy = ['ippo', 'mappo', 'matd3', 'iac', 'coma', 'vdac']

hyperpamameters = ['gamma']


class TestValueBaseAlgo(unittest.TestCase):
    def test_drl_discrete_action_space(self):
        for algo in test_algos_drl_discrete:
            tuner = HyperParameterTuner(algo=algo,
                                        config_path=f'../../examples/{algo}/{algo}_configs/{algo}_cartpole.yaml',
                                        running_steps=running_steps,
                                        test_episodes=test_episodes)
            selected_hyperparameters = tuner.select_hyperparameter(hyperpamameters)
            tuner.tune(selected_hyperparameters, n_trials=n_trials, pruner=None)

    def test_drl_continuous_action_space(self):
        for algo in test_algos_drl_continuous:
            tuner = HyperParameterTuner(algo=algo,
                                        config_path=f'../../examples/{algo}/{algo}_configs/{algo}_pendulum.yaml',
                                        running_steps=running_steps,
                                        test_episodes=test_episodes)
            selected_hyperparameters = tuner.select_hyperparameter(hyperpamameters)
            tuner.tune(selected_hyperparameters, n_trials=n_trials, pruner=None)

    # def test_marl_value_decomposition(self):
    #     for algo in test_algos_marl_value_decomposition:
    #         tuner = HyperParameterTuner(algo=algo,
    #                                     config_path=f'../algo/{algo}_configs/{algo}_cartpole.yaml',
    #                                     running_steps=running_steps,
    #                                     test_episodes=test_episodes)
    #         selected_hyperparameters = tuner.select_hyperparameter(hyperpamameters)
    #         tuner.tune(selected_hyperparameters, n_trials=n_trials, pruner=None)
    #
    # def test_marl_policy_gradient(self):
    #     for algo in test_algos_marl_policy_gradient:
    #         tuner = HyperParameterTuner(algo=algo,
    #                                     config_path=f'../algo/{algo}_configs/{algo}_cartpole.yaml',
    #                                     running_steps=running_steps,
    #                                     test_episodes=test_episodes)
    #         selected_hyperparameters = tuner.select_hyperparameter(hyperpamameters)
    #         tuner.tune(selected_hyperparameters, n_trials=n_trials, pruner=None)
    #
    # def test_marl_on_policy(self):
    #     for algo in test_algos_marl_on_policy:
    #         tuner = HyperParameterTuner(algo=algo,
    #                                     config_path=f'../algo/{algo}_configs/{algo}_cartpole.yaml',
    #                                     running_steps=running_steps,
    #                                     test_episodes=test_episodes)
    #         selected_hyperparameters = tuner.select_hyperparameter(hyperpamameters)
    #         tuner.tune(selected_hyperparameters, n_trials=n_trials, pruner=None)


if __name__ == "__main__":
    unittest.main()
