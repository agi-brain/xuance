import optuna
from copy import deepcopy
from argparse import Namespace
from xuance.common import MultiObjectiveTuner, set_hyperparameters, List
from xuance.common.tuning_tools import build_search_space, Hyperparameter
from xuance.environment import make_envs
from optuna.visualization import plot_pareto_front


class MyMultiObjectiveTuner(MultiObjectiveTuner):
    def __init__(self, method='dqn',
                 config_path='../dqn/dqn_configs/dqn_cartpole.yaml',
                 running_steps=10000,
                 test_episodes=2):
        super().__init__(method, config_path, running_steps, test_episodes)

    def objective(self, trail: optuna.trial, selected_hyperparameters: List[Hyperparameter]):
        search_space = build_search_space(trail, selected_hyperparameters)
        config_trail = deepcopy(self.configs_dict)
        config_trail.update(search_space)
        configs_trail = Namespace(**config_trail)
        envs_trail = make_envs(configs_trail)
        agent_trail = self.agent(configs_trail, envs_trail)
        agent_trail.train(train_steps=self.running_steps)
        scores = agent_trail.test(env_fn=self.eval_env_fn, test_episodes=self.test_episodes)
        returns = agent_trail.returns.sum() / envs_trail.num_envs
        agent_trail.finish()
        envs_trail.close()
        scores_mean = sum(scores) / len(scores)
        return scores_mean, returns


tuner = MyMultiObjectiveTuner(method='dqn',
                              config_path='../dqn/dqn_configs/dqn_cartpole.yaml',
                              running_steps=10000,
                              test_episodes=2)

selected_hyperparameters = tuner.select_hyperparameter(['learning_rate', 'gamma'])

overrides = {
    'learning_rate': [0.001, 0.0001, 0.00001],  # Categorical: Search in the set of {0.001, 0.0001, 0.00001}.
    'gamma': (0.9, 0.99)  # Search from 0.9 to 0.99.
}
selected_hyperparameters = set_hyperparameters(selected_hyperparameters, overrides)

study = tuner.tune(selected_hyperparameters, n_trials=30, pruner=None, directions=['maximize', 'maximize'])
fig = plot_pareto_front(study, target_names=["scores", "returns"])
fig.show()
