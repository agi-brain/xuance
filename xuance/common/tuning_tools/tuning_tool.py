import optuna
import importlib
from copy import deepcopy
from argparse import Namespace
from xuance.environment import make_envs
from xuance.common import get_configs, Union, Any, Optional, List
from xuance.common.tuning_tools.hyperparameters import Hyperparameter, AlgorithmHyperparametersRegistry


def build_search_space(trail: optuna.trial, hyperparameters: List[Hyperparameter]) -> dict:
    search_space = {}
    for param in hyperparameters:
        if isinstance(param.distribution, list) or param.type == "categorical":
            search_space[param.name] = trail.suggest_categorical(param.name, param.distribution)
        else:
            if param.type == "float":
                if param.log:
                    search_space[param.name] = trail.suggest_loguniform(param.name,
                                                                        low=param.distribution[0],
                                                                        high=param.distribution[1])
                else:
                    search_space[param.name] = trail.suggest_uniform(param.name,
                                                                     low=param.distribution[0],
                                                                     high=param.distribution[1])
            elif param.type == "int":
                search_space[param.name] = trail.suggest_int(param.name, param.distribution[0], param.distribution[1])
            else:
                raise ValueError(f"Unsupported hyperparameter type: {param.type}")
    return search_space


def set_hyperparameters(hyperparameters: List[Hyperparameter], overrides: dict) -> List[Hyperparameter]:
    for param in hyperparameters:
        if param.name in overrides.keys():
            new_distribution = overrides[param.name]
            if isinstance(new_distribution, tuple) or isinstance(new_distribution, list):
                param.distribution = new_distribution
            else:
                raise ValueError(f"Unsupported distribution type for {param.name}: {type(new_distribution)}")
    return hyperparameters


class HyperParameterTuner:
    def __init__(self,
                 method: str,
                 config_path: str,
                 running_steps: Optional[int] = None,
                 test_episodes: Optional[int] = None):
        """
        Initialize the HyperParameterTuner module.

        Args:
            agent_name (str): The name of the method (or agent).
            env_id (str): The environment id.
            config_path (str): The configurations.
            running_steps (int): Number of steps to run a trail.
            test_episodes (int): Number of episodes to evaluate the agent's policy.
        """
        self.method = method
        self.configs_dict = get_configs(config_path)
        self.running_steps = self.configs_dict['running_steps'] if running_steps is None else running_steps
        self.test_episodes = self.configs_dict['test_episodes'] if test_episodes is None else test_episodes
        if self.configs_dict['dl_toolbox'] == "torch":
            from xuance.torch.agents import REGISTRY_Agents
        elif self.configs_dict['dl_toolbox'] == "tensorflow":
            from xuance.tensorflow.agents import REGISTRY_Agents
        elif self.configs_dict['dl_toolbox'] == "mindspore":
            from xuance.mindspore.agents import REGISTRY_Agents
        else:
            raise AttributeError(f"XuanCe currently does not support {self.configs_dict['dl_toolbox']}!")
        self.agent_name = self.configs_dict['agent']
        self.agent = REGISTRY_Agents[self.agent_name]
        module = importlib.import_module(f"xuance.common.tuning_tools.hyperparameters.{self.method}")
        params = getattr(module, f"{self.method}_hyperparams")
        AlgorithmHyperparametersRegistry.register_algorithm(self.configs_dict['agent'], params)

    def list_hyperparameters(self) -> List[Hyperparameter]:
        return AlgorithmHyperparametersRegistry.get_hyperparameters(self.agent_name)

    def select_hyperparameter(self, hyperparameter_names: List[str]) -> List[Hyperparameter]:
        all_hyperparams = self.list_hyperparameters()
        selected_hyperparams = [param for param in all_hyperparams if param.name in hyperparameter_names]
        if not selected_hyperparams:
            raise ValueError("No hyperparameters selected for tuning.")
        return selected_hyperparams

    def eval_env_fn(self):
        """
        The environment for evaluating the agent's policy.

        Returns: Vectorized environments.
        """
        configs_test = Namespace(**self.configs_dict)
        configs_test.parallels = 1
        return make_envs(configs_test)

    def objective(self, trail: optuna.trial, selected_hyperparameters: List[Hyperparameter]) -> float:
        """
        Define the objective function.

        Args:
            selected_hyperparameters:
            trail:

        Returns:

        """
        search_space = build_search_space(trail, selected_hyperparameters)
        config_trail = deepcopy(self.configs_dict)
        config_trail.update(search_space)
        configs_trail = Namespace(**config_trail)
        envs_trail = make_envs(configs_trail)
        agent_trail = self.agent(configs_trail, envs_trail)
        agent_trail.train(train_steps=self.running_steps)
        scores = agent_trail.test(env_fn=self.eval_env_fn, test_episodes=self.test_episodes)
        agent_trail.finish()
        envs_trail.close()
        scores_mean = sum(scores) / len(scores)
        return scores_mean

    def tune(self,
             selected_hyperparameters: List[Hyperparameter],
             n_trails: int = 1,
             pruner: Optional[optuna.pruners.BasePruner] = None) -> optuna.study.Study:
        """
        Start the tuning process.

        Args:
            n_trails:
            pruner:

        Returns:

        """
        study = optuna.create_study(direction="maximize", pruner=pruner)

        def objective_wrapper(trial):
            return self.objective(trial, selected_hyperparameters)

        study.optimize(objective_wrapper, n_trials=n_trails)

        print("Best hyper-parameters: ", study.best_params)
        print("Best value: ", study.best_value)

        return study

