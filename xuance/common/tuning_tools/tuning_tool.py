import optuna
import importlib
from copy import deepcopy
from argparse import Namespace
from operator import itemgetter
from xuance.environment import make_envs
from xuance.common import get_configs, Optional, List
from xuance.common.tuning_tools.hyperparameters import Hyperparameter, AlgorithmHyperparametersRegistry


def build_search_space(trial: optuna.trial, hyperparameters: List[Hyperparameter]) -> dict:
    '''
    Build the search space for tuning hyperparameters using Optuna.

    This function iterates over a list of hyperparameters and defines the search space
    based on their type and distribution. It supports categorical, float (both uniform
    and log-uniform), and integer hyperparameters.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object used to suggest hyperparameter values.
        hyperparameters (List[Hyperparameter]): A list of Hyperparameter instances defining
            the hyperparameters to tune.

    Returns:
        dict: A dictionary mapping hyperparameter names to their suggested values for the current trial.

    Raises:
        ValueError: If an unsupported hyperparameter type is encountered.

    Example:
        >>> hyperparams = [
        ...     Hyperparameter(name="learning_rate", type="float", distribution=(1e-5, 1e-2), log=True, default=1e-3),
        ...     Hyperparameter(name="batch_size", type="int", distribution=[32, 64, 128, 256], default=64),
        ... ]
        >>> trial = optuna.trial.create_trial()
        >>> search_space = build_search_space(trial, hyperparams)
    '''
    search_space = {}
    for param in hyperparameters:
        if isinstance(param.distribution, list) or param.type == "categorical":
            search_space[param.name] = trial.suggest_categorical(param.name, param.distribution)
        else:
            if param.type == "float":
                if param.log:
                    search_space[param.name] = trial.suggest_loguniform(param.name,
                                                                        low=param.distribution[0],
                                                                        high=param.distribution[1])
                else:
                    search_space[param.name] = trial.suggest_uniform(param.name,
                                                                     low=param.distribution[0],
                                                                     high=param.distribution[1])
            elif param.type == "int":
                search_space[param.name] = trial.suggest_int(param.name, param.distribution[0], param.distribution[1])
            else:
                raise ValueError(f"Unsupported hyperparameter type: {param.type}")
    return search_space


def set_hyperparameters(hyperparameters: List[Hyperparameter], overrides: dict) -> List[Hyperparameter]:
    '''
    Override the distributions of specified hyperparameters.

    This function updates the distribution of hyperparameters based on the provided overrides.
    It ensures that the new distribution is either a tuple or a list. If an unsupported
    distribution type is provided, it raises a ValueError.

    Args:
        hyperparameters (List[Hyperparameter]): The list of Hyperparameter instances to override.
        overrides (dict): A dictionary mapping hyperparameter names to their new distributions.

    Returns:
        List[Hyperparameter]: The updated list of Hyperparameter instances with overridden distributions.

    Raises:
        ValueError: If an unsupported distribution type is provided for any hyperparameter.

    Example:
        >>> hyperparams = [
        ...     Hyperparameter(name="learning_rate", type="float", distribution=(1e-5, 1e-2), log=True, default=1e-3),
        ...     Hyperparameter(name="batch_size", type="int", distribution=[32, 64, 128, 256], default=64),
        ... ]
        >>> overrides = {
        ...     "learning_rate": [0.001, 0.0001, 0.00001],
        ... }
        >>> updated_hyperparams = set_hyperparameters(hyperparams, overrides)
    '''
    for param in hyperparameters:
        if param.name in overrides.keys():
            new_distribution = overrides[param.name]
            if isinstance(new_distribution, tuple) or isinstance(new_distribution, list):
                param.distribution = new_distribution
            else:
                raise ValueError(f"Unsupported distribution type for {param.name}: {type(new_distribution)}")
    return hyperparameters


class HyperParameterTuner:
    """
    A class to facilitate automatic hyperparameter tuning for reinforcement learning algorithms using Optuna.

    The HyperParameterTuner class provides methods to list, select, and tune hyperparameters for a specified
    algorithm within the XuanCe framework. It integrates with Optuna to perform efficient hyperparameter
    optimization.

    Attributes:
        method (str): The name of the method or agent (e.g., 'dqn').
        config_path (str): The path to the configuration YAML file.
        running_steps (Optional[int]): Number of steps to run a trial. Defaults to the value in the configuration.
        test_episodes (Optional[int]): Number of episodes to evaluate the agent's policy. Defaults to the value in the configuration.
        agent_name (str): The name of the agent as specified in the configuration.
        agent: The agent instance retrieved from the agent registry.
    """
    def __init__(self,
                 method: str,
                 config_path: str,
                 running_steps: Optional[int] = None,
                 test_episodes: Optional[int] = None):
        """
        Initialize the HyperParameterTuner module.

        This constructor sets up the tuner by loading configurations, selecting the appropriate
        deep learning toolbox, and registering the algorithm's hyperparameters.

        Args:
            method (str): The name of the method or agent (e.g., 'dqn').
            config_path (str): The path to the configuration YAML file.
            running_steps (Optional[int], optional): Number of steps to run a trial. Defaults to None,
                which means it will use the value from the configuration.
            test_episodes (Optional[int], optional): Number of episodes to evaluate the agent's policy.
                Defaults to None, which means it will use the value from the configuration.

        Raises:
            AttributeError: If the specified deep learning toolbox is not supported.
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
        """
        List the hyperparameters of the selected algorithm.

        This method retrieves all registered hyperparameters for the algorithm specified during
        the initialization of the tuner.

        Returns:
            List[Hyperparameter]: A list of Hyperparameter instances associated with the selected algorithm.

        Example:
            >>> tuner = HyperParameterTuner(method='dqn', config_path='config.yaml')
            >>> hyperparams = tuner.list_hyperparameters()
            >>> for hp in hyperparams:
            ...     print(hp.name, hp.type)
            learning_rate float
            gamma float
        """
        return AlgorithmHyperparametersRegistry.get_hyperparameters(self.agent_name)

    def select_hyperparameter(self, hyperparameter_names: List[str]) -> List[Hyperparameter]:
        """
        Select specific hyperparameters for tuning based on their names.

        This method filters the list of all hyperparameters to include only those specified
        by the user. It raises an error if no hyperparameters are selected.

        Args:
            hyperparameter_names (List[str]): A list of hyperparameter names to select for tuning.

        Returns:
            List[Hyperparameter]: A list of selected Hyperparameter instances.

        Raises:
            ValueError: If no hyperparameters are selected for tuning.

        Example:
            >>> tuner = HyperParameterTuner(method='dqn', config_path='config.yaml')
            >>> selected = tuner.select_hyperparameter(['learning_rate', 'gamma'])
            >>> for hp in selected:
            ...     print(hp.name, hp.type)
            learning_rate float
            gamma float
        """
        all_hyperparams = self.list_hyperparameters()
        selected_hyperparams = [param for param in all_hyperparams if param.name in hyperparameter_names]
        if not selected_hyperparams:
            raise ValueError("No hyperparameters selected for tuning.")
        return selected_hyperparams

    def eval_env_fn(self):
        """
        Create the environment for evaluating the agent's policy.

        This method configures a single (vectorized) environment instance used solely for
        evaluating the performance of the trained agent.

        Returns:
            Vectorized Environment: An instance of the environment configured for evaluation.
        """
        configs_test = Namespace(**self.configs_dict)
        configs_test.parallels = 1
        return make_envs(configs_test)

    def objective(self, trail: optuna.trial, selected_hyperparameters: List[Hyperparameter]) -> float:
        """
        Define the objective function for Optuna optimization.

        This function builds the search space, updates the configuration with suggested
        hyperparameters, initializes the environment and agent, trains the agent, evaluates
        its performance, and returns the mean score as the objective value.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object used for suggesting hyperparameter values.
            selected_hyperparameters (List[Hyperparameter]): A list of Hyperparameter instances selected for tuning.

        Returns:
            float: The mean score obtained from evaluating the agent's policy.

        Example:
            >>> tuner = HyperParameterTuner(method='dqn', config_path='config.yaml')
            >>> hyperparams = tuner.select_hyperparameter(['learning_rate', 'gamma'])
            >>> study = optuna.create_study(direction="maximize")
            >>> study.optimize(lambda trial: tuner.objective(trial, hyperparams), n_trials=10)
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
             n_trials: int = 1,
             pruner: Optional[optuna.pruners.BasePruner] = None,
             direction: str = "maximize") -> optuna.study.Study:
        """
        Start the hyperparameter tuning process.

        This method initializes an Optuna study, defines the objective function wrapper,
        and begins the optimization process to search for the best hyperparameter values.

        Args:
            selected_hyperparameters (List[Hyperparameter]): A list of Hyperparameter instances selected for tuning.
            n_trials (int, optional): The number of trials to run during optimization. Defaults to 1.
            pruner (Optional[optuna.pruners.BasePruner], optional): An Optuna pruner to terminate unpromising trials early.
                Defaults to None.
            direction (str): The optimization directions. Defaults to "maximize".

        Returns:
            optuna.study.Study: The Optuna study object containing the results of the optimization.

        Example:
            >>> tuner = HyperParameterTuner(method='dqn', config_path='config.yaml')
            >>> hyperparams = tuner.select_hyperparameter(['learning_rate', 'gamma'])
            >>> study = tuner.tune(selected_hyperparameters=hyperparams, n_trials=30)
            >>> print(study.best_params)
        """
        study = optuna.create_study(direction=direction, pruner=pruner)

        def objective_wrapper(trial):
            return self.objective(trial, selected_hyperparameters)

        study.optimize(objective_wrapper, n_trials=n_trials)

        print("Best hyperparameters: ", study.best_params)
        print("Best value: ", study.best_value)

        return study


class MultiObjectiveTuner(HyperParameterTuner):
    """
    A class to facilitate multi-objective hyperparameter tuning for reinforcement learning algorithms using Optuna.
    """
    def __init__(self, **kwargs):
        super(MultiObjectiveTuner, self).__init__(**kwargs)

    def objective(self, trial: optuna.trial, selected_hyperparameters: List[Hyperparameter],
                  selected_objectives: List[str] = None) -> float:
        """
        Define the objective function for Optuna optimization.

        This function builds the search space, updates the configuration with suggested
        hyperparameters, initializes the environment and agent, trains the agent, evaluates
        its performance, and returns the mean score as the objective value.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object used for suggesting hyperparameter values.
            selected_hyperparameters (List[Hyperparameter]): A list of Hyperparameter instances selected for tuning.
            selected_objectives (List[str]): A list of objectives selected for tuning.

        Returns:
            float: The mean score obtained from evaluating the agent's policy.

        Example:
            >>> tuner = HyperParameterTuner(method='dqn', config_path='config.yaml')
            >>> hyperparams = tuner.select_hyperparameter(['learning_rate', 'gamma'])
            >>> study = optuna.create_study(direction="maximize")
            >>> study.optimize(lambda trial: tuner.objective(trial, hyperparams), n_trials=10)
        """
        search_space = build_search_space(trial, selected_hyperparameters)
        config_trail = deepcopy(self.configs_dict)
        config_trail.update(search_space)
        configs_trail = Namespace(**config_trail)
        envs_trail = make_envs(configs_trail)
        agent_trail = self.agent(configs_trail, envs_trail)
        train_info = agent_trail.train(train_steps=self.running_steps)
        scores = agent_trail.test(env_fn=self.eval_env_fn, test_episodes=self.test_episodes)
        agent_trail.finish()
        envs_trail.close()
        scores_mean = sum(scores) / len(scores)
        train_info["test_score"] = scores_mean
        objectives = itemgetter(*selected_objectives)(train_info)
        return objectives

    def tune(self,
             selected_hyperparameters: List[Hyperparameter],
             n_trials: int = 1,
             pruner: Optional[optuna.pruners.BasePruner] = None,
             directions: Optional[list] = None,
             selected_objectives: List[str] = None,) -> optuna.study.Study:
        """
        Start the hyperparameter tuning process.

        This method initializes an Optuna study, defines the objective function wrapper,
        and begins the optimization process to search for the best hyperparameter values.

        Args:
            selected_hyperparameters (List[Hyperparameter]): A list of Hyperparameter instances selected for tuning.
            n_trials (int, optional): The number of trials to run during optimization. Defaults to 1.
            pruner (Optional[optuna.pruners.BasePruner], optional): An Optuna pruner to terminate unpromising trials early.
                Defaults to None.
            directions: The optimization directions. Default is None.
            selected_objectives (List[str]): A list of objectives selected for tuning.

        Returns:
            optuna.study.Study: The Optuna study object containing the results of the optimization.

        Example:
            >>> tuner = HyperParameterTuner(method='dqn', config_path='config.yaml')
            >>> hyperparams = tuner.select_hyperparameter(['learning_rate', 'gamma'])
            >>> study = tuner.tune(selected_hyperparameters=hyperparams, n_trials=30, selected_objectives=['test_score', 'loss'])
            >>> print(study.best_params)
        """
        study = optuna.create_study(directions=directions, pruner=pruner)

        def objective_wrapper(trial):
            return self.objective(trial, selected_hyperparameters, selected_objectives)

        study.optimize(objective_wrapper, n_trials=n_trials)

        print("Number of finished trials: ", len(study.trials))

        return study
