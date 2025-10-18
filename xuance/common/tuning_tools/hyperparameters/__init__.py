from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class Hyperparameter:
    """
    Represents a hyperparameter for algorithm tuning.

    This dataclass defines the structure of a hyperparameter, including its name, type, distribution,
    whether it should be sampled on a logarithmic scale, and its default value.

    Attributes:
        name (str): The name of the hyperparameter.
        type (str): The type of the hyperparameter. Supported types include 'int', 'float', and 'categorical'.
        distribution (Union[List[Any], tuple]): The possible values or range for the hyperparameter.
            - For 'categorical' types, this should be a list of possible values.
            - For 'int' and 'float' types, this should be a tuple defining the range (min, max).
        log (bool, optional): Indicates whether the hyperparameter should be sampled on a logarithmic scale.
            This is typically used for hyperparameters like learning rates that span several orders of magnitude.
            Defaults to False.
        default (Optional[Any], optional): The default value of the hyperparameter if no tuning is performed.
            This provides a fallback value to ensure the algorithm can run with standard settings.
            Defaults to None.
    """
    name: str  # The name of the hyperparameter.
    type: str  # 'int', 'float', 'categorical'.
    distribution: Union[List[Any], tuple]  # Possible values or range.
    log: bool = False  # A flag to sample the value from the log domain or not.
    default: Optional[Any] = None  # Default value.


class AlgorithmHyperparametersRegistry:
    """
    A registry for managing hyperparameters of different algorithms.

    This class allows for the registration of algorithms along with their corresponding
    hyperparameters. It provides methods to retrieve hyperparameters for a specific
    algorithm and to list all registered algorithms.

    Attributes:
        _registry (dict): A class-level dictionary mapping algorithm names to their
                          list of hyperparameters.
    """
    _registry = {}

    @classmethod
    def register_algorithm(cls, algorithm_name: str, hyperparameters: List[Hyperparameter]):
        """
        Register an algorithm along with its hyperparameters.

        This method adds an algorithm and its associated hyperparameters to the registry.
        If the algorithm already exists, its hyperparameters will be updated.

        Args:
            algorithm_name (str): The name of the algorithm to register.
            hyperparameters (List[Hyperparameter]): A list of Hyperparameter instances
                                                   defining the algorithm's hyperparameters.

        Example:
            >>> hyperparams = [
            ...     Hyperparameter(name="learning_rate", type="float", distribution=(1e-5, 1e-2), log=True, default=1e-3),
            ...     Hyperparameter(name="gamma", type="float", distribution=(0.85, 0.99), log=False, default=0.99),
            ... ]
            >>> AlgorithmHyperparametersRegistry.register_algorithm("DQN", hyperparams)
        """
        cls._registry[algorithm_name] = hyperparameters

    @classmethod
    def get_hyperparameters(cls, algorithm_name: str) -> List[Hyperparameter]:
        """
        Retrieve the list of hyperparameters for a given algorithm.

        Args:
            algorithm_name (str): The name of the algorithm whose hyperparameters are to be retrieved.

        Returns:
            List[Hyperparameter]: A list of Hyperparameter instances associated with the specified algorithm.
                                  Returns an empty list if the algorithm is not registered.

        Example:
            >>> hyperparams = AlgorithmHyperparametersRegistry.get_hyperparameters("DQN")
            >>> for hp in hyperparams:
            ...     print(hp.name, hp.type)
            learning_rate float
            gamma float
        """
        return cls._registry.get(algorithm_name, [])

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """
        List all registered algorithms.

        Returns:
            List[str]: A list of all algorithm names that have been registered in the registry.

        Example:
            >>> algorithms = AlgorithmHyperparametersRegistry.list_algorithms()
            >>> print(algorithms)
            ['DQN', 'A2C', 'SAC']
        """
        return list(cls._registry.keys())
