from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class Hyperparameter:
    name: str
    type: str  # 'int', 'float', 'categorical'
    distribution: Union[List[Any], tuple]  # Possible values or range
    log: bool = False  # A flag to sample the value from the log domain or not.
    default: Optional[Any] = None  # Default value.


class AlgorithmHyperparametersRegistry:
    _registry = {}

    @classmethod
    def register_algorithm(cls, algorithm_name: str, hyperparameters: List[Hyperparameter]):
        cls._registry[algorithm_name] = hyperparameters

    @classmethod
    def get_hyperparameters(cls, algorithm_name: str) -> List[Hyperparameter]:
        return cls._registry.get(algorithm_name, [])

    @classmethod
    def list_algorithms(cls) -> List[str]:
        return list(cls._registry.keys())
