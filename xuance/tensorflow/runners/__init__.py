from .runner_basic import RunnerBase
from .runner_drl import RunnerDRL
from .runner_marl import RunnerMARL
from .runner_pettingzoo import RunnerPettingzoo

REGISTRY_Runner = {
    "DL_toolbox": "TensorFlow2",
    "DRL": RunnerDRL,
    "MARL": RunnerMARL,
    "RunnerPettingzoo": RunnerPettingzoo,
}

__all__ = [
    "RunnerBase",
    "RunnerDRL",
    "RunnerMARL",
    "REGISTRY_Runner"
]
