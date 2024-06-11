from .runner_drl import Runner_DRL
from .runner_pettingzoo import Pettingzoo_Runner

REGISTRY_Runner = {
    "DL_toolbox": "MindSpore",
    "DRL": Runner_DRL,
    "Pettingzoo_Runner": Pettingzoo_Runner,
}

__all__ = [
    "Runner_DRL",
    "Pettingzoo_Runner",
    "REGISTRY_Runner",
]
