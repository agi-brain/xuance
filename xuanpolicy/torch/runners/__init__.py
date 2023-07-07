from .runner_drl import Runner_DRL
from .runner_pettingzoo import Pettingzoo_Runner
from .runner_magent import MAgent_Runner

REGISTRY = {
    "DL_toolbox": "PyTorch",
    "DRL": Runner_DRL,
    "Pettingzoo_Runner": Pettingzoo_Runner,
    "MAgent_Runner": MAgent_Runner
}
