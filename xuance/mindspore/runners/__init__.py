from .runner_drl import Runner_DRL
from .runner_pettingzoo import Pettingzoo_Runner

REGISTRY = {
    "DL_toolbox": "MindSpore",
    "DRL": Runner_DRL,
    "Pettingzoo_Runner": Pettingzoo_Runner,
}
