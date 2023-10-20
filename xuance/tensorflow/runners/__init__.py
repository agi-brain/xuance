from .runner_drl import Runner_DRL as DRL_runner
from .runner_marl import Runner as MARL_runner

REGISTRY = {
    "DL_toolbox": "TensorFlow",
    "DRL": DRL_runner,
    "MARL": MARL_runner
}
