from .runner_drl import Runner_DRL as DRL_runner
from .runner_mpe import MPE_Runner as MPE_Runner

REGISTRY = {
    "DL_toolbox": "PyTorch",
    "DRL": DRL_runner,
    "MPE_Runner": MPE_Runner
}
