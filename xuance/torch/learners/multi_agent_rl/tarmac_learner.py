from argparse import Namespace
from typing import List

from torch import nn

from xuance.torch.learners.multi_agent_rl.ic3net_learner import IC3Net_Learner


class TarMAC_Learner(IC3Net_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(TarMAC_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)
