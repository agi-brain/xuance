from argparse import Namespace
from typing import List

import torch
from torch import nn

from xuance.torch.learners.multi_agent_rl.commnet_learner import CommNet_Learner


class DGN_Learner(CommNet_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(DGN_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)