"""
Multi-agent Soft Actor-critic (MASAC) with discrete action spaces.
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.common import List
from argparse import Namespace
from xuance.torch.learners.multi_agent_rl.isac_learner import ISAC_Learner
from operator import itemgetter


class MASACDIS_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module,
                 callback):
        super(MASACDIS_Learner, self).__init__(config, model_keys, agent_keys, policy, callback)

    def update(self, sample):
        self.iterations += 1
        info = {}

        self.policy.soft_update(self.tau)
        return info

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        return info
