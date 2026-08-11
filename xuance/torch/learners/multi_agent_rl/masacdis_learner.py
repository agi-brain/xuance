"""
Multi-agent Soft Actor-critic (MASAC) with discrete action spaces.
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.common import AgentGrouping
from argparse import Namespace
from xuance.torch.learners.multi_agent_rl.isac_learner import ISAC_Learner
from operator import itemgetter


class MASACDIS_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(MASACDIS_Learner, self).__init__(config, agent_grouping, model, callback)

    def update(self, sample):
        self.iterations += 1
        info = {}

        self.policy.soft_update(self.tau)
        return info

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        return info
