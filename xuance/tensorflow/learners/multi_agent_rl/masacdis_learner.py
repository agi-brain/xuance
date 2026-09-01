"""
Multi-agent Soft Actor-critic (MASAC) with discrete action spaces.
Implementation: Pytorch
"""
from argparse import Namespace
from xuance.common import AgentGrouping

from xuance.tensorflow import Module
from xuance.tensorflow.learners.multi_agent_rl.isac_learner import ISAC_Learner


class MASACDIS_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: Module,
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
