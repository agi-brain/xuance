from argparse import Namespace
from torch import nn
from xuance.common import AgentGrouping
from xuance.torch.learners.multi_agent_rl.ic3net_learner import IC3Net_Learner


class TarMAC_Learner(IC3Net_Learner):
    def __init__(self,
                 config: Namespace,
                 agent_grouping: AgentGrouping,
                 model: nn.Module,
                 callback):
        super(TarMAC_Learner, self).__init__(config, agent_grouping, model, callback)
