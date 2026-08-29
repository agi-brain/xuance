"""
Multi-pass parameterised deep Q network (MP-DQN)
Paper link: https://arxiv.org/pdf/1905.04388.pdf
Implementation: Pytorch
"""
from argparse import Namespace
from torch.nn import Module
from xuance.torch.learners.policy_gradient.pdqn_learner import PDQN_Learner


class MPDQN_Learner(PDQN_Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(MPDQN_Learner, self).__init__(config, model, callback)
