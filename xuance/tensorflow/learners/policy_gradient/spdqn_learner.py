"""
Split parameterised deep Q network (SP-DQN)
Paper link: https://arxiv.org/pdf/1810.06394.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace

from xuance.tensorflow import Module
from xuance.tensorflow.learners.policy_gradient.pdqn_learner import PDQN_Learner


class SPDQN_Learner(PDQN_Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(SPDQN_Learner, self).__init__(config, model, callback)
