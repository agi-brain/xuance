"""
Multi-pass parameterised deep Q network (MP-DQN)
Paper link: https://arxiv.org/pdf/1905.04388.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.tensorflow import Module
from xuance.tensorflow.learners.policy_gradient.pdqn_learner import PDQN_Learner


class MPDQN_Learner(PDQN_Learner):
    def __init__(self,
                 config: Namespace,
                 model: Module,
                 callback):
        super(MPDQN_Learner, self).__init__(config, model, callback)
