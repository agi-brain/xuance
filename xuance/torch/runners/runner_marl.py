"""
This is demo of runner for multi-agent reinforcement learning.
"""
from .runner_basic import Runner_Base


class Runner_MARL(Runner_Base):
    def __init__(self, args):
        super(Runner_MARL, self).__init__(args)

