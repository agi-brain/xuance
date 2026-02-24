from .runner_pettingzoo import RunnerPettingzoo


class RunnerMAgent(RunnerPettingzoo):
    def __init__(self, args):
        super(RunnerMAgent, self).__init__(args)
