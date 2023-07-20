from .runner_pettingzoo import Pettingzoo_Runner


class Football_Runner(Pettingzoo_Runner):
    def __init__(self, args):
        super(Football_Runner, self).__init__(args)
        self.fps = 50
