from xuance.environment import make_envs
from xuance.mindspore.utils.operations import set_seed


class Runner_Base(object):
    def __init__(self, args):
        # set random seeds
        set_seed(args.seed)

        # build environments
        self.envs = make_envs(args)
        self.envs.reset()
        self.n_envs = self.envs.num_envs

    def run(self):
        pass
