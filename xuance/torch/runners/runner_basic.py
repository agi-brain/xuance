from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed


class Runner_Base(object):
    def __init__(self, config):
        # set random seeds
        set_seed(config.seed)

        # build environments
        self.envs = make_envs(config)
        self.envs.reset()
        self.n_envs = self.envs.num_envs
        self.rank = 0

    def run(self):
        raise NotImplementedError

    def benchmark(self):
        raise NotImplementedError

    def rprint(self, info: str):
        if self.rank == 0:
            print(info)
