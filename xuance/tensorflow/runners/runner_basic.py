from xuance.environment import make_envs
from xuance.tensorflow.utils.operations import set_seed


class RunnerBase(object):
    def __init__(self, config):
        # set random seeds
        set_seed(config.seed)

        # build environments
        self.envs = make_envs(config)
        self.envs.reset()
        self.n_envs = self.envs.num_envs

    def run(self):
        raise NotImplementedError

    def benchmark(self):
        raise NotImplementedError
