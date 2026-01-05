from xuance.environment import make_envs


class RunnerBase(object):
    def __init__(self, config):
        # build environments
        self.envs = make_envs(config)
        self.envs.reset()
        self.n_envs = self.envs.num_envs

    def run(self):
        raise NotImplementedError

    def benchmark(self):
        raise NotImplementedError
