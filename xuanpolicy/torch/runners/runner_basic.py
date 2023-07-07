from xuanpolicy.environment import make_envs


class Runner_Base(object):
    def __init__(self, args):
        # build environments
        self.envs = make_envs(args)
        self.envs.reset()

        if args.vectorize != 'NOREQUIRED':
            self.n_envs = self.envs.num_envs

    def run(self):
        pass
