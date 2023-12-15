from xuance.environment import make_envs
from xuance.tensorflow.utils.operations import set_seed
import tensorflow.keras as tk


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


class MyLinearLR(tk.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, start_factor, end_factor, total_iters):
        self.initial_learning_rate = initial_learning_rate
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        self.learning_rate = self.initial_learning_rate
        self.delta_factor = (end_factor - start_factor) * self.initial_learning_rate / self.total_iters

    def __call__(self, step):
        self.learning_rate += self.delta_factor
        return self.learning_rate

