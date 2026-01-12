# Test the model-based algorithms with PyTorch.

from argparse import Namespace
from xuance import get_runner
import unittest

n_steps = 1000
device = 'cuda:0'
test_mode = False


class TestValueBaseAlgo(unittest.TestCase):

    def test_dreamer_v2(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=running_steps, test_mode=test_mode)
        runner = get_runner(algo="dreamerv2", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_dreamer_v3(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=running_steps, test_mode=test_mode)
        runner = get_runner(algo="dreamerv3", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_harmony_dream(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=running_steps, test_mode=test_mode)
        args.harmony = True
        runner = get_runner(algo="dreamerv3", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()


if __name__ == "__main__":
    unittest.main()
