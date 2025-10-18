# Test the value-based algorithms with PyTorch.

from argparse import Namespace
from xuance import get_runner
import unittest

n_steps = 100000
device = 'cuda:0'
test_mode = False


class TestValueBaseAlgo(unittest.TestCase):
    def test_dqn(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode)
        runner = get_runner(method="dqn", env='atari', env_id='ALE/Breakout-v5', parser_args=args)
        runner.benchmark()

    def test_ppo(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode)
        runner = get_runner(method="ppo", env='atari', env_id='ALE/Breakout-v5', parser_args=args)
        runner.benchmark()


if __name__ == "__main__":
    unittest.main()
