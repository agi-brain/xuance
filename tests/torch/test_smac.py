# Test the value-based algorithms with PyTorch.

from argparse import Namespace
import importlib.util
from xuance import get_runner
import unittest

running_steps = 10000
device = 'cuda:0'


@unittest.skipUnless(importlib.util.find_spec("smac") is not None, "smac is not installed")
class TestSMAC(unittest.TestCase):
    def test_dqn(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=running_steps)
        runner = get_runner(algo="qmix", env='sc2', env_id='3m', parser_args=args)
        runner.run(mode="benchmark", running_steps=running_steps)

    def test_ppo(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=running_steps)
        runner = get_runner(algo="mappo", env='sc2', env_id='3m', parser_args=args)
        runner.run(mode="benchmark", running_steps=running_steps)


if __name__ == "__main__":
    unittest.main()
