# Test the model-based algorithms with PyTorch.

from argparse import Namespace
from xuance import get_runner
import unittest

running_steps = 1000
device = 'cuda:0'


@unittest.skip("DreamerV2/DreamerV3 policy migration to rl_models is incomplete")
class TestModelBaseAlgo(unittest.TestCase):

    def test_dreamer_v2(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=running_steps)
        runner = get_runner(algo="dreamerv2", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run(mode="train", running_steps=running_steps)

    def test_dreamer_v3(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=running_steps)
        runner = get_runner(algo="dreamerv3", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run(mode="train", running_steps=running_steps)

    def test_harmony_dream(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=running_steps)
        args.harmony = True
        runner = get_runner(algo="dreamerv3", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run(mode="train", running_steps=running_steps)


if __name__ == "__main__":
    unittest.main()
