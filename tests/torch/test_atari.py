# Test the value-based algorithms with PyTorch.

from argparse import Namespace
import importlib.util
from xuance import get_runner
import unittest

device = 'cuda:0'


@unittest.skipUnless(importlib.util.find_spec("ale_py") is not None, "ale_py is not installed")
class TestAtari(unittest.TestCase):
    def test_dqn(self):
        running_steps = 500
        args = Namespace(
            dl_toolbox='torch',
            device=device,
            running_steps=running_steps,
            buffer_size=500,
            start_training=100,
            training_frequency=25,
            render=False,
        )
        runner = get_runner(algo="dqn", env='atari', env_id='ALE/Breakout-v5', parser_args=args)
        runner.run(mode="train", running_steps=running_steps)

    def test_ppo(self):
        running_steps = 1024
        args = Namespace(
            dl_toolbox='torch',
            device=device,
            running_steps=running_steps,
            horizon_size=64,
            n_epochs=2,
            render=False,
        )
        runner = get_runner(algo="ppo", env='atari', env_id='ALE/Breakout-v5', parser_args=args)
        runner.run(mode="train", running_steps=running_steps)


if __name__ == "__main__":
    unittest.main()
