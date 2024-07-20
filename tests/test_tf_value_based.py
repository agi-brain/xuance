# Test the value-based algorithms with TensorFlow.

from argparse import Namespace
from xuance import get_runner
import unittest

n_steps = 10000
test_mode = False


class TestValueBaseAlgo(unittest.TestCase):
    def test_dqn(self):
        args = Namespace(**dict(dl_toolbox='tensorflow', running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="dqn", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_ddqn(self):
        args = Namespace(**dict(dl_toolbox='tensorflow', running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="ddqn", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_dueldqn(self):
        args = Namespace(**dict(dl_toolbox='tensorflow', running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="dueldqn", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_noisydqn(self):
        args = Namespace(**dict(dl_toolbox='tensorflow', running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="noisydqn", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_perdqn(self):
        args = Namespace(**dict(dl_toolbox='tensorflow', running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="perdqn", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_qrdqn(self):
        args = Namespace(**dict(dl_toolbox='tensorflow', running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="qrdqn", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_c51dqn(self):
        args = Namespace(**dict(dl_toolbox='tensorflow', running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="c51", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_drqn(self):
        args = Namespace(**dict(dl_toolbox='tensorflow', running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="drqn", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()


if __name__ == "__main__":
    unittest.main()
