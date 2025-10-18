# Test the value-based algorithms with PyTorch.

from argparse import Namespace
from xuance import get_runner
import unittest

is_benchmark = False

n_steps = 10000
eval_interval = 5000
device = 'cuda:0'
test_mode = False
args = Namespace(dl_toolbox='torch', device=device, 
                 running_steps=n_steps, eval_interval=eval_interval, test_mode=test_mode)
env_name = "classic_control"
env_id = "CartPole-v1"


class TestValueBaseAlgo(unittest.TestCase):
    def test_c51dqn(self):
        runner = get_runner(method="c51", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_ddqn(self):
        runner = get_runner(method="ddqn", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_dqn(self):
        runner = get_runner(method="dqn", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_drqn(self):
        runner = get_runner(method="drqn", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_dueldqn(self):
        runner = get_runner(method="dueldqn", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_noisydqn(self):
        runner = get_runner(method="noisydqn", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_perdqn(self):
        runner = get_runner(method="perdqn", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_qrdqn(self):
        runner = get_runner(method="qrdqn", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()


if __name__ == "__main__":
    unittest.main()
