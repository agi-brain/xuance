# Test the policy-based algorithms with MindSpore.

from argparse import Namespace
from xuance import get_runner
import unittest

n_steps = 10000
device = 'CPU'
test_mode = False


class TestValueBaseAlgo(unittest.TestCase):
    def test_pg_continuous(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="pg", env='classic_control', env_id='Pendulum-v1', parser_args=args)
        runner.run()

    def test_pg_discrete(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="pg", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_ppg_continuous(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="ppg", env='classic_control', env_id='Pendulum-v1', parser_args=args)
        runner.run()

    def test_ppg_discrete(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="ppg", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_a2c_continuous(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="a2c", env='classic_control', env_id='Pendulum-v1', parser_args=args)
        runner.run()

    def test_a2c_discrete(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="a2c", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_ddpg(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="ddpg", env='classic_control', env_id='Pendulum-v1', parser_args=args)
        runner.run()

    def test_td3(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="td3", env='classic_control', env_id='Pendulum-v1', parser_args=args)
        runner.run()

    def test_sac_continuous(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="sac", env='classic_control', env_id='Pendulum-v1', parser_args=args)
        runner.run()

    def test_sac_discrete(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="sac", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

    def test_ppo_continuous(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="ppo", env='classic_control', env_id='Pendulum-v1', parser_args=args)
        runner.run()

    def test_ppo_discrete(self):
        args = Namespace(**dict(dl_toolbox='mindspore', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="ppo", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()


if __name__ == "__main__":
    unittest.main()
