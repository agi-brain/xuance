# Test the MARL-based algorithms with PyTorch.

from argparse import Namespace
from xuance import get_runner
import unittest

n_steps = 10000
device = 'cuda:0'
test_mode = False


class TestValueBaseAlgo(unittest.TestCase):
    def test_iql(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="iql", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    def test_vdn(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="vdn", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    def test_qmix(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="qmix", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    def test_wqmix(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="wqmix", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    # def test_qtran(self):
    #     args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
    #     runner = get_runner(method="qtran", env='mpe', env_id='simple_spread_v3', parser_args=args)
    #     runner.run()

    # def test_dcg(self):
    #     args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
    #     runner = get_runner(method="dcg", env='mpe', env_id='simple_spread_v3', parser_args=args)
    #     runner.run()

    # def test_vdac(self):
    #     args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
    #     runner = get_runner(method="vdac", env='mpe', env_id='simple_spread_v3', parser_args=args)
    #     runner.run()

    # def test_coma(self):
    #     args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
    #     runner = get_runner(method="coma", env='mpe', env_id='simple_spread_v3', parser_args=args)
    #     runner.run()

    def test_ippo(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="ippo", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    def test_mappo(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="mappo", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    def test_iddpg(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="iddpg", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    def test_maddpg(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="maddpg", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    def test_matd3(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="matd3", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    def test_isac(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="isac", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    def test_masac(self):
        args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
        runner = get_runner(method="masac", env='mpe', env_id='simple_spread_v3', parser_args=args)
        runner.run()

    # def test_mfq(self):
    #     args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
    #     runner = get_runner(method="mfq", env='mpe', env_id='simple_spread_v3', parser_args=args)
    #     runner.run()

    # def test_mfac(self):
    #     args = Namespace(**dict(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode))
    #     runner = get_runner(method="mfac", env='mpe', env_id='simple_spread_v3', parser_args=args)
    #     runner.run()


if __name__ == "__main__":
    unittest.main()
