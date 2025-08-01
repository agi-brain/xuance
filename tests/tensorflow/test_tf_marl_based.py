# Test the MARL-based algorithms with PyTorch.

from argparse import Namespace
from copy import deepcopy
from xuance import get_runner
import unittest

is_benchmark = False

n_steps = 10000
eval_interval = 5000
device = 'GPU'
test_mode = False
args = Namespace(dl_toolbox='tensorflow', device=device,
                 running_steps=n_steps, eval_interval=eval_interval, test_mode=test_mode)
env_name = "mpe"
env_id = "simple_spread_v3"


class TestValueBaseAlgo(unittest.TestCase):
    def test_coma(self):
        runner = get_runner(method="coma", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    # def test_commnet(self):
    #     runner = get_runner(method="commnet", env=env_name, env_id=env_id, parser_args=args)
    #     if is_benchmark:
    #         runner.benchmark()
    #     else:
    #         runner.run()

    # def test_dcg(self):
    #     runner = get_runner(method="dcg", env=env_name, env_id=env_id, parser_args=args)
    #     if is_benchmark:
    #         runner.benchmark()
    #     else:
    #         runner.run()

    def test_iac_continuous(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = True
        args_continuous.policy = "Gaussian_MAAC_Policy"
        runner = get_runner(method="iac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_iac_discrete(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = False
        args_continuous.policy = "Categorical_MAAC_Policy"
        runner = get_runner(method="iac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    # def test_ic3net(self):
    #     runner = get_runner(method="ic3net", env=env_name, env_id=env_id, parser_args=args)
    #     if is_benchmark:
    #         runner.benchmark()
    #     else:
    #         runner.run()

    def test_iddpg(self):
        runner = get_runner(method="iddpg", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_ippo_continuous(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = True
        args_continuous.policy = "Gaussian_MAAC_Policy"
        runner = get_runner(method="ippo", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_ippo_discrete(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = False
        args_continuous.policy = "Categorical_MAAC_Policy"
        runner = get_runner(method="ippo", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_iql(self):
        runner = get_runner(method="iql", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_isac(self):
        runner = get_runner(method="isac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_maddpg(self):
        runner = get_runner(method="maddpg", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_mappo_continuous(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = True
        args_continuous.policy = "Gaussian_MAAC_Policy"
        runner = get_runner(method="mappo", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_mappo_discrete(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = False
        args_continuous.policy = "Categorical_MAAC_Policy"
        runner = get_runner(method="mappo", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_masac(self):
        runner = get_runner(method="masac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_matd3(self):
        runner = get_runner(method="matd3", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_mfac(self):
        runner = get_runner(method="mfac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()
    #
    def test_mfq(self):
        runner = get_runner(method="mfq", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_qmix(self):
        runner = get_runner(method="qmix", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    # def test_qtran(self):
    #     runner = get_runner(method="qtran", env=env_name, env_id=env_id, parser_args=args)
    #     if is_benchmark:
    #         runner.benchmark()
    #     else:
    #         runner.run()

    # def test_tarmac(self):
    #     runner = get_runner(method="tarmac", env=env_name, env_id=env_id, parser_args=args)
    #     if is_benchmark:
    #         runner.benchmark()
    #     else:
    #         runner.run()

    def test_vdac_continuous(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = True
        args_continuous.policy = "Gaussian_MAAC_Policy"
        runner = get_runner(method="vdac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_vdac_discrete(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = False
        args_continuous.policy = "Categorical_MAAC_Policy"
        runner = get_runner(method="vdac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_vdn(self):
        runner = get_runner(method="vdn", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_wqmix(self):
        runner = get_runner(method="wqmix", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()


if __name__ == "__main__":
    unittest.main()
