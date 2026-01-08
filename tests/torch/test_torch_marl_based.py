# Test the MARL-based algorithms with PyTorch.

from argparse import Namespace
from copy import deepcopy
from xuance import get_runner
import unittest

is_benchmark = False

n_steps = 10000
eval_interval = 5000
device = 'cuda:0'
test_mode = False
args = Namespace(dl_toolbox='torch', device=device,
                 running_steps=n_steps, eval_interval=eval_interval, test_mode=test_mode)
env_name = "mpe"
env_id = "simple_spread_v3"


class TestValueBaseAlgo(unittest.TestCase):
    def test_coma(self):
        runner = get_runner(algo="coma", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_commnet(self):
        runner = get_runner(algo="commnet", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    # def test_dcg(self):
    #     runner = get_runner(algo="dcg", env=env_name, env_id=env_id, parser_args=args)
    #     if is_benchmark:
    #         runner.benchmark()
    #     else:
    #         runner.run()

    def test_iac_continuous(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = True
        args_continuous.policy = "Gaussian_MAAC_Policy"
        runner = get_runner(algo="iac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_iac_discrete(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = False
        args_continuous.policy = "Categorical_MAAC_Policy"
        runner = get_runner(algo="iac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_ic3net(self):
        runner = get_runner(algo="ic3net", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_iddpg(self):
        runner = get_runner(algo="iddpg", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_ippo_continuous(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = True
        args_continuous.policy = "Gaussian_MAAC_Policy"
        runner = get_runner(algo="ippo", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_ippo_discrete(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = False
        args_continuous.policy = "Categorical_MAAC_Policy"
        runner = get_runner(algo="ippo", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_iql(self):
        runner = get_runner(algo="iql", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_isac(self):
        runner = get_runner(algo="isac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_maddpg(self):
        runner = get_runner(algo="maddpg", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_mappo_continuous(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = True
        args_continuous.policy = "Gaussian_MAAC_Policy"
        runner = get_runner(algo="mappo", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_mappo_discrete(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = False
        args_continuous.policy = "Categorical_MAAC_Policy"
        runner = get_runner(algo="mappo", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_masac(self):
        runner = get_runner(algo="masac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_itd3(self):
        runner = get_runner(algo="itd3", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_matd3(self):
        runner = get_runner(algo="matd3", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_mfac(self):
        runner = get_runner(algo="mfac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_mfq(self):
        runner = get_runner(algo="mfq", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_qmix(self):
        runner = get_runner(algo="qmix", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_qtran(self):
        runner = get_runner(algo="qtran", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_tarmac(self):
        runner = get_runner(algo="tarmac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_vdac_continuous(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = True
        args_continuous.policy = "Gaussian_MAAC_Policy"
        runner = get_runner(algo="vdac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_vdac_discrete(self):
        args_continuous = deepcopy(args)
        args_continuous.continuous_action = False
        args_continuous.policy = "Categorical_MAAC_Policy"
        runner = get_runner(algo="vdac", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_vdn(self):
        runner = get_runner(algo="vdn", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()

    def test_wqmix(self):
        runner = get_runner(algo="wqmix", env=env_name, env_id=env_id, parser_args=args)
        if is_benchmark:
            runner.benchmark()
        else:
            runner.run()


if __name__ == "__main__":
    unittest.main()
