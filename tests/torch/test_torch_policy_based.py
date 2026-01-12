# Test the policy-based algorithms with PyTorch.

from argparse import Namespace
from copy import deepcopy
from xuance import get_runner
import unittest

run_mode = "benchmark"  # Choices: "train", "test", "benchmark"

running_steps = 10000
eval_interval = 5000
device = 'cuda:0'
args = Namespace(dl_toolbox='torch', device=device,
                 running_steps=running_steps, eval_interval=eval_interval)
env_name = "classic_control"
env_id_continuous = "Pendulum-v1"
env_id_discrete = "CartPole-v1"


class TestValueBaseAlgo(unittest.TestCase):
    
    """A2C"""
    def test_a2c_continuous(self):
        runner = get_runner(algo="a2c", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_a2c_discrete(self):
        runner = get_runner(algo="a2c", env=env_name, env_id=env_id_discrete, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """DDPG"""
    def test_ddpg(self):
        runner = get_runner(algo="ddpg", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)
        
    """MP-DQN"""
    
    """NPG"""
    # def test_npg_continuous(self):
    #     args_npg = deepcopy(args)
    #     args_npg.running_steps = 10
    #     runner = get_runner(algo="npg", env=env_name, env_id=env_id_continuous, parser_args=args_npg)
    #     runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    # def test_npg_discrete(self):
    #     args_npg = deepcopy(args)
    #     args_npg.running_steps = 1000
    #     runner = get_runner(algo="npg", env=env_name, env_id=env_id_discrete, parser_args=args_npg)
    #     runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """P-DQN"""

    """PG"""
    def test_pg_continuous(self):
        runner = get_runner(algo="pg", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_pg_discrete(self):
        runner = get_runner(algo="pg", env=env_name, env_id=env_id_discrete, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """PPG"""
    def test_ppg_continuous(self):
        runner = get_runner(algo="ppg", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_ppg_discrete(self):
        runner = get_runner(algo="ppg", env=env_name, env_id=env_id_discrete, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """PPO"""
    def test_ppo_continuous(self):
        runner = get_runner(algo="ppo", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_ppo_discrete(self):
        runner = get_runner(algo="ppo", env=env_name, env_id=env_id_discrete, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """SAC"""
    def test_sac_continuous(self):
        runner = get_runner(algo="sac", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    def test_sac_discrete(self):
        runner = get_runner(algo="sac", env=env_name, env_id=env_id_discrete, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)

    """SP-DQN"""
    
    """TD3"""
    def test_td3(self):
        runner = get_runner(algo="td3", env=env_name, env_id=env_id_continuous, parser_args=args)
        runner.run(mode=run_mode, running_steps=running_steps, eval_interval=eval_interval)


if __name__ == "__main__":
    unittest.main()
