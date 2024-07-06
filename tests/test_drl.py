from argparse import Namespace
from xuance import get_runner
import unittest


class TestQLearningBaseAlgo(unittest.TestCase):
    def test_dqn(self):
        args = Namespace(**dict(parallels=2, running_steps=10000))
        runner = get_runner(method="dqn", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()

if __name__ == "__main__":
    unittest.main()
