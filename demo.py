import argparse
from xuanpolicy import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Multi-Agent Reinforcement Learning With Causality Detection.")
    parser.add_argument("--method", type=str, default="ddpg")
    parser.add_argument("--env", type=str, default="mujoco/Hopper")
    parser.add_argument("--test", type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    runner = get_runner(method=parser.method,
                        env=parser.env,
                        parser_args=parser,
                        is_test=parser.test)
    runner.run()
