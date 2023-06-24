import argparse
from xuanpolicy import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Multi-Agent Reinforcement Learning With Causality Detection.")
    parser.add_argument("--method", type=str, default="drqn")
    parser.add_argument("--env", type=str, default="box2d")
    parser.add_argument("--env-id", type=str, default="CarRacing-v2")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    runner = get_runner(method=parser.method,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser,
                        is_test=parser.test)
    runner.run()
