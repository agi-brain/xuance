import argparse
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Run an MARL demo.")
    parser.add_argument("--method", type=str, default="vdac")
    parser.add_argument("--env", type=str, default="sc2")
    parser.add_argument("--env-id", type=str, default="3m")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--mixer", type=str, default="VDN")
    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    runner = get_runner(method=parser.method,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser,
                        is_test=parser.test)
    runner.benchmark()
