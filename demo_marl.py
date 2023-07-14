import argparse
from xuanpolicy import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Run an MARL demo.")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--env", type=str, default="sc2")
    parser.add_argument("--env-id", type=str, default="8m")
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
