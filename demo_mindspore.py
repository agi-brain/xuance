import argparse
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Run a demo.")
    parser.add_argument("--method", type=str, default="ddpg")
    parser.add_argument("--env", type=str, default="classic_control")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="GPU")
    parser.add_argument("--dl_toolbox", type=str, default="mindspore")
    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    runner = get_runner(method=parser.method,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser,
                        is_test=parser.test)
    runner.run()
