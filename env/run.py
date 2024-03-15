import argparse
from xuance import get_runner
import tensorboard


def parse_args():
    parser = argparse.ArgumentParser("Run a demo.")
    parser.add_argument("--method", type=str, default="dqn")
    parser.add_argument("--env", type=str, default="snake_env")
    parser.add_argument("--env-id", type=str, default="snake_env-v0")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config", type=str, default="./test.yaml")

    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    runner = get_runner(method=parser.method,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser,
                        config_path=parser.config,
                        is_test=True)
    runner.run()