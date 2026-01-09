import argparse
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Run a Benchmark.")
    parser.add_argument("--algo", type=str, default="dqn")
    parser.add_argument("--env", type=str, default="classic_control")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config-path", type=str, default="./config")
    parser.add_argument("--result-path", type=str, default="./result")

    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    parser.env_seed = parser.seed
    runner = get_runner(algo=parser.algo,
                        env=parser.env,
                        env_id=parser.env_id,
                        config_path=parser.config_path,
                        parser_args=parser)
    runner.run(mode='benchmark',
               benchmark_result_path=f"{parser.result_path}")

