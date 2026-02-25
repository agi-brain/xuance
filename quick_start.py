import xuance, argparse


def parse_args():
    parser = argparse.ArgumentParser("Run a demo.")
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--env", type=str, default="classic_control")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    runner = xuance.get_runner(algo=parser.algo,
                               env=parser.env,
                               env_id=parser.env_id,
                               parser_args=parser)
    runner.run(mode="train")
