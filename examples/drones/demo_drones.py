import argparse
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Run a demo.")
    parser.add_argument("--method", type=str, default="iddpg")
    parser.add_argument("--env", type=str, default="drones")
    parser.add_argument("--env-id", type=str, default="MultiHoverAviary")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--parallels", type=int, default=10)
    parser.add_argument("--benchmark", type=int, default=1)
    parser.add_argument("--test-episode", type=int, default=5)

    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    ########################################################################
    # You can also modify the arguments not listed above here. For example:
    # parser.parallels = 1
    # parser.running_steps = 1000000
    # ...
    ########################################################################
    runner = get_runner(method=parser.method,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser,
                        is_test=parser.test)
    if parser.benchmark:
        runner.benchmark()
    else:
        runner.run()
