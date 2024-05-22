import copy

from xuance.torch.runners.runner_basic import Runner_Base
from xuance.torch.agents import REGISTRY_Agents

import argparse
from xuance import get_arguments
import time


def parse_args():
    parser = argparse.ArgumentParser("Run benchmark results for MARL.")
    parser.add_argument("--method", type=str, default="iddpg")
    parser.add_argument("--env", type=str, default="mpe")
    parser.add_argument("--env-id", type=str, default="simple_spread_v3")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


class Runnner_MARL_Base(Runner_Base):
    def __init__(self, config):
        super(Runnner_MARL_Base, self).__init__(config)
        config.n_agents = self.envs.n_agents_all
        self.agents = REGISTRY_Agents[config.agent](config, self.envs)
        self.agents.train(1000)
        self.envs.close()


if __name__ == '__main__':
    parser = parse_args()
    parser.render = True
    parser.render_mode = "human"
    # parser.parallels = 1
    config = get_arguments(method=parser.method,
                           env=parser.env,
                           env_id=parser.env_id,
                           parser_args=parser,
                           is_test=parser.test)
    runner = Runnner_MARL_Base(config)
    runner.run()
