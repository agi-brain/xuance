import os
import socket
import time
from pathlib import Path
import wandb
import argparse
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from xuance import get_arguments
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.common import get_time_string


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: MASAC.")
    parser.add_argument("--method", type=str, default="masac")
    parser.add_argument("--env", type=str, default="mpe")
    parser.add_argument("--env-id", type=str, default="simple_spread_v3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--benchmark", type=int, default=1)
    parser.add_argument("--config", type=str, default="examples/masac/masac_simple_spread_config.yaml")

    return parser.parse_args()


class Runner(object):
    def __init__(self, args):
        # set random seeds
        set_seed(args.seed)

        # prepare directories
        self.args = args
        self.args.agent_name = args.agent
        time_string = get_time_string()
        folder_name = f"seed_{args.seed}_" + time_string
        self.args.model_dir_load = self.args.model_dir
        self.args.model_dir_save = os.path.join(os.getcwd(), self.args.model_dir, folder_name)

        # build environments
        self.envs = make_envs(args)
        self.n_envs = self.envs.num_envs
        self.args.n_agents = self.envs.num_agents

        from xuance.torch.agents import MASAC_Agents
        self.agents = MASAC_Agents(self.args, self.envs)

    def run(self):
        if self.args.test_mode:
            def env_fn():
                args_test = deepcopy(self.args)
                args_test.parallels = args_test.test_episode
                return make_envs(args_test)

            self.render = True
            self.agents.load_model(self.args.model_dir_load)
            self.test(env_fn, n_episodes=self.args.test_episode)
            print("Finish testing.")
        else:
            n_train_episodes = self.args.running_steps // self.episode_length // self.n_envs
            self.train_episode(n_train_episodes)
            print("Finish training.")
            self.agents.save_model("final_train_model.pth")

    def benchmark(self):
        def env_fn():
            args_test = deepcopy(self.args)
            args_test.parallels = args_test.test_episode
            return make_envs(args_test)

        n_train_episodes = self.args.running_steps // self.n_envs
        n_eval_interval = self.args.eval_interval // self.n_envs
        num_epoch = int(n_train_episodes / n_eval_interval)

        test_scores = self.agents.test(env_fn, n_episodes=self.args.test_episode)
        best_scores = {
            "mean": np.mean(test_scores),
            "std": np.std(test_scores),
            "step": self.agents.current_step
        }
        self.agents.save_model("best_model.pth")

        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            self.agents.train(n_steps=n_eval_interval)
            test_scores = self.agents.test(env_fn, n_episodes=self.args.test_episode)

            mean_test_scores = np.mean(test_scores)
            if mean_test_scores > best_scores["mean"]:
                best_scores = {
                    "mean": mean_test_scores,
                    "std": np.std(test_scores),
                    "step": self.agents.current_step
                }
                # save best model
                self.agents.save_model("best_model.pth")

        # end benchmarking
        print("Finish benchmarking.")
        print("Best Score, Mean: ", best_scores["mean"], "Std: ", best_scores["std"])

    def finish(self):
        self.envs.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()


if __name__ == "__main__":
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser,
                         is_test=parser.test)
    runner = Runner(args)

    if args.benchmark:
        runner.benchmark()
    else:
        runner.run()

    runner.finish()
